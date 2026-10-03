"""SM89 FlashAttention 2 的 TileLang 框架：BF16、Q64/KV64、128 threads。

当前只定义接口、block 映射和工作区；尚未加载 Q/K/V、计算或写入 output。
make_flash_attention2() 仅构建 PrimFunc，供逐步实现和检查 IR，不自动 JIT/launch。
"""

import tilelang.language as T


Q_TILE = 64
KV_TILE = 64
HEAD_NUM = 32
KV_HEAD_NUM = 8
HEAD_DIM = 128
THREADS = 128
DTYPE = "bfloat16"
ACCUM_DTYPE = "float32"


def make_flash_attention2(num_pages: int, scheduler_q_tile: int = Q_TILE):
    """返回 attention 的 PrimFunc；两个参数在构建 IR 时确定。

    num_pages 是物理 KV pool 的页数；每页 1 token。
    scheduler_q_tile=64 对应独立算子测试，128 对应 RustInfer 的 tile map。
    b1 是调优目标，batch 和 packed Q 总长度 num_tokens 均为运行时参数。

    保留 CuTe 版本的设备输入语义；TileLang lowering 后的 CUDA ABI 需另行校验。
    stream 属于 host launcher，故不放进 device 函数参数。
    调用方须保证 block_size=1、head_num=32、kv_head_num=8、head_dim=128；
    Q/O 的最后一维连续，row/head strides 以元素为单位且为 8 的倍数。
    """
    if num_pages <= 0:
        raise ValueError("num_pages must be positive")
    if scheduler_q_tile not in (64, 128):
        raise ValueError("scheduler_q_tile must be 64 or 128")

    @T.prim_func
    def flash_attention2_bf16_b1_hq32_hkv8_d128_q64_kv64_bs1(
        q: T.handle,                 # BF16 [num_tokens, 32, 128]
        k_pool: T.handle,            # BF16 [num_pages, 1, 8, 128]
        v_pool: T.handle,            # BF16 [num_pages, 1, 8, 128]
        output: T.handle,            # BF16 [num_tokens, 32, 128]
        block_tables: T.handle,      # uint32 [batch, max_blocks_per_seq]
        kv_lens: T.handle,           # int32 [batch]
        cu_q_lens: T.handle,         # int32 [batch + 1]
        block2req: T.handle,         # int32 [total_q_tiles]
        block2tile: T.handle,        # int32 [total_q_tiles]
        valid_q_tiles: T.handle,     # int32 [1]，GPU 上的实际 tile 数
        q_stride_seq: T.int64,
        q_stride_head: T.int64,
        o_stride_seq: T.int64,
        o_stride_head: T.int64,
        max_blocks_per_seq: T.int32,
        block_size: T.int32,
        total_q_tiles: T.int32,      # launch 容量，允许 CUDA Graph padding
        batch: T.int32,
        num_tokens: T.int32,         # 所有请求的 Q 行数之和，不是单个请求长度
        head_num: T.int32,
        kv_head_num: T.int32,
        head_dim: T.int32,
        scale: T.float32,            # 通常为 1 / sqrt(128)
        causal: T.bool,
    ):
        # GMEM 视图；显式 stride 支持融合 QKV 中的非连续 Q。
        Q = T.match_buffer(
            q, (num_tokens, HEAD_NUM, HEAD_DIM), DTYPE,
            strides=(q_stride_seq, q_stride_head, 1), align=16,
        )
        K = T.match_buffer(k_pool, (num_pages, 1, KV_HEAD_NUM, HEAD_DIM), DTYPE, align=16)
        V = T.match_buffer(v_pool, (num_pages, 1, KV_HEAD_NUM, HEAD_DIM), DTYPE, align=16)
        O = T.match_buffer(
            output, (num_tokens, HEAD_NUM, HEAD_DIM), DTYPE,
            strides=(o_stride_seq, o_stride_head, 1), align=16,
        )
        # Mixed prefill suffix 的 metadata 指针可能只满足 4-byte alignment。
        Pages = T.match_buffer(block_tables, (batch, max_blocks_per_seq), "uint32", align=4)
        KvLens = T.match_buffer(kv_lens, (batch,), "int32", align=4)
        CuQ = T.match_buffer(cu_q_lens, (batch + 1,), "int32", align=4)
        TileRequests = T.match_buffer(block2req, (total_q_tiles,), "int32", align=4)
        TileIndices = T.match_buffer(block2tile, (total_q_tiles,), "int32", align=4)
        ActiveTiles = T.match_buffer(valid_q_tiles, (1,), "int32", align=4)

        # 每个 CTA 负责一个请求的一个 Q64 tile、一个 Q head。
        with T.Kernel(
            total_q_tiles * (scheduler_q_tile // Q_TILE), HEAD_NUM, threads=THREADS,
        ) as (block, head):
            # 计划中的共享内存：Q/K/V 各 16 KiB，共 48 KiB。
            # 这里只声明，尚无搬运；空框架 lowering 时可能被优化掉。
            q_shared = T.alloc_shared((Q_TILE, HEAD_DIM), DTYPE)
            k_shared = T.alloc_shared((KV_TILE, HEAD_DIM), DTYPE)
            v_shared = T.alloc_shared((KV_TILE, HEAD_DIM), DTYPE)

            # Fragment 是整个 CTA 的逻辑 tile，由编译器分配到各线程寄存器。
            scores = T.alloc_fragment((Q_TILE, KV_TILE), ACCUM_DTYPE)
            probs = T.alloc_fragment((Q_TILE, KV_TILE), DTYPE)
            output_acc = T.alloc_fragment((Q_TILE, HEAD_DIM), ACCUM_DTYPE)
            row_max = T.alloc_fragment((Q_TILE,), ACCUM_DTYPE)
            row_sum = T.alloc_fragment((Q_TILE,), ACCUM_DTYPE)

            tile = block // (scheduler_q_tile // Q_TILE)
            sub_tile = block % (scheduler_q_tile // Q_TILE)
            # 先排除 graph padding，再读 tile map，避免访问无效 metadata。
            if tile < ActiveTiles[0]:
                req = TileRequests[tile]
                q_begin = CuQ[req]
                q_end = CuQ[req + 1]
                local_q_start = TileIndices[tile] * scheduler_q_tile + sub_tile * Q_TILE
                q_start = q_begin + local_q_start
                if q_start < q_end:
                    q_len = q_end - q_begin
                    kv_len = KvLens[req]
                    kv_head = head // (HEAD_NUM // KV_HEAD_NUM)

                    # TODO 1: Q -> q_shared；处理请求尾部，加入异步搬运/等待。
                    # TODO 2: 初始化 row_max=-inf、row_sum=0、output_acc=0。
                    # TODO 3: KV tile 循环，按 Pages[req, logical_token] 读页号。
                    #   - 分别搬运 K/V，安排 load 与计算的流水。
                    #   - scores = Q @ K^T，乘 scale，并应用 tail/causal mask。
                    #     causal 底右对齐：key <= kv_len - q_len + local_q。
                    #   - online softmax，更新 row_max/row_sum 并重缩放 output_acc。
                    #   - probs 转 BF16，output_acc += probs @ V（FP32 累加）。
                    # TODO 4: output_acc / row_sum，转 BF16 写 O；空 KV 行写零。
                    T.evaluate(0)

    return flash_attention2_bf16_b1_hq32_hkv8_d128_q64_kv64_bs1
