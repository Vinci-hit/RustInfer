"""SM89 FlashAttention 2: B=1 tuned, ragged batches, BF16, 32 Q heads, D=128.

Separately pipelines paged K/V loads with BF16 QK/PV MMA and FP32 online softmax.
Includes Tensor Core PV and normalized BF16 output. compile_attention.py provides
the optional RustInfer SM89 AOT backend with 128-row scheduler metadata.
"""

import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import cpasync, warp


Q_TILE = 64
KV_TILE = 64
HEAD_DIM = 128
THREADS = 128


@cute.jit
def load_q_tile_async(
    q_tensor: cute.Tensor,         # GMEM BF16 [S, 32, 128], stride=(6144, 128, 1)
    shared_q: cute.Tensor,  # SMEM BF16 [64, 128], stride=(128, 1), 16 KiB
    row_start: cutlass.Int32,  # cu_q_lens[req] + local_tile * 64
    row_end: cutlass.Int32,    # cu_q_lens[req + 1]; never cross into another request
    head: cutlass.Int32,       # Query head index in [0, 32)
):
    """Issue cp.async for BF16 [64, 128]; caller commits and waits.

    Requires 128 threads, unit column stride and 16-byte aligned row/head
    addresses (including q_tensor.iterator). Each copy moves eight BF16 elements.
    Tail rows are zero-filled. This helper never launches a separate kernel
    and does not wait or synchronize the block.
    """
    tid, _, _ = cute.arch.thread_idx()
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128,
    )
    for offset in cutlass.range(Q_TILE * HEAD_DIM // (THREADS * 8), unroll=1):
        element = (tid + offset * THREADS) * 8
        row = element // HEAD_DIM
        col = element % HEAD_DIM
        if row_start + row < row_end:
            src_offset = ((row_start + row) * q_tensor.stride[0]
                          + head * q_tensor.stride[1] + col)
            src_offset = cute.assume(src_offset, divby=8)
            src = cute.make_tensor(q_tensor.iterator + src_offset, cute.make_layout((8,)))
            dst = cute.make_tensor(
                shared_q.iterator + cute.assume(element, divby=8),
                cute.make_layout((8,)),
            )
            cute.copy(copy_atom, src, dst)
        else:
            for i in cutlass.range_constexpr(8):
                shared_q[row, col + i] = cutlass.BFloat16(0)


@cute.jit
def load_paged_tile_async(
    pool: cute.Pointer,  # BF16 [physical_pages, 1, 8, 128]
    cached_page: cutlass.Int32,  # one page ID per lane; eight IDs distributed per half-warp
    shared: cute.Tensor,  # BF16 [64, 128], one reusable slot
    kv_head: cutlass.Int32,
    kv_start: cutlass.Int32,
    kv_len: cutlass.Int32,
):
    """Issue a K OR V tile from cached page IDs; no page-table reads here.

    Page size is one. Sixteen lanes copy one row; lanes 0..7 of each
    half-warp cache its eight row IDs (lanes 8..15 mirror them). Shuffle
    before the bounds branch so every lane participates, including tails.
    One lookahead page load per thread/tile is shared by both K and V.
    """
    tid, _, _ = cute.arch.thread_idx()
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16,
                               num_bits_per_copy=128)
    for offset in cutlass.range(KV_TILE * HEAD_DIM // (THREADS * 8), unroll=1):
        element = (tid + offset * THREADS) * 8
        row = element // HEAD_DIM
        col = element % HEAD_DIM
        token = kv_start + row
        page_id = cute.arch.shuffle_sync(cached_page, (tid % 32 // 16) * 16 + offset)
        if token < kv_len:
            page = page_id.to(cutlass.Uint32).to(cutlass.Int64)
            src_offset = (page * 8 + kv_head) * HEAD_DIM + col
            src_offset = cute.assume(src_offset, divby=8)
            dst_offset = cute.assume(element, divby=8)
            src = cute.make_tensor(pool + src_offset, cute.make_layout((8,)))
            dst = cute.make_tensor(shared.iterator + dst_offset, cute.make_layout((8,)))
            cute.copy(atom, src, dst)
        else:
            for i in cutlass.range_constexpr(8):
                shared[row, col + i] = cutlass.BFloat16(0)


@cute.jit
def q_tile_bounds(
    tile: cutlass.Int32,
    cu_q_lens: cute.Tensor,  # int32 [batch + 1]
    block2req: cute.Tensor,  # int32 [total_q_tiles]
    block2tile: cute.Tensor,  # int32 [total_q_tiles]
):
    """Resolve an active packed tile into its request and global Q row bounds."""
    req = block2req[tile]
    row_start = cu_q_lens[req] + block2tile[tile] * Q_TILE
    row_end = cu_q_lens[req + 1]
    return req, row_start, row_end


@cute.kernel
def flash_attention2_bf16_b1_hq32_hkv8_d128_q64_kv64_bs1(
    q: cute.Pointer,
    k_pool: cute.Pointer,
    v_pool: cute.Pointer,
    output: cute.Pointer,
    block_tables: cute.Pointer,
    kv_lens: cute.Pointer,
    cu_q_lens: cute.Pointer,
    block2req: cute.Pointer,
    block2tile: cute.Pointer,
    valid_q_tiles: cute.Pointer,
    q_stride_seq: cutlass.Int64,
    q_stride_head: cutlass.Int64,
    o_stride_seq: cutlass.Int64,
    o_stride_head: cutlass.Int64,
    max_blocks_per_seq: cutlass.Int32,
    block_size: cutlass.Int32,
    total_q_tiles: cutlass.Int32,
    batch: cutlass.Int32,
    num_tokens: cutlass.Int32,
    head_num: cutlass.Int32,
    kv_head_num: cutlass.Int32,
    head_dim: cutlass.Int32,
    scale: cutlass.Float32,
    causal: cutlass.Boolean,
    stream,  # Reserved compatibility argument; launch stream belongs to .launch().
    scheduler_q_tile: cutlass.Constexpr = 64,  # 128 for RustInfer scheduler metadata.
):
    """Attention kernel: separate K/V prefetch, online softmax and PV.

    Kernel name encodes the dispatch specialization:
      bf16 = data type; b1 = tuning target, NOT a batch-size restriction; hq32/hkv8 = Q/KV head counts;
      d128 = head dimension; q64/kv64 = tile sizes; bs1 = one token per KV page.
    Sequence length num_tokens remains dynamic, including 128/512/896/2048.
    Pointer shapes (all strides count elements):
      q/output: BF16 [num_tokens, head_num, head_dim]
      k_pool/v_pool: BF16 [physical_pages, block_size, kv_head_num, head_dim]
      block_tables: uint32 [batch, max_blocks_per_seq]
      kv_lens: int32 [batch]; cu_q_lens: int32 [batch + 1]
      block2req/block2tile: int32 [total_q_tiles]; valid_q_tiles: int32 [1]
    Original runtime arguments are retained; scheduler_q_tile is compile-time only.
    stream is retained for interface compatibility; the actual CUDA launch
    stream must be supplied to .launch(stream=stream) by the caller.

    Current specialization:
      batch is dynamic; Q packs all requests along its first axis
      Q dtype = BF16; shape = [S, 32, 128]
      Q strides = [6144, 128, 1] elements for a fused QKV view
      Q base address = 16-byte aligned (declare when constructing the tensor)
      grid = (total_q_tiles * (scheduler_q_tile // 64), 32, 1)
      block = (128, 1, 1)
      shared Q: BF16 [64, 128], 16 KiB / block
      shared K and V each: BF16 [64, 128], 16 KiB / block
      P stays in registers: FP32 probabilities converted to BF16 for PV
      total shared memory = 48 KiB; launcher must support opt-in shared memory

    For B=1, S=896 gives Q=[896,32,128] and grid=(14,32,1).
    Tile maps use scheduler_q_tile rows independently for each request:
      active tiles = sum(ceil(query_length[req] / scheduler_q_tile)).
    The default is 64. RustInfer AOT uses 128 and splits each scheduler tile
    into two Q64 CTAs; an empty second half exits before touching K/V.
    Metadata pointers require only 4-byte alignment, including shifted suffixes.
    total_q_tiles is capacity; valid_q_tiles[0] is the active device count.
    Inactive capacity entries are never read (supports graph replay padding).
    Other row/head strides divisible by 8 elements are also accepted.
    Fully causal-masked KV tiles are skipped; boundary tiles retain element masks.
    Q/KV tails are zero-padded; causal/tail masks, online softmax and PV are
    implemented. P is rounded to BF16 for PV; output is normalized in FP32 and
    stored as BF16. Fully masked/empty-KV rows produce zero output.
    Launch this kernel with .launch(...) from a @cute.jit entry point.
    """
    assert q.value_type == cutlass.BFloat16
    assert k_pool.value_type == cutlass.BFloat16
    assert v_pool.value_type == cutlass.BFloat16
    assert output.value_type == cutlass.BFloat16
    # Pointer lowering may represent uint32 metadata with an int32 pointee.
    assert block_tables.value_type in (cutlass.Uint32, cutlass.Int32)
    assert kv_lens.value_type == cutlass.Int32
    assert cu_q_lens.value_type == cutlass.Int32
    assert block2req.value_type == cutlass.Int32
    assert block2tile.value_type == cutlass.Int32
    assert valid_q_tiles.value_type == cutlass.Int32
    # Current implementation fixes Hq=32, Hkv=8, D=128; batch is dynamic.
    # block_size MUST be 1; retained as an ABI parameter, not used in addressing.
    # These scalar preconditions are the launcher's responsibility.
    q_tensor = cute.make_tensor(
        q,
        cute.make_layout(
            (num_tokens, 32, HEAD_DIM),
            stride=(q_stride_seq, q_stride_head, 1),
        ),
    )

    cu_q = cute.make_tensor(cu_q_lens, cute.make_layout((batch + 1,)))
    tile_requests = cute.make_tensor(block2req, cute.make_layout((total_q_tiles,)))
    tile_indices = cute.make_tensor(block2tile, cute.make_layout((total_q_tiles,)))
    active_tiles = cute.make_tensor(valid_q_tiles, cute.make_layout((1,)))
    block, head, _ = cute.arch.block_idx()
    assert scheduler_q_tile == 64 or scheduler_q_tile == 128
    tile = block // (scheduler_q_tile // Q_TILE)
    sub_tile = block % (scheduler_q_tile // Q_TILE)
    kv_lengths = cute.make_tensor(kv_lens, cute.make_layout((batch,)))
    pages = cute.make_tensor(block_tables, cute.make_layout(
        (batch, max_blocks_per_seq), stride=(max_blocks_per_seq, 1)))
    allocator = SmemAllocator()
    # Iterator swizzles operate on BYTE addresses. Q/K/V rows are 256 bytes:
    # XOR row bits [8:10] into 16-byte chunk bits [4:6], preserving each
    # contiguous 16-byte cp.async copy. P stays entirely in registers.
    # Keeping swizzle on the iterator makes pointer-offset copies, tensor
    # indexing and the transposed V view use the same mapping.
    qkv_swizzle = cute.make_swizzle(3, 4, 4)
    shared_q = allocator.allocate_tensor(
        cutlass.BFloat16,  # BF16: 2 bytes per element
        cute.make_layout((Q_TILE, HEAD_DIM), stride=(HEAD_DIM, 1)),  # [64, 128]
        byte_alignment=1024, swizzle=qkv_swizzle,
    )
    kv_layout = cute.make_layout(
        (KV_TILE, HEAD_DIM), stride=(HEAD_DIM, 1))
    shared_k = allocator.allocate_tensor(cutlass.BFloat16, kv_layout, byte_alignment=1024, swizzle=qkv_swizzle)
    shared_v = allocator.allocate_tensor(cutlass.BFloat16, kv_layout, byte_alignment=1024, swizzle=qkv_swizzle)
    # Block-uniform condition: all threads either participate or skip.
    if tile < active_tiles[0]:
        req = tile_requests[tile]
        row_start = cu_q[req] + tile_indices[tile] * scheduler_q_tile + sub_tile * Q_TILE
        row_end = cu_q[req + 1]
        if row_start < row_end:
            kv_head = head // 4  # GQA: 32 query heads / 8 KV heads
            kv_len = kv_lengths[req]
            # Bottom-right causal alignment: a global packed Q row r may attend
            # through key position kv_len - row_end + r. Use the last VALID row
            # of this Q tile to obtain an exclusive bound for the entire block.
            kv_limit = kv_len
            if causal:
                q_tile_end = min(row_start + Q_TILE, row_end)
                kv_limit = min(kv_len, max(0, kv_len - row_end + q_tile_end))
            # kv_limit=0 means every row in this block is fully masked. No K/V
            # loads or MMA/softmax iterations are needed; final output is zero.

            tid, _, _ = cute.arch.thread_idx()
            # This lane caches one of the eight rows copied by its half-warp.
            # Prime K0/V0 before Q loading; all lanes participate in later shuffles.
            page_row = tid // 16 + (tid % 8) * 8
            cached_page = cutlass.Int32(0)
            if page_row < kv_limit:
                cached_page = pages[req, page_row].to(cutlass.Int32)
            load_q_tile_async(q_tensor, shared_q, row_start, row_end, head)
            if kv_limit > 0:
                load_paged_tile_async(k_pool, cached_page, shared_k,
                                      kv_head, 0, kv_limit)
            cute.arch.cp_async_commit_group()  # Q + K_0; V is issued in the loop

            # Tensor Core operand mapping: 4 warps along M, each owns 16 Q rows.
            # One MMA instruction computes [16,8,16], with BF16 inputs / FP32 sum.
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
                atom_layout_mnk=(4, 1, 1),
            )
            tid, _, _ = cute.arch.thread_idx()
            thr_mma = tiled_mma.get_slice(tid)

            # Two register slots allow the next ldmatrix loads to precede current MMA.
            q_first = cute.local_tile(shared_q, (Q_TILE, 16), (0, 0))
            k_first = cute.local_tile(shared_k, (KV_TILE, 16), (0, 0))
            q_regs = thr_mma.make_fragment_A(thr_mma.partition_A(q_first))
            k_regs = thr_mma.make_fragment_B(thr_mma.partition_B(k_first))
            q_fragments = (q_regs, thr_mma.make_fragment_A(thr_mma.partition_A(q_first)))
            k_fragments = (k_regs, thr_mma.make_fragment_B(thr_mma.partition_B(k_first)))
            score_coords = thr_mma.partition_C(cute.make_identity_tensor((Q_TILE, KV_TILE)))
            scores = cute.make_rmem_tensor(score_coords.shape, cutlass.Float32)
            q_ldmatrix = cute.make_tiled_copy_A(cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                cutlass.BFloat16), tiled_mma)
            k_ldmatrix = cute.make_tiled_copy_B(cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
                cutlass.BFloat16), tiled_mma)
            q_loader = q_ldmatrix.get_slice(tid)
            k_loader = k_ldmatrix.get_slice(tid)
            # Each lane owns two rows; four adjacent lanes cooperate on each row.
            # m/l are replicated across those four lanes and persist across KV tiles.
            m = cute.make_rmem_tensor((2,), cutlass.Float32)
            l = cute.make_rmem_tensor((2,), cutlass.Float32)
            alpha = cute.make_rmem_tensor((2,), cutlass.Float32)
            m.fill(float("-inf"))
            l.fill(0.0)
            output_coords = thr_mma.partition_C(cute.make_identity_tensor((Q_TILE, HEAD_DIM)))
            output_acc = cute.make_rmem_tensor(output_coords.shape, cutlass.Float32)
            output_acc.fill(0.0)
            # CuTe B uses [N,K]. V physically stores [key_token, output_dim], so
            # this view is [output_dim, key_token] with the first dimension contiguous.
            v_view = cute.make_tensor(shared_v.iterator,
                cute.make_layout((HEAD_DIM, KV_TILE), stride=(1, HEAD_DIM)))
            v_first = cute.local_tile(v_view, (HEAD_DIM, 16), (0, 0))
            p_regs = cute.make_rmem_tensor(q_regs.shape, cutlass.BFloat16)
            v_regs = thr_mma.make_fragment_B(thr_mma.partition_B(v_first))
            v_fragments = (v_regs, thr_mma.make_fragment_B(thr_mma.partition_B(v_first)))
            v_ldmatrix = cute.make_tiled_copy_B(cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2),
                cutlass.BFloat16), tiled_mma)
            v_loader = v_ldmatrix.get_slice(tid)
            # Keep online maxima in log2 units, so exp2 needs no per-value conversion.
            scale_log2 = scale * cutlass.Float32(1.4426950408889634)
            num_kv_tiles = cute.ceil_div(kv_limit, KV_TILE)
            for kv_tile in range(num_kv_tiles):
                # V_i can transfer while QK_i executes. The older group contains
                # Q + K_0 on the first iteration, and K_i on later iterations.
                load_paged_tile_async(v_pool, cached_page, shared_v,
                                      kv_head, kv_tile * KV_TILE, kv_limit)
                cute.arch.cp_async_commit_group()  # V_i
                # V_i consumed the old IDs; look ahead to K_(i+1)/V_(i+1).
                # QK_i separates this LDG from its first use as a KV address.
                if kv_tile + 1 < num_kv_tiles:
                    token = (kv_tile + 1) * KV_TILE + page_row
                    cached_page = cutlass.Int32(0)
                    if token < kv_limit:
                        cached_page = pages[req, token].to(cutlass.Int32)
                cute.arch.cp_async_wait_group(1)   # Q/K_i ready; V_i may be pending
                cute.arch.sync_threads()
                k_current = shared_k  # [N,D], no physical transpose
                scores.fill(0.0)  # New score block for each KV tile, not across tiles

                # Prime slot 0. ldmatrix is not cp.async: dependency scoreboarding
                # protects the reads; alternate register slots expose independent work.
                q_slice = cute.local_tile(shared_q, (Q_TILE, 16), (0, 0))
                k_slice = cute.local_tile(k_current, (KV_TILE, 16), (0, 0))
                cute.copy(q_ldmatrix, q_loader.partition_S(q_slice), q_loader.retile(q_fragments[0]))
                cute.copy(k_ldmatrix, k_loader.partition_S(k_slice), k_loader.retile(k_fragments[0]))
                # Two steps per loop iteration keep the two slots statically indexed
                # without fully expanding all eight D slices (which increased spills).
                for pair in cutlass.range(HEAD_DIM // 32, unroll=1):
                    q_next = cute.local_tile(shared_q, (Q_TILE, 16), (0, pair * 2 + 1))
                    k_next = cute.local_tile(k_current, (KV_TILE, 16), (0, pair * 2 + 1))
                    cute.copy(q_ldmatrix, q_loader.partition_S(q_next), q_loader.retile(q_fragments[1]))
                    cute.copy(k_ldmatrix, k_loader.partition_S(k_next), k_loader.retile(k_fragments[1]))
                    cute.gemm(tiled_mma, scores, q_fragments[0], k_fragments[0], scores)
                    if pair + 1 < HEAD_DIM // 32:
                        q_next = cute.local_tile(shared_q, (Q_TILE, 16), (0, pair * 2 + 2))
                        k_next = cute.local_tile(k_current, (KV_TILE, 16), (0, pair * 2 + 2))
                        cute.copy(q_ldmatrix, q_loader.partition_S(q_next), q_loader.retile(q_fragments[0]))
                        cute.copy(k_ldmatrix, k_loader.partition_S(k_next), k_loader.retile(k_fragments[0]))
                    cute.gemm(tiled_mma, scores, q_fragments[1], k_fragments[1], scores)

                # All warps must finish reading K_i before any overwrites its slot.
                cute.arch.sync_threads()
                # Reuse the single K slot for K_(i+1) during softmax and PV_i.
                if kv_tile + 1 < num_kv_tiles:
                    load_paged_tile_async(k_pool, cached_page, shared_k,
                                          kv_head, (kv_tile + 1) * KV_TILE, kv_limit)
                # Commit even on the last iteration so wait_group(1) waits for V_i.
                cute.arch.cp_async_commit_group()  # K_(i+1), or an empty final group

                # scores now holds Q[64,128] @ K_tile[64,128]^T in FP32.
                # MMA C has 4 values per m16n8 atom per lane: two columns in
                # row 0 and two in row 1. Eight N atoms cover all 64 key columns.
                # This decision is uniform across the CTA. Interior tiles need
                # scaling only; diagonal, Q-tail and KV-tail tiles retain full masks.
                full_tile = ((row_start + Q_TILE <= row_end)
                             & ((kv_tile + 1) * KV_TILE <= kv_len))
                if causal:
                    full_tile = full_tile & ((kv_tile + 1) * KV_TILE - 1
                                            <= kv_len - row_end + row_start)
                if full_tile:
                    for i in cutlass.range_constexpr(cute.size(scores)):
                        scores[i] = scores[i] * scale_log2
                else:
                    for i in cutlass.range_constexpr(cute.size(scores)):
                        coord = score_coords[i]
                        q_row = row_start + coord[0]
                        key_pos = kv_tile * KV_TILE + coord[1]
                        valid = (q_row < row_end) & (key_pos < kv_len)
                        if causal:
                            valid = valid & (key_pos <= kv_len - row_end + q_row)
                        value = cutlass.Float32(float("-inf"))
                        if valid:
                            value = scores[i] * scale_log2
                        scores[i] = value
                # m and scores are now log2-scaled. Native fmax avoids the generic
                # compare/select expansion; nan=True retains NaN propagation.
                for row_slot in cutlass.range_constexpr(2):
                    tile_max = cutlass.Float32(float("-inf"))
                    for n in cutlass.range_constexpr(KV_TILE // 8):
                        for j in cutlass.range_constexpr(2):
                            i = n * 4 + row_slot * 2 + j
                            tile_max = cute.arch.fmax(tile_max, scores[i], nan=True)
                    tile_max = cute.arch.warp_reduction_max(tile_max, threads_in_group=4)
                    new_m = cute.arch.fmax(m[row_slot], tile_max, nan=True)
                    # Empty/fully masked rows must not evaluate -inf - (-inf).
                    exp_base = cutlass.Float32(0.0)
                    if new_m != cutlass.Float32(float("-inf")):
                        exp_base = new_m
                    correction = cutlass.Float32(0.0)
                    if m[row_slot] != cutlass.Float32(float("-inf")):
                        correction = cute.math.exp2(m[row_slot] - exp_base, approx=True)
                    alpha[row_slot] = correction
                    tile_sum = cutlass.Float32(0.0)
                    for n in cutlass.range_constexpr(KV_TILE // 8):
                        for j in cutlass.range_constexpr(2):
                            i = n * 4 + row_slot * 2 + j
                            probability = cute.math.exp2(scores[i] - exp_base, approx=True)
                            scores[i] = probability
                            tile_sum = tile_sum + probability
                    tile_sum = cute.arch.warp_reduction_sum(tile_sum, threads_in_group=4)
                    l[row_slot] = correction * l[row_slot] + tile_sum
                    m[row_slot] = new_m

                for i in cutlass.range_constexpr(cute.size(output_acc)):
                    row_slot = (i % 4) // 2
                    output_acc[i] = output_acc[i] * alpha[row_slot]

                cute.arch.cp_async_wait_group(1)  # V_i ready; K_(i+1) may be pending
                cute.arch.sync_threads()
                v_current = cute.make_tensor(
                    shared_v.iterator,
                    cute.make_layout((HEAD_DIM, KV_TILE), stride=(1, HEAD_DIM)))
                # QK C[16,8] has four values/lane: two columns in each of
                # two rows. Two adjacent N=8 fragments form PV A[16,16]'s
                # eight values in the same lane, in exactly this flat order.
                # Round P to BF16 directly in registers; no shared-memory staging.
                v_slice = cute.local_tile(v_current, (HEAD_DIM, 16), (0, 0))
                cute.copy(v_ldmatrix, v_loader.partition_S(v_slice), v_loader.retile(v_fragments[0]))
                for p_tile in cutlass.range_constexpr(KV_TILE // 16):
                    if cutlass.const_expr(p_tile + 1 < KV_TILE // 16):
                        v_next = cute.local_tile(v_current, (HEAD_DIM, 16), (0, p_tile + 1))
                        cute.copy(v_ldmatrix, v_loader.partition_S(v_next),
                                  v_loader.retile(v_fragments[(p_tile + 1) % 2]))
                    for i in cutlass.range_constexpr(cute.size(p_regs)):
                        p_regs[i] = scores[p_tile * cute.size(p_regs) + i].to(cutlass.BFloat16)
                    cute.gemm(tiled_mma, output_acc, p_regs,
                              v_fragments[p_tile % 2], output_acc)
                # All warps must finish PV reads before V_(i+1) overwrites V_i.
                # Next iteration issues V_(i+1), then waits for K_(i+1).
                cute.arch.sync_threads()

            # Drain remaining groups, including Q/K_0 for zero-length KV requests.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            output_tensor = cute.make_tensor(output, cute.make_layout(
                (num_tokens, 32, HEAD_DIM), stride=(o_stride_seq, o_stride_head, 1)))
            for i in cutlass.range_constexpr(cute.size(output_acc)):
                coord = output_coords[i]
                q_row = row_start + coord[0]
                if q_row < row_end:
                    row_slot = (i % 4) // 2
                    value = cutlass.Float32(0.0)
                    if l[row_slot] > 0.0:
                        value = output_acc[i] / l[row_slot]
                    output_tensor[q_row, head, coord[1]] = value.to(cutlass.BFloat16)
