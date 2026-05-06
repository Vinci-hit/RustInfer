//! ModelRunner —— 单进程 Worker 的 GPU 执行者。
//!
//! # 设计
//!
//! - `ModelRunner` 持有**模型 + 所有地址稳定的 device 资源**（Workspace、采样
//!   输出、per-slot InferenceState）。
//! - Runner 的 step **无参**：外部（同进程 server 线程 / 测试）通过 Runner
//!   暴露的 getter 把数据写到各 device tensor，然后用一对 `SyncFlags` 握手
//!   通知 Runner。
//! - Runner 常驻 loop：spin 等 `input_ready=true` → 读 meta → 置
//!   `input_ready=false` → forward → 置 `output_ready=true` → 等 server 消费
//!   → 循环。
//!
//! # 并发模型
//!
//! 单进程两线程：
//! - **Server 线程**：tokenize / 调度 / 填 Runner 的输入 buffer / 读 output。
//! - **Runner 线程**：`ModelRunner::run()`。
//! - 共享 `Arc<ModelRunner>`。内部可变状态（states / output / meta slot）用
//!   `UnsafeCell` 包裹；互斥性靠 `SyncFlags` 的 Acquire/Release 语义保证：
//!   server 只在 `input_ready=false` 时写、`output_ready=true` 时读，其他阶段
//!   runner 独占。
//!
//! # StepMeta
//!
//! 每次 forward 所需的 host 元信息（定长数组 + 标量）。runner 读一份就置
//! `input_ready=false`，server 即可填下一步。device 上的 tensor（per-seq i32
//! 数组、输入 token、KV cache 指针表等）地址稳定，server 直接写入。

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use crate::model::llm::{ForwardCtx, LlmModel};
use crate::model::runtime::InferenceState;
use crate::worker::batch_workspace::BatchWorkspace;

// ============================================================================
//  常量：单 worker 的容量上限（编译期）
// ============================================================================

/// 单步最多能同时处理的 seq 数。决定 `StepMeta` 里定长数组的大小。
pub const MAX_BATCH_SEQS: usize = 32;

// ============================================================================
//  StepMeta —— 每 step 的 host 元信息
// ============================================================================

/// 一次 forward 所需的 host-side 元信息。定长数组避免动态分配；runner 只读
/// 前 `num_seqs` / `num_seqs + 1` 个有效项。
///
/// **对应的 device tensor**（`input_tokens / input_pos / kv_lens_dev /
/// scatter_slot_indices_dev / scatter_seq_positions_dev /
/// scatter_seq_starts_dev / scatter_seq_lens_dev / k_cache_ptrs_dev /
/// v_cache_ptrs_dev / 以及 ragged 调度表）**都在 `BatchWorkspace` 里，地址
/// 跨 step 稳定；server 负责填入。
#[derive(Clone)]
pub struct StepMeta {
    pub num_prefill: usize,
    pub num_decode: usize,
    /// `[num_seqs + 1]`，`q_start_loc[i+1] - q_start_loc[i]` 就是 seq i 的 q_len。
    pub q_start_loc: [i32; MAX_BATCH_SEQS + 1],
    /// `[num_seqs]`，每 seq 对应的 slot id（= runner.states 里的下标）。
    pub slot_indices: [i32; MAX_BATCH_SEQS],
    /// `[num_seqs]`，每 seq 在 input_tokens/pos 缓冲中的起始 token idx
    /// （等价于 q_start_loc[..num_seqs]，为了让模型层 /op 层读取无歧义单独存）。
    pub positions_start: [i32; MAX_BATCH_SEQS],
    /// Ragged prefill kernel 的 grid size；decode-only 时填 0。
    pub total_q_tiles: i32,
}

impl StepMeta {
    pub fn zeroed() -> Self {
        Self {
            num_prefill: 0,
            num_decode: 0,
            q_start_loc: [0; MAX_BATCH_SEQS + 1],
            slot_indices: [0; MAX_BATCH_SEQS],
            positions_start: [0; MAX_BATCH_SEQS],
            total_q_tiles: 0,
        }
    }
    pub fn num_seqs(&self) -> usize {
        self.num_prefill + self.num_decode
    }
    pub fn is_decode_only(&self) -> bool {
        self.num_prefill == 0
    }
    pub fn total_tokens(&self) -> usize {
        self.q_start_loc[self.num_seqs()] as usize
    }
}

// ============================================================================
//  WorkerBatchMeta —— 给模型层 / op 层读 meta 的"稳定 view"
// ============================================================================

/// 模型层/op 层的 meta 视图 —— 保持原有 API（`seq_start(i)` 等）。
///
/// 本轮重构前它的字段是若干 `&[i32]`；现在底层 host 数据由 `StepMeta` 拥有，
/// 这里只是一个短命引用；device 数据全部在 `BatchWorkspace` 里（runner 持有）。
pub struct WorkerBatchMeta<'a> {
    meta: &'a StepMeta,
}

impl<'a> WorkerBatchMeta<'a> {
    pub fn from_step(meta: &'a StepMeta) -> Self {
        Self { meta }
    }

    pub fn num_seqs(&self) -> usize { self.meta.num_seqs() }
    pub fn num_decode(&self) -> usize { self.meta.num_decode }
    pub fn num_prefill(&self) -> usize { self.meta.num_prefill }
    pub fn is_decode_only(&self) -> bool { self.meta.is_decode_only() }
    pub fn total_tokens(&self) -> usize { self.meta.total_tokens() }

    pub fn seq_slot(&self, i: usize) -> usize {
        self.meta.slot_indices[i] as usize
    }
    pub fn seq_start(&self, i: usize) -> usize {
        self.meta.q_start_loc[i] as usize
    }
    pub fn seq_end(&self, i: usize) -> usize {
        self.meta.q_start_loc[i + 1] as usize
    }
    pub fn seq_len(&self, i: usize) -> usize {
        self.seq_end(i) - self.seq_start(i)
    }
    /// 本步 seq i 的起始 position（= 已有 KV 长度 = 该 seq 写入新 KV 的起点）。
    pub fn seq_pos(&self, i: usize) -> i32 {
        self.meta.positions_start[i]
    }
    pub fn seq_end_pos(&self, i: usize) -> Result<usize> {
        let past = self.meta.positions_start[i] as usize;
        let q_len = self.seq_len(i);
        Ok(past + q_len)
    }
}

// ============================================================================
//  SyncFlags —— server / runner 单生产者单消费者握手
// ============================================================================

/// 两个原子 bool 组成的信号位。
///
/// 约定：
/// - `input_ready = true`：server 已经把所有输入（device tensor + host meta）
///   写完，runner 可以启动本 step；
/// - runner 消费完后置 `input_ready = false`；
/// - forward 结束 runner 置 `output_ready = true`；
/// - server 读完 output 后置 `output_ready = false`，开启下一轮。
///
/// 握手保证 runner 读/写 runner 内可变资源时 server 不会并发读/写。
pub struct SyncFlags {
    pub input_ready: AtomicBool,
    pub output_ready: AtomicBool,
    /// 可选的优雅退出信号。置 true 时 runner 的 `run()` 循环会在下一轮开始处返回。
    pub shutdown: AtomicBool,
}

impl SyncFlags {
    fn new() -> Self {
        Self {
            input_ready: AtomicBool::new(false),
            output_ready: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
        }
    }
}

// ============================================================================
//  ModelRunner —— 纯 GPU 执行者
// ============================================================================

pub struct ModelRunner<M: LlmModel> {
    model: M,
    device: DeviceType,
    /// CUDA 上下文（stream / cuBLAS / cuDNN handle / graph cache）。包成
    /// UnsafeCell 以便 `forward_one_step(&self)` 内部调用 `capture_end`
    /// （`&mut CudaConfig`）—— 互斥性同样靠 [`SyncFlags`] 语义保证（runner
    /// 线程独占持有此 cell）。
    #[cfg(feature = "cuda")]
    cuda_cfg: UnsafeCell<crate::cuda::CudaConfig>,

    /// 启用 decode-only CUDA Graph。首次某 batch_size 的 decode-only step
    /// 会做 stream capture + instantiate；之后同 batch_size 的 decode-only step
    /// 直接 replay 已 cache 的 graph。`true` = 启用（默认）。
    /// `false` = 永远走 eager 路径（用于性能基线测试 / 调试）。
    enable_decode_graph: bool,

    /// 已经 warm-up 过 eager forward 的 `num_decode` 桶。CUDA Graph capture
    /// 不允许 cuBLAS / cuDNN 在 default stream 上做 lazy init —— 第一次
    /// 在某 batch size 上走 forward 时，cuBLASLt 会跑算法选择 + 描述符缓存
    /// 等隐式工作，触发 `cudaErrorStreamCaptureImplicit`。所以策略是：
    /// 同一个 num_decode **第一次走 eager**（让所有 lazy init 完成），
    /// **第二次起才 capture**，**第三次起 replay**。
    #[cfg(feature = "cuda")]
    warmed_decode_buckets: UnsafeCell<std::collections::HashSet<usize>>,

    /// 所有跨 step 地址稳定的 device tensor 都在这里。
    /// 用 `UnsafeCell` 包裹是因为 runner 里 workspace 的某些 `refresh_*` 方法
    /// 需要 `&mut self`，而 runner 对外只有 `&self`（Arc 共享）。互斥性由
    /// `SyncFlags` 保证：server 只在 `input_ready=false` 期间访问。
    workspace: UnsafeCell<BatchWorkspace>,

    // ─── 下面 3 个字段由 runner 读写，用 UnsafeCell 绕 &self → &mut T；
    //     互斥性靠 SyncFlags 语义保证（见顶部注释）。───
    /// per-slot InferenceState，len = max_batch_seqs。index = slot id。
    states: UnsafeCell<Vec<InferenceState>>,
    /// 本步采样结果，`[max_batch_seqs] i32`，device。
    output_tokens_dev: UnsafeCell<crate::tensor::Tensor>,
    /// 当前 step 的 host meta。server 写、runner 读一次就释放 input slot。
    meta_slot: UnsafeCell<StepMeta>,

    flags: SyncFlags,
}

// Runner 的字段包含 `UnsafeCell<...>` 和裸指针（workspace 里）；手动标记 Send/Sync
// 以允许 Arc 跨线程共享。互斥性由 SyncFlags 的 Acquire/Release 保证。
unsafe impl<M: LlmModel> Send for ModelRunner<M> {}
unsafe impl<M: LlmModel> Sync for ModelRunner<M> {}

impl<M: LlmModel> ModelRunner<M> {
    /// 构造 runner。**一次性**分配：
    /// - `max_batch_seqs` 个 `InferenceState`（空 KV cache）
    /// - `BatchWorkspace`（包含所有 per-step scratch）
    /// - `output_tokens_dev`: `[max_batch_seqs]` i32 device
    ///
    /// 返回 `Self`。调用者用 `Arc::new(runner)` 跨线程共享。
    pub fn new(
        model: M,
        device: DeviceType,
        max_batch_tokens: usize,
        max_batch_seqs: usize,
    ) -> Result<Self> {
        assert!(
            max_batch_seqs <= MAX_BATCH_SEQS,
            "ModelRunner::new: max_batch_seqs {} > MAX_BATCH_SEQS {}",
            max_batch_seqs, MAX_BATCH_SEQS,
        );

        // 1. per-slot InferenceState
        let mut states = Vec::with_capacity(max_batch_seqs);
        for _ in 0..max_batch_seqs {
            states.push(InferenceState::new(model.config(), device)?);
        }

        // 2. workspace：所有跨 step 地址稳定的 device tensor
        let mut workspace = BatchWorkspace::new(
            model.config(),
            max_batch_tokens,
            max_batch_seqs,
            device,
        )?;
        model.fill_rope_cache(&mut workspace.sin_cache, &mut workspace.cos_cache)?;

        // 3. 采样输出
        let output_tokens_dev = crate::tensor::Tensor::new(
            &[max_batch_seqs],
            crate::base::DataType::I32,
            device,
        )?;

        // 4. CUDA config（stream / cublas handle）
        #[cfg(feature = "cuda")]
        let cuda_cfg = crate::cuda::CudaConfig::new()?;

        Ok(Self {
            model,
            device,
            #[cfg(feature = "cuda")]
            cuda_cfg: UnsafeCell::new(cuda_cfg),
            // 默认开启 decode-only CUDA Graph。第一次某 num_decode 走 eager
            // warm-up（也让 attention kernel 内部的 `cudaFuncSetAttribute`
            // 一次性 setup 完成），第二次起 capture + replay。
            enable_decode_graph: true,
            #[cfg(feature = "cuda")]
            warmed_decode_buckets: UnsafeCell::new(std::collections::HashSet::new()),
            workspace: UnsafeCell::new(workspace),
            states: UnsafeCell::new(states),
            output_tokens_dev: UnsafeCell::new(output_tokens_dev),
            meta_slot: UnsafeCell::new(StepMeta::zeroed()),
            flags: SyncFlags::new(),
        })
    }

    /// 启用 / 关闭 decode-only CUDA Graph。**调用时机**：runner 构造之后、
    /// 启动 `run()` 线程之前。线程已起来之后切换会有数据竞争（`enable_decode_graph`
    /// 字段没有原子语义），调用方需自行保证。
    #[cfg(feature = "cuda")]
    pub fn set_decode_graph_enabled(&mut self, enabled: bool) {
        self.enable_decode_graph = enabled;
        if !enabled {
            self.invalidate_decode_graphs();
        }
    }

    /// 清空已 capture 的所有 decode graph slot。
    ///
    /// 调用时机：调用方（e.g. server）改动了任何 graph 复用所依赖的
    /// device-side 状态（KV cache 扩容导致 ptr 表变化、batch member 重组等），
    /// 必须主动调用一次让 runner 在下一步 step 重新 capture。
    /// 同时清空 warm-up 集合，让相关 batch size 重新走 eager warm-up。
    #[cfg(feature = "cuda")]
    pub fn invalidate_decode_graphs(&self) {
        let cfg = unsafe { &mut *self.cuda_cfg.get() };
        cfg.graphs
            .retain(|slot, _| !matches!(slot, crate::cuda::GraphSlot::LlmDecode(_)));
        let warmed = unsafe { &mut *self.warmed_decode_buckets.get() };
        warmed.clear();
    }

    // ─── Server 写入接口（ready=false 期间 server 拥有这些 cell 的访问权）───

    /// Workspace 的只读引用。Server 通过它拿到具体的 device tensor 写数据：
    ///   - `workspace.input_tokens` / `workspace.input_pos`
    ///   - `workspace.kv_lens_dev`
    ///   - `workspace.scatter_slot_indices_dev` 等（用 `refresh_*` 方法）
    ///   - `workspace.k_cache_ptrs_dev` / `v_cache_ptrs_dev`（用
    ///     `fill_cache_ptrs_from_states` + 对应 &mut 访问）
    ///
    /// # Safety
    /// 只有持有"ready=false"的 server 线程才能调用（且不能并发持有多个引用）。
    pub unsafe fn workspace(&self) -> &BatchWorkspace {
        unsafe { &*self.workspace.get() }
    }

    /// 等价于 `workspace()`，但返回可变引用 —— 用于调用 workspace 的 `refresh_*`
    /// 方法（它们需要 &mut self）。
    ///
    /// # Safety
    /// 只有持有"ready=false"的 server 线程才能调用。
    pub unsafe fn workspace_mut(&self) -> &mut BatchWorkspace {
        unsafe { &mut *self.workspace.get() }
    }

    /// 取 worker stream 给 hot-path H2D 用的辅助函数。
    ///
    /// # Safety
    /// 调用方需保证此引用存活期间没有别的代码动 `cuda_cfg.stream`（实际由
    /// `SyncFlags` 的 input_ready=false 期间 server 独占语义保证）。
    #[cfg(feature = "cuda")]
    pub unsafe fn cuda_stream(&self) -> crate::cuda::ffi::cudaStream_t {
        unsafe { (*self.cuda_cfg.get()).stream }
    }

    /// 指定 slot 的 InferenceState 的可变引用（用于 server 初始化 / 重置 /
    /// `ensure_capacity` 等）。
    ///
    /// # Safety
    /// Server 只能在 `input_ready=false` 期间调用。
    pub unsafe fn state_mut(&self, slot: usize) -> &mut InferenceState {
        let v = unsafe { &mut *self.states.get() };
        &mut v[slot]
    }

    /// 返回 `Vec<InferenceState>` 的裸可变指针。用于需要同时拿多个互异 slot
    /// `&mut` 的场景（例如填 cache_ptrs 表）。
    ///
    /// # Safety
    /// Server 只能在 `input_ready=false` 期间调用，且调用方必须保证不同指针之间
    /// slot 不重合。
    pub unsafe fn states_ptr_mut(&self) -> *mut Vec<InferenceState> {
        self.states.get()
    }

    /// 写 host-side meta。必须在 `set_input_ready()` 之前调用。
    ///
    /// # Safety
    /// Server 只能在 `input_ready=false` 期间调用。
    pub unsafe fn write_meta(&self, meta: StepMeta) {
        unsafe { *self.meta_slot.get() = meta; }
    }

    /// 告诉 runner：输入已就绪，开始执行。
    pub fn set_input_ready(&self) {
        self.flags.input_ready.store(true, Ordering::Release);
    }

    /// 读当前 input_ready（调试/轮询用）。
    pub fn input_ready(&self) -> bool {
        self.flags.input_ready.load(Ordering::Acquire)
    }

    // ─── Server 读取输出接口（output_ready=true 期间）───

    /// 本步采样输出的 device tensor。Server 自己 `to_cpu()` 读回 host。
    /// 长度为 `max_batch_seqs`，但只有前 `num_seqs` 有效。
    ///
    /// # Safety
    /// Server 只能在 `output_ready=true` 期间调用。
    pub unsafe fn output_tokens_dev(&self) -> &crate::tensor::Tensor {
        unsafe { &*self.output_tokens_dev.get() }
    }

    /// 读当前 output_ready。
    pub fn output_ready(&self) -> bool {
        self.flags.output_ready.load(Ordering::Acquire)
    }

    /// 告诉 runner：server 已读完 output，可以开始下一轮了。
    pub fn set_output_consumed(&self) {
        self.flags.output_ready.store(false, Ordering::Release);
    }

    /// 请求 runner 在下一轮 loop 起点退出 `run()`。
    pub fn request_shutdown(&self) {
        self.flags.shutdown.store(true, Ordering::Release);
    }

    pub fn model(&self) -> &M { &self.model }

    // ─── Runner 主循环 ───

    /// 常驻 loop。调用线程会被阻塞在这里直到 `request_shutdown()`。
    ///
    /// 任一步错误直接 panic —— runner 线程退出即意味着 server 永远拿不到
    /// output_ready=true，进程必须同时终止。
    pub fn run(&self) {
        loop {
            // 1. 等输入 ready。spin 等待；热路径预期 server 几 µs 内 ready。
            while !self.flags.input_ready.load(Ordering::Acquire) {
                if self.flags.shutdown.load(Ordering::Acquire) {
                    return;
                }
                std::hint::spin_loop();
            }

            // 2. 读 meta、释放输入 slot。
            //    meta 是 Copy-ish（定长 [i32; N] + 标量），clone 一份离线处理。
            let meta: StepMeta = unsafe { (*self.meta_slot.get()).clone() };
            self.flags.input_ready.store(false, Ordering::Release);

            // 3. 执行 forward。错误 → panic。
            if let Err(e) = self.forward_one_step(&meta) {
                panic!("ModelRunner::run forward error: {:?}", e);
            }

            // 4. 通知 server。
            self.flags.output_ready.store(true, Ordering::Release);

            // 5. 等 server 消费 output。
            while self.flags.output_ready.load(Ordering::Acquire) {
                if self.flags.shutdown.load(Ordering::Acquire) {
                    return;
                }
                std::hint::spin_loop();
            }
        }
    }

    fn forward_one_step(&self, meta: &StepMeta) -> Result<()> {
        let num_seqs = meta.num_seqs();
        if num_seqs == 0 {
            return Ok(());
        }

        // ── Gather per-slot InferenceState 的 mut refs ──
        let states_all = unsafe { &mut *self.states.get() };
        let mut state_refs: Vec<&mut InferenceState> = Vec::with_capacity(num_seqs);
        // 用 split_at_mut / 索引避免 aliasing；slot id 由 server 保证互异。
        let mut slots: Vec<usize> = (0..num_seqs)
            .map(|i| meta.slot_indices[i] as usize)
            .collect();
        // 检查互异
        {
            let mut sorted = slots.clone();
            sorted.sort_unstable();
            for w in sorted.windows(2) {
                if w[0] == w[1] {
                    return Err(Error::InvalidArgument(format!(
                        "slot_indices not unique: {:?}",
                        &meta.slot_indices[..num_seqs]
                    )).into());
                }
            }
        }
        for &slot in &slots {
            let p = &mut states_all[slot] as *mut InferenceState;
            state_refs.push(unsafe { &mut *p });
        }
        let _ = &mut slots; // silence unused if future drops

        // ── WorkerBatchMeta（薄 host meta 视图）——需要先建好，供 workspace 刷 ragged 计划用。
        let meta_view = WorkerBatchMeta::from_step(meta);

        // ── 刷新 ragged 计划（prefill 才需要）；结果覆盖 plan.total_q_tiles。──
        #[cfg(feature = "cuda")]
        let total_q_tiles: i32 = if meta.num_prefill > 0 {
            let ws_mut = unsafe { &mut *self.workspace.get() };
            ws_mut.refresh_ragged_plan(&meta_view)?
        } else {
            0
        };
        #[cfg(not(feature = "cuda"))]
        let total_q_tiles: i32 = 0;

        // ── AttentionPlan：直接填裸指针 ──
        let attn_plan = self.build_attention_plan(meta, total_q_tiles)?;

        // ── output tensor（&mut via UnsafeCell）──
        let output_mut = unsafe { &mut *self.output_tokens_dev.get() };

        // ── CudaConfig ──
        //
        // 用 UnsafeCell 取出 immutable view 给 OpConfig（forward 期间 op 内部
        // 只读 stream / cublas handle）。同一时刻只有 runner 线程在 step，互
        // 斥性靠 SyncFlags 语义保证，下面 graph capture/launch 阶段需要 mut
        // 时再单独 reborrow。
        #[cfg(feature = "cuda")]
        let cuda_cfg_ref = unsafe { &*self.cuda_cfg.get() };
        #[cfg(feature = "cuda")]
        let cuda_cfg: Option<&crate::OpConfig> = Some(cuda_cfg_ref as &crate::OpConfig);
        #[cfg(not(feature = "cuda"))]
        let cuda_cfg: Option<&crate::OpConfig> = None;

        // ── ForwardCtx ──
        let ws = unsafe { &*self.workspace.get() };
        let mut ctx = ForwardCtx::new(
            ws,
            &meta_view,
            state_refs.as_mut_slice(),
            cuda_cfg,
            self.device,
            self.model.config(),
            attn_plan,
            output_mut,
        )?;

        // ── Forward：decode-only 路径下走 CUDA Graph capture / replay ──
        //
        // 触发条件：
        //   - feature = "cuda"
        //   - `enable_decode_graph` 为 true（默认）
        //   - 本步是 decode-only（所有 seq q_len == 1）
        //
        // 控制平面（refresh_scatter_indices / refresh_ragged_plan /
        // fill_cache_ptrs_from_states 等 H2D 写入）在 capture 之前已完成；
        // capture 期间只记录 GPU compute kernel 序列。replay 时这些 H2D
        // 数据由调用方在 step 之间通过 workspace 写入，graph 只复用 kernel 流。
        // ── Forward：所有 GPU 计算都套 thread-local stream，让
        //    `Tensor::copy_from_on_current_stream` 等取到的 stream 一致都是
        //    worker stream，不会跑到 default stream（这是 CUDA Graph 的硬性
        //    要求；同时也避免任何隐式跨 stream 的同步）。
        //
        //    decode-only 路径下分到 `forward_via_graph`：第一次 num_decode
        //    走 eager warm-up（让所有 host-同步的 lazy init 跑完），
        //    第二次起进入 stream-capture，第三次起 replay。
        #[cfg(feature = "cuda")]
        {
            use crate::cuda::with_cuda_stream;
            let stream = unsafe { (*self.cuda_cfg.get()).stream };
            return with_cuda_stream(stream, || {
                if self.enable_decode_graph && meta.is_decode_only() {
                    self.forward_via_graph(num_seqs, &mut ctx)
                } else {
                    self.model.forward(&mut ctx)
                }
            });
        }
        #[cfg(not(feature = "cuda"))]
        self.model.forward(&mut ctx)
    }

    /// CUDA Graph 路径：按 `num_decode` 分桶，已 capture 则 replay；尚未
    /// capture 则要求该桶**至少 warm-up 过一次** eager forward（防止 cuBLAS /
    /// cuDNN 在 capture 期间触发 default stream lazy init），未 warm 时本步
    /// 直接走 eager 并把桶标记 warmed；已 warm 但未 capture 时本步即为 capture
    /// step（capture + instantiate + 立刻 replay 一次让本步真正算出来）。
    ///
    /// 调用前提：调用方已确认 `meta.is_decode_only() && enable_decode_graph`，
    /// 且 ctx 已构造完成。
    #[cfg(feature = "cuda")]
    fn forward_via_graph(
        &self,
        num_decode: usize,
        ctx: &mut ForwardCtx<'_, '_>,
    ) -> Result<()> {
        let slot = crate::cuda::GraphSlot::LlmDecode(num_decode);

        // 已经 capture 过 → 直接 replay。
        {
            let cfg = unsafe { &*self.cuda_cfg.get() };
            if cfg.graph_ready(slot) {
                return cfg.launch(slot);
            }
        }

        // 未 warm-up → 走 eager（让 attention kernel 内部的
        // `cudaFuncSetAttribute`、cuBLASLt heuristic 等一次性 host-同步 setup
        // 都跑完），标记 warmed。下一步同 num_decode 才进入 capture。
        let warmed = unsafe { &mut *self.warmed_decode_buckets.get() };
        if !warmed.contains(&num_decode) {
            self.model.forward(ctx)?;
            warmed.insert(num_decode);
            return Ok(());
        }

        // 已 warm 未 capture → 本步即 capture step。
        let cfg = unsafe { &mut *self.cuda_cfg.get() };
        cfg.capture_begin_relaxed()?;
        let forward_result = self.model.forward(ctx);
        if let Err(e) = forward_result {
            // 清掉 capture 状态，避免 stream 卡在 capturing。
            unsafe {
                let mut graph: crate::cuda::ffi::cudaGraph_t = std::ptr::null_mut();
                let _ = crate::cuda::ffi::cudaStreamEndCapture(cfg.stream, &mut graph);
                if !graph.is_null() {
                    crate::cuda::ffi::cudaGraphDestroy(graph);
                }
            }
            return Err(e);
        }
        cfg.capture_end(slot)?;
        // capture 期间 kernel 不真执行 → 立刻 replay 一次让本步真正算出。
        cfg.launch(slot)
    }
    /// 从 workspace 预填好的 device scratch 构造本步 AttentionPlan。**零 H2D**。
    ///
    /// `total_q_tiles` 由 `refresh_ragged_plan` 在 ragged 路径下算出，decode-only
    /// 下填 0。
    fn build_attention_plan(
        &self,
        meta: &StepMeta,
        total_q_tiles: i32,
    ) -> Result<crate::op::attention::AttentionPlan> {
        use crate::op::attention::{AttentionKind, AttentionPlan};
        #[cfg(feature = "cuda")]
        {
            let ws = unsafe { &*self.workspace.get() };
            let cfg = self.model.config();
            let kind = if meta.is_decode_only() {
                AttentionKind::DecodeOnly
            } else {
                AttentionKind::Ragged
            };
            let kv_stride_s = cfg.kv_dim as i64;
            let kv_stride_h = cfg.head_size as i64;

            Ok(AttentionPlan {
                kind,
                k_cache_ptrs_dev: ws.k_cache_ptrs_dev as *const *const std::ffi::c_void,
                v_cache_ptrs_dev: ws.v_cache_ptrs_dev as *const *const std::ffi::c_void,
                kv_stride_s,
                kv_stride_h,
                req_to_slot_dev: ws.scatter_slot_indices_dev,
                kv_lens_dev: ws.kv_lens_dev.as_i32()?.data_ptr(),
                max_batch_seqs: ws.max_batch_seqs,
                workspace: ws.flash_decode_workspace_dev,
                cu_q_lens_dev: ws.ragged_cu_q_lens_dev,
                block2req_dev: ws.ragged_block2req_dev,
                block2tile_dev: ws.ragged_block2tile_dev,
                total_q_tiles,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (meta, total_q_tiles);
            Ok(AttentionPlan::empty())
        }
    }
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "cuda")]
#[cfg(feature = "models")]
mod tests {
    //! Runner 端到端测试。
    //!
    //! 测试思路：主线程扮演 server，起 runner 在独立线程上 `run()`，
    //! 通过 `SyncFlags` 握手完成 prefill + 若干 decode。
    //!
    //! 需要真实 Llama3 权重；路径从环境变量 `LLAMA3_MODEL_PATH` 读取，
    //! 未设置时跳过。
    use super::*;
    use crate::base::DeviceType;
    use crate::model::llm::llama3::Llama3;
    use std::sync::Arc;

    pub(super) fn get_model_path() -> Option<std::path::PathBuf> {
        std::env::var("LLAMA3_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                // 常见回退路径（与仓库约定一致）
                let p = std::path::PathBuf::from(
                    "/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b",
                );
                if p.exists() { Some(p) } else { None }
            })
    }

    /// 主测试：构造 runner → 起线程 → prefill 一个短 prompt + 连 8 个 decode。
    ///
    /// 验证：
    ///   - 握手协议工作正常（没有死锁 / 数据竞争）；
    ///   - 模型实际能跑通 forward（不 panic）；
    ///   - 每步输出 token id 合法（[0, vocab_size)）。
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
    fn runner_prefill_decode_smoke() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "runner_prefill_decode_smoke: LLAMA3_MODEL_PATH not set; skipping"
                );
                return Ok(());
            }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 1usize;
        let runner = Arc::new(ModelRunner::new(
            model,
            device,
            max_batch_tokens,
            max_batch_seqs,
        )?);

        // ─── 起 runner 线程 ───
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // ─── 准备 prompt ───
        let prompt = "Hello, my name is";
        let prompt_tokens: Vec<i32> = runner
            .model()
            .tokenizer()
            .encode(prompt)?
            .into_iter()
            .collect();
        let prompt_len = prompt_tokens.len();
        assert!(prompt_len > 0 && prompt_len <= max_batch_tokens);

        // ─── Slot 0 的 kv_cache ensure capacity ───
        let max_total = prompt_len + 16;
        unsafe { runner.state_mut(0).kv_cache.ensure_capacity(max_total)?; }

        // ─── Step 1: Prefill ───
        //   - input_tokens / input_pos / kv_lens_dev 写 device
        //   - slot_indices / seq_start_pos / seq_starts / seq_lens 通过
        //     workspace.refresh_scatter_indices(meta_view) 一次填
        //   - fill_cache_ptrs_from_states 一次（这里是首次，必填）
        //   - 写 StepMeta
        //   - set_input_ready → 等 output_ready
        let prefill_meta = {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 1;
            m.num_decode = 0;
            m.q_start_loc[0] = 0;
            m.q_start_loc[1] = prompt_len as i32;
            m.slot_indices[0] = 0;
            m.positions_start[0] = 0;
            // total_q_tiles 由 runner 内部 `refresh_ragged_plan` 覆盖，这里填 0 占位。
            m.total_q_tiles = 0;
            m
        };
        fill_inputs_for_step(
            &runner,
            &prompt_tokens,
            &(0..prompt_len as i32).collect::<Vec<i32>>(),
            &[0i32],
            &prefill_meta,
        )?;
        unsafe { runner.write_meta(prefill_meta.clone()); }
        runner.set_input_ready();

        // 等 runner 完成
        while !runner.output_ready() {
            std::hint::spin_loop();
        }
        let first_token = unsafe { runner.output_tokens_dev() }
            .to_cpu()?
            .as_i32()?
            .as_slice()?[0];
        runner.set_output_consumed();
        assert!(
            first_token >= 0 && (first_token as usize) < vocab,
            "prefill first_token {} out of range [0, {})",
            first_token, vocab
        );
        eprintln!("prefill first_token = {}", first_token);

        // ─── Step 2..: Decode 8 轮 ───
        let mut generated = vec![first_token];
        for step_i in 0..8 {
            let pos = (prompt_len + step_i) as i32;
            let last_tok = *generated.last().unwrap();

            let decode_meta = {
                let mut m = StepMeta::zeroed();
                m.num_prefill = 0;
                m.num_decode = 1;
                m.q_start_loc[0] = 0;
                m.q_start_loc[1] = 1;
                m.slot_indices[0] = 0;
                m.positions_start[0] = pos;
                m.total_q_tiles = 0;
                m
            };
            fill_inputs_for_step(
                &runner,
                &[last_tok],
                &[pos],
                &[pos],
                &decode_meta,
            )?;
            unsafe { runner.write_meta(decode_meta.clone()); }
            runner.set_input_ready();

            while !runner.output_ready() {
                std::hint::spin_loop();
            }
            let tok = unsafe { runner.output_tokens_dev() }
                .to_cpu()?
                .as_i32()?
                .as_slice()?[0];
            runner.set_output_consumed();
            assert!(
                tok >= 0 && (tok as usize) < vocab,
                "decode step {} token {} out of range", step_i, tok
            );
            generated.push(tok);
        }
        eprintln!("generated {} tokens: {:?}", generated.len(), generated);
        assert_eq!(generated.len(), 9);

        // ─── Shutdown ───
        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    /// 小工具：把本步所需的 GPU 输入 / workspace 控制数组 / cache_ptrs 都填好。
    ///
    /// hot-path H2D 全部走 `cudaMemcpyAsync(stream)` —— 不打断 GPU pipeline。
    /// 调用方保证 `tokens / positions / kv_lens` slice 在 step 完成前一直有效
    /// （drive_step 在调本函数和 set_input_ready / 等 output 之间持有这些
    /// `Vec`，自然满足）。
    pub(super) fn fill_inputs_for_step<M: LlmModel>(
        runner: &Arc<ModelRunner<M>>,
        tokens: &[i32],
        positions: &[i32],
        kv_lens: &[i32],
        meta: &StepMeta,
    ) -> Result<()> {
        let ws = unsafe { runner.workspace_mut() };
        #[cfg(feature = "cuda")]
        let stream = unsafe { runner.cuda_stream() };

        // 1. input_tokens / input_pos / kv_lens device H2D（async on worker stream）
        #[cfg(feature = "cuda")]
        {
            ws.input_tokens.as_i32_mut()?.buffer_mut().copy_from_host_async(tokens, stream)?;
            ws.input_pos.as_i32_mut()?.buffer_mut().copy_from_host_async(positions, stream)?;
            ws.kv_lens_dev.as_i32_mut()?.buffer_mut().copy_from_host_async(kv_lens, stream)?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            ws.input_tokens.as_i32_mut()?.buffer_mut().copy_from_host(tokens)?;
            ws.input_pos.as_i32_mut()?.buffer_mut().copy_from_host(positions)?;
            ws.kv_lens_dev.as_i32_mut()?.buffer_mut().copy_from_host(kv_lens)?;
        }

        // 2. per-seq scatter 控制数组 —— 直接从 meta 做一份 meta view，调 workspace 的 refresh
        let meta_view = WorkerBatchMeta::from_step(meta);
        #[cfg(feature = "cuda")]
        ws.refresh_scatter_indices(&meta_view, stream)?;
        #[cfg(not(feature = "cuda"))]
        let _ = &meta_view;

        // 3. KV cache base 指针表。本测试每步都 fill 一次（幂等；若 batch 成员 /
        //    KV 扩容都不变，理论上可以省；为简化测试逻辑不 skip）。
        //    Slot ids 必须互异（runner 本身也在 forward_one_step 里校验）。
        let num_seqs = meta.num_seqs();
        let slots_all = unsafe { &mut *(runner.states_ptr_mut()) };
        let mut refs: Vec<&mut InferenceState> = Vec::with_capacity(num_seqs);
        let mut slot_ids: Vec<usize> = Vec::with_capacity(num_seqs);
        for i in 0..num_seqs {
            let slot = meta.slot_indices[i] as usize;
            if slot_ids.iter().any(|&s| s == slot) {
                return Err(Error::InvalidArgument(format!(
                    "fill_inputs_for_step: duplicate slot {} in meta", slot
                )).into());
            }
            slot_ids.push(slot);
            let p = &mut slots_all[slot] as *mut InferenceState;
            refs.push(unsafe { &mut *p });
        }
        #[cfg(feature = "cuda")]
        ws.fill_cache_ptrs_from_states(&slot_ids, &mut refs, stream)?;
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&slot_ids, &mut refs, ws);
        }
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    //  高层 helper：跑完一 step 并返回 output token slice（host）
    // ──────────────────────────────────────────────────────────────────────

    /// 驱动一步：填输入 + write_meta + set_input_ready + spin 等 output_ready
    /// + 读 output token + set_output_consumed。
    ///
    /// 返回 `Vec<i32>`，长度 = `meta.num_seqs()`。
    /// 驱动一步：填输入 + write_meta + set_input_ready + spin 等 output_ready
    /// + 读 output token + set_output_consumed。
    ///
    /// 返回 `Vec<i32>`，长度 = `meta.num_seqs()`。
    ///
    /// `past_kv_lens[i]` 是第 i 条 seq **进入本步前**已写在 KV cache 里的 token 数
    /// （= 该 seq 当前 KV 起始位置）。本函数会自动把它加上 q_len 得到 attention
    /// kernel 要求的 "current total KV length (past + new)"，写入 workspace。
    pub(super) fn drive_step<M: LlmModel>(
        runner: &Arc<ModelRunner<M>>,
        tokens: &[i32],
        positions: &[i32],
        past_kv_lens: &[i32],
        meta: &StepMeta,
    ) -> Result<Vec<i32>> {
        // attn kernel 的 kv_lens = past + q_len（含本步刚 scatter 的 K/V）。
        let total_kv_lens: Vec<i32> = (0..meta.num_seqs())
            .map(|i| past_kv_lens[i] + meta.q_start_loc[i + 1] - meta.q_start_loc[i])
            .collect();
        fill_inputs_for_step(runner, tokens, positions, &total_kv_lens, meta)?;
        unsafe { runner.write_meta(meta.clone()); }
        runner.set_input_ready();
        while !runner.output_ready() {
            std::hint::spin_loop();
        }
        let tokens_out: Vec<i32> = unsafe { runner.output_tokens_dev() }
            .to_cpu()?
            .as_i32()?
            .as_slice()?
            .iter()
            .take(meta.num_seqs())
            .copied()
            .collect();
        runner.set_output_consumed();
        Ok(tokens_out)
    }

    /// 构造 prefill StepMeta：单 seq，`q_len = prompt_len`，pos 从 0 开始。
    pub(super) fn make_prefill_meta(slot: i32, prompt_len: usize) -> StepMeta {
        let mut m = StepMeta::zeroed();
        m.num_prefill = 1;
        m.num_decode = 0;
        m.q_start_loc[0] = 0;
        m.q_start_loc[1] = prompt_len as i32;
        m.slot_indices[0] = slot;
        m.positions_start[0] = 0;
        m.total_q_tiles = 0;
        m
    }

    /// 构造 single-seq decode StepMeta：一 seq，q_len = 1。
    pub(super) fn make_single_decode_meta(slot: i32, pos: i32) -> StepMeta {
        let mut m = StepMeta::zeroed();
        m.num_prefill = 0;
        m.num_decode = 1;
        m.q_start_loc[0] = 0;
        m.q_start_loc[1] = 1;
        m.slot_indices[0] = slot;
        m.positions_start[0] = pos;
        m.total_q_tiles = 0;
        m
    }

    // ──────────────────────────────────────────────────────────────────────
    //  测试 B：两条 seq 同时 decode（continuous batching）
    // ──────────────────────────────────────────────────────────────────────

    /// 两条请求共用 runner，各自 prefill 完之后**一个 step 里同时做 1 token
    /// decode**。验证：
    ///   - batch_size = 2 的 flash-decode kernel 正确（split-K workspace 共用）；
    ///   - per-seq 的 KV cache 各自正确隔离；
    ///   - 每条输出 token 合法。
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
    fn runner_two_requests_decode_only() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 2usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // ─── Prefill seq 0 + seq 1（各自独立 prefill 一次）───
        let prompts = ["The capital of France is", "I like to eat"];
        let mut kv_lens = [0i32, 0i32];
        let mut last_tokens: [i32; 2] = [-1, -1];

        for (slot, prompt) in prompts.iter().enumerate() {
            let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
            let p_len = toks.len();
            unsafe { runner.state_mut(slot).kv_cache.ensure_capacity(p_len + 8)?; }

            let meta = make_prefill_meta(slot as i32, p_len);
            let positions: Vec<i32> = (0..p_len as i32).collect();
            let out = drive_step(&runner, &toks, &positions, &[0i32], &meta)?;
            assert_eq!(out.len(), 1);
            assert!(out[0] >= 0 && (out[0] as usize) < vocab);
            last_tokens[slot] = out[0];
            kv_lens[slot] = p_len as i32;
            eprintln!("seq {} prefill → token {}", slot, out[0]);
        }

        // ─── 一个 step 同时 decode 两条（batch_size=2）───
        let meta = {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 0;
            m.num_decode = 2;
            m.q_start_loc[0] = 0;
            m.q_start_loc[1] = 1;
            m.q_start_loc[2] = 2;
            m.slot_indices[0] = 0;
            m.slot_indices[1] = 1;
            m.positions_start[0] = kv_lens[0];
            m.positions_start[1] = kv_lens[1];
            m.total_q_tiles = 0;
            m
        };
        let tokens = [last_tokens[0], last_tokens[1]];
        let positions = [kv_lens[0], kv_lens[1]];
        let out = drive_step(&runner, &tokens, &positions, &kv_lens, &meta)?;
        eprintln!("batched decode → {:?}", out);
        assert_eq!(out.len(), 2);
        for (i, &t) in out.iter().enumerate() {
            assert!(
                t >= 0 && (t as usize) < vocab,
                "batched decode seq {} token {} out of range", i, t
            );
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    //  测试 C：Mixed prefill + decode（真正走 Ragged kernel）
    // ──────────────────────────────────────────────────────────────────────

    /// slot 0 已经 decode 了若干步；slot 1 这一步才做 prefill（q_len > 1）。
    /// Ragged kernel 必须正确处理两条 q_len 不同的 seq（1 和 prompt_len）。
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
    fn runner_mixed_prefill_decode() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 2usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // ── slot 0：独立 prefill + 3 步 decode，让 kv_len 推到 prompt_len+3 ──
        let prompt0 = "The quick brown fox jumps";
        let toks0: Vec<i32> = runner.model().tokenizer().encode(prompt0)?;
        let p0_len = toks0.len();
        unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p0_len + 16)?; }

        let meta = make_prefill_meta(0, p0_len);
        let pos: Vec<i32> = (0..p0_len as i32).collect();
        let out = drive_step(&runner, &toks0, &pos, &[0i32], &meta)?;
        let mut last0 = out[0];
        let mut kv0 = p0_len as i32;

        for _ in 0..3 {
            let meta = make_single_decode_meta(0, kv0);
            let out = drive_step(&runner, &[last0], &[kv0], &[kv0], &meta)?;
            last0 = out[0];
            kv0 += 1;
        }
        eprintln!("slot 0 warmed up: kv_len={}, last_token={}", kv0, last0);

        // ── slot 1：一个新 prompt，这一步**和 slot 0 一起 forward**：
        //   slot 0 做 1-token decode，slot 1 做 prefill（q_len = prompt_len）。
        //   这就是"混合 batch"，走 Ragged 路径。──
        let prompt1 = "I love to";
        let toks1: Vec<i32> = runner.model().tokenizer().encode(prompt1)?;
        let p1_len = toks1.len();
        unsafe { runner.state_mut(1).kv_cache.ensure_capacity(p1_len + 8)?; }

        let total_tokens = 1 + p1_len;
        let meta = {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 1;
            m.num_decode = 1;
            // seq 0（decode, q_len=1），seq 1（prefill, q_len=p1_len）
            m.q_start_loc[0] = 0;
            m.q_start_loc[1] = 1;
            m.q_start_loc[2] = total_tokens as i32;
            m.slot_indices[0] = 0;
            m.slot_indices[1] = 1;
            m.positions_start[0] = kv0;
            m.positions_start[1] = 0;
            m.total_q_tiles = 0; // runner 会 refresh_ragged_plan 覆盖
            m
        };
        // 拼 packed tokens / positions
        let mut tokens = Vec::with_capacity(total_tokens);
        tokens.push(last0);
        tokens.extend_from_slice(&toks1);
        let mut positions = Vec::with_capacity(total_tokens);
        positions.push(kv0);
        for i in 0..p1_len { positions.push(i as i32); }
        let kv_lens = [kv0, 0i32];

        let out = drive_step(&runner, &tokens, &positions, &kv_lens, &meta)?;
        eprintln!("mixed step → {:?}", out);
        assert_eq!(out.len(), 2);
        for (i, &t) in out.iter().enumerate() {
            assert!(
                t >= 0 && (t as usize) < vocab,
                "mixed step seq {} token {} out of range", i, t
            );
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    //  测试 D：长 prompt 触发 KV cache 扩容
    // ──────────────────────────────────────────────────────────────────────

    /// Prompt 长度 > 初始 KV cache 容量（默认 512），强制触发
    /// `ensure_capacity` 重新 alloc → KV cache base 地址变化。因为本测试的
    /// `fill_inputs_for_step` 每步都 `fill_cache_ptrs_from_states`，扩容后
    /// 立即刷入新地址，kernel 使用的永远是最新的指针表。
    ///
    /// 验证：扩容路径不 crash，且扩容后的 decode 依然得到合法 token。
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
    fn runner_long_prompt_triggers_kv_grow() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 1024usize;
        let max_batch_seqs = 1usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // 用 repeat 造一个 > 512 tokens 的 prompt。
        // 单词 "the " 大约 1 token，这里 ~600 个单词 = ~600 tokens。
        let big_prompt = "the ".repeat(600);
        let toks: Vec<i32> = runner.model().tokenizer().encode(&big_prompt)?;
        let p_len = toks.len();
        assert!(p_len > 512, "prompt length {} not long enough to trigger KV grow", p_len);
        assert!(p_len <= max_batch_tokens, "prompt {} > max_batch_tokens {}", p_len, max_batch_tokens);
        eprintln!("long prompt tokenized to {} tokens", p_len);

        // ensure_capacity 将触发 reallocation：old cap = 512 → new cap ≥ p_len。
        unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p_len + 4)?; }

        let meta = make_prefill_meta(0, p_len);
        let positions: Vec<i32> = (0..p_len as i32).collect();
        let out = drive_step(&runner, &toks, &positions, &[0i32], &meta)?;
        assert_eq!(out.len(), 1);
        assert!(
            out[0] >= 0 && (out[0] as usize) < vocab,
            "long prefill token {} out of range", out[0]
        );
        eprintln!("long prompt prefill first_token = {}", out[0]);

        // 再跑一步 decode 确保扩容后依然工作
        let meta = make_single_decode_meta(0, p_len as i32);
        let out = drive_step(&runner, &[out[0]], &[p_len as i32], &[p_len as i32], &meta)?;
        assert!(
            out[0] >= 0 && (out[0] as usize) < vocab,
            "post-grow decode token {} out of range", out[0]
        );
        eprintln!("post-grow decode token = {}", out[0]);

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    //  测试 E：多个真实 prompt 的"说人话"检查
    // ──────────────────────────────────────────────────────────────────────

    /// 喂几个有代表性的 prompt，各跑若干步 decode，decode 回文本打印出来。
    /// 自动断言只做最低限度（非空 + token 合法），主要靠**测试输出的文本**
    /// 肉眼判断能不能说人话。跑法：
    ///
    /// ```text
    /// cargo test -p infer-worker --features "cuda,models" \
    ///     --lib worker::runner::tests::runner_generate_sentences \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
    fn runner_generate_sentences() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Llama3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 256usize;
        let max_batch_seqs = 1usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // 一些典型 prompt：问答 / 续写 / 指令 / 对话。
        let prompts = [
            "The capital of France is",
            "Once upon a time in a small village,",
            "Q: What is 2 + 2?\nA:",
            "Hello, how are you today? I am",
        ];
        let max_new_tokens = 30usize;

        for (idx, prompt) in prompts.iter().enumerate() {
            // 每个 prompt 换一个干净的 slot（避免上个 prompt 的 KV 残留）。
            // 本测试 max_batch_seqs=1，所以强制每次重建 slot 0 的 KV cache。
            //
            // 最简单做法：重建整个 runner。但这里用更轻量的手段——
            // 直接替换 slot 0 的 InferenceState。
            {
                // Input-ready 必须是 false 时才能改 state
                assert!(!runner.input_ready());
                let new_state = InferenceState::new(
                    runner.model().config(),
                    device,
                )?;
                let slot_mut = unsafe { runner.state_mut(0) };
                *slot_mut = new_state;
            }

            let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
            let p_len = toks.len();
            assert!(
                p_len > 0 && p_len + max_new_tokens <= max_batch_tokens,
                "prompt '{}' tokenized to {} tokens, overflows budget",
                prompt, p_len,
            );
            unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p_len + max_new_tokens)?; }

            // Prefill
            let meta = make_prefill_meta(0, p_len);
            let pos: Vec<i32> = (0..p_len as i32).collect();
            let out = drive_step(&runner, &toks, &pos, &[0i32], &meta)?;
            let mut generated: Vec<i32> = vec![out[0]];
            let mut kv_len = p_len as i32;

            // Decode 循环
            let tokenizer = runner.model().tokenizer();
            let eos_candidates: Vec<i32> = (0..vocab as i32)
                .filter(|&t| tokenizer.is_eos(t))
                .collect();
            let mut hit_eos = false;

            for _ in 0..(max_new_tokens - 1) {
                let last = *generated.last().unwrap();
                if eos_candidates.contains(&last) {
                    hit_eos = true;
                    break;
                }
                let meta = make_single_decode_meta(0, kv_len);
                let out = drive_step(&runner, &[last], &[kv_len], &[kv_len], &meta)?;
                assert!(
                    out[0] >= 0 && (out[0] as usize) < vocab,
                    "prompt #{}: decoded token {} out of vocab", idx, out[0]
                );
                generated.push(out[0]);
                kv_len += 1;
            }

            let decoded = tokenizer.decode(&generated).unwrap_or_default();
            assert!(
                !decoded.trim().is_empty(),
                "prompt #{} ('{}') → empty decoded string", idx, prompt
            );
            eprintln!("─────────────────────────────────────────");
            eprintln!("[{}] prompt: {:?}", idx, prompt);
            eprintln!("[{}] tokens ({}): {:?}", idx, generated.len(), generated);
            eprintln!("[{}] decoded: {}{}",
                idx, decoded,
                if hit_eos { " <EOS>" } else { "" });
        }
        eprintln!("─────────────────────────────────────────");

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
#[cfg(feature = "models")]
mod tests_qwen3 {
    //! Qwen3 端到端测试，复用 [`super::tests`] 里的 step helper。结构与 Llama3
    //! 测试同形：smoke / two_requests_decode_only / mixed_prefill_decode /
    //! generate_sentences。模型路径取 `QWEN3_MODEL_PATH`，不存在就 fallback
    //! 到 `/apdcephfs_qy2/.../qwen3-4b-instruct`（4B BF16，含 q_norm/k_norm）。
    use super::*;
    use super::tests::{drive_step, make_prefill_meta, make_single_decode_meta};
    use crate::base::DeviceType;
    use crate::model::llm::qwen3::Qwen3;
    use std::sync::Arc;

    fn get_qwen3_model_path() -> Option<std::path::PathBuf> {
        std::env::var("QWEN3_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let p = std::path::PathBuf::from(
                    "/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct",
                );
                if p.exists() { Some(p) } else { None }
            })
    }

    /// 单请求 prefill + 多步 decode：跑通一段，输出非空。
    #[test]
    #[ignore = "requires QWEN3_MODEL_PATH env or well-known model path"]
    fn runner_qwen3_prefill_decode_smoke() -> Result<()> {
        let path = match get_qwen3_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no qwen3 model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Qwen3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 1usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        let prompt = "The capital of France is";
        let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
        let p_len = toks.len();
        unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p_len + 16)?; }

        let meta = make_prefill_meta(0, p_len);
        let pos: Vec<i32> = (0..p_len as i32).collect();
        let out = drive_step(&runner, &toks, &pos, &[0i32], &meta)?;
        let first = out[0];
        assert!(first >= 0 && (first as usize) < vocab);
        eprintln!("qwen3 prefill first_token = {}", first);

        let mut kv_len = p_len as i32;
        let mut last = first;
        for _ in 0..8 {
            let meta = make_single_decode_meta(0, kv_len);
            let out = drive_step(&runner, &[last], &[kv_len], &[kv_len], &meta)?;
            assert!(out[0] >= 0 && (out[0] as usize) < vocab);
            last = out[0];
            kv_len += 1;
        }
        eprintln!("qwen3 decode last_token = {}", last);

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    /// 双请求纯 decode：先各自 prefill 拿 first token，再 batched decode 一步。
    #[test]
    #[ignore = "requires QWEN3_MODEL_PATH env or well-known model path"]
    fn runner_qwen3_two_requests_decode_only() -> Result<()> {
        let path = match get_qwen3_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no qwen3 model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Qwen3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 2usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        let prompts = ["The capital of France is", "The sky is"];
        let mut last_toks = [0i32; 2];
        let mut kv_lens = [0i32; 2];

        for (i, prompt) in prompts.iter().enumerate() {
            let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
            let p_len = toks.len();
            unsafe { runner.state_mut(i).kv_cache.ensure_capacity(p_len + 4)?; }
            let meta = make_prefill_meta(i as i32, p_len);
            let pos: Vec<i32> = (0..p_len as i32).collect();
            let out = drive_step(&runner, &toks, &pos, &[0i32], &meta)?;
            assert!(out[0] >= 0 && (out[0] as usize) < vocab);
            eprintln!("qwen3 seq {} prefill → token {}", i, out[0]);
            last_toks[i] = out[0];
            kv_lens[i] = p_len as i32;
        }

        // 一次 batched decode（两条 seq，每条 q_len=1）
        let mut meta = StepMeta::zeroed();
        meta.num_prefill = 0;
        meta.num_decode = 2;
        meta.slot_indices[0] = 0;
        meta.slot_indices[1] = 1;
        meta.q_start_loc[0] = 0;
        meta.q_start_loc[1] = 1;
        meta.q_start_loc[2] = 2;
        meta.positions_start[0] = kv_lens[0];
        meta.positions_start[1] = kv_lens[1];

        let tokens: Vec<i32> = vec![last_toks[0], last_toks[1]];
        let positions: Vec<i32> = vec![kv_lens[0], kv_lens[1]];
        let kv_lens_in: Vec<i32> = vec![kv_lens[0], kv_lens[1]];
        let out = drive_step(&runner, &tokens, &positions, &kv_lens_in, &meta)?;
        eprintln!("qwen3 batched decode → {:?}", &out[..2]);
        for &t in &out[..2] {
            assert!(t >= 0 && (t as usize) < vocab);
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    /// 多 prompt 多步生成，验证输出"说人话"（非空、可解码）。
    #[test]
    #[ignore = "requires QWEN3_MODEL_PATH env or well-known model path"]
    fn runner_qwen3_generate_sentences() -> Result<()> {
        let path = match get_qwen3_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no qwen3 model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Qwen3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 256usize;
        let max_batch_seqs = 1usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        let prompts = [
            "The capital of France is",
            "Once upon a time in a small village,",
            "Q: What is 2 + 2?\nA:",
            "Hello, how are you today? I am",
        ];
        let max_new_tokens = 30usize;

        for (idx, prompt) in prompts.iter().enumerate() {
            {
                assert!(!runner.input_ready());
                let new_state = InferenceState::new(runner.model().config(), device)?;
                let slot_mut = unsafe { runner.state_mut(0) };
                *slot_mut = new_state;
            }

            let toks: Vec<i32> = runner.model().tokenizer().encode(prompt)?;
            let p_len = toks.len();
            assert!(
                p_len > 0 && p_len + max_new_tokens <= max_batch_tokens,
                "prompt '{}' tokenized to {} tokens, overflows budget",
                prompt, p_len,
            );
            unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p_len + max_new_tokens)?; }

            let meta = make_prefill_meta(0, p_len);
            let pos: Vec<i32> = (0..p_len as i32).collect();
            let out = drive_step(&runner, &toks, &pos, &[0i32], &meta)?;
            let mut generated: Vec<i32> = vec![out[0]];
            let mut kv_len = p_len as i32;

            let tokenizer = runner.model().tokenizer();
            let eos_candidates: Vec<i32> = (0..vocab as i32)
                .filter(|&t| tokenizer.is_eos(t))
                .collect();
            let mut hit_eos = false;

            for _ in 0..(max_new_tokens - 1) {
                let last = *generated.last().unwrap();
                if eos_candidates.contains(&last) {
                    hit_eos = true;
                    break;
                }
                let meta = make_single_decode_meta(0, kv_len);
                let out = drive_step(&runner, &[last], &[kv_len], &[kv_len], &meta)?;
                assert!(
                    out[0] >= 0 && (out[0] as usize) < vocab,
                    "prompt #{}: decoded token {} out of vocab", idx, out[0]
                );
                generated.push(out[0]);
                kv_len += 1;
            }

            let decoded = tokenizer.decode(&generated).unwrap_or_default();
            assert!(
                !decoded.trim().is_empty(),
                "prompt #{} ('{}') → empty decoded string", idx, prompt
            );
            eprintln!("─────────────────────────────────────────");
            eprintln!("[{}] prompt: {:?}", idx, prompt);
            eprintln!("[{}] tokens ({}): {:?}", idx, generated.len(), generated);
            eprintln!("[{}] decoded: {}{}",
                idx, decoded,
                if hit_eos { " <EOS>" } else { "" });
        }
        eprintln!("─────────────────────────────────────────");

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }

    /// 混合 prefill + decode：slot 0 已 warm 到若干步 decode，slot 1 同步发起新 prefill。
    /// Ragged kernel 必须正确处理两条 q_len 不同的 seq（1 和 prompt_len）。
    #[test]
    #[ignore = "requires QWEN3_MODEL_PATH env or well-known model path"]
    fn runner_qwen3_mixed_prefill_decode() -> Result<()> {
        let path = match get_qwen3_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no qwen3 model path"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let model = Qwen3::new(&path, device)?;
        let vocab = model.config().vocab_size;

        let max_batch_tokens = 512usize;
        let max_batch_seqs = 2usize;
        let runner = Arc::new(ModelRunner::new(model, device, max_batch_tokens, max_batch_seqs)?);
        let runner_loop = Arc::clone(&runner);
        let runner_handle = std::thread::spawn(move || runner_loop.run());

        // slot 0：独立 prefill + 3 步 decode 暖好 kv
        let prompt0 = "The quick brown fox jumps";
        let toks0: Vec<i32> = runner.model().tokenizer().encode(prompt0)?;
        let p0_len = toks0.len();
        unsafe { runner.state_mut(0).kv_cache.ensure_capacity(p0_len + 16)?; }
        let meta = make_prefill_meta(0, p0_len);
        let pos: Vec<i32> = (0..p0_len as i32).collect();
        let out = drive_step(&runner, &toks0, &pos, &[0i32], &meta)?;
        let mut last0 = out[0];
        let mut kv0 = p0_len as i32;
        for _ in 0..3 {
            let meta = make_single_decode_meta(0, kv0);
            let out = drive_step(&runner, &[last0], &[kv0], &[kv0], &meta)?;
            last0 = out[0];
            kv0 += 1;
        }
        eprintln!("qwen3 slot 0 warmed: kv_len={}, last_token={}", kv0, last0);

        // slot 1：新 prompt，与 slot 0 同步 forward —— slot 0 decode（q_len=1），
        // slot 1 prefill（q_len=p1_len）
        let prompt1 = "I love to";
        let toks1: Vec<i32> = runner.model().tokenizer().encode(prompt1)?;
        let p1_len = toks1.len();
        unsafe { runner.state_mut(1).kv_cache.ensure_capacity(p1_len + 8)?; }

        let total_tokens = 1 + p1_len;
        let meta = {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 1;
            m.num_decode = 1;
            m.q_start_loc[0] = 0;
            m.q_start_loc[1] = 1;
            m.q_start_loc[2] = total_tokens as i32;
            m.slot_indices[0] = 0;
            m.slot_indices[1] = 1;
            m.positions_start[0] = kv0;
            m.positions_start[1] = 0;
            m.total_q_tiles = 0;
            m
        };
        let mut tokens = Vec::with_capacity(total_tokens);
        tokens.push(last0);
        tokens.extend_from_slice(&toks1);
        let mut positions = Vec::with_capacity(total_tokens);
        positions.push(kv0);
        for i in 0..p1_len { positions.push(i as i32); }
        let kv_lens = [kv0, 0i32];

        let out = drive_step(&runner, &tokens, &positions, &kv_lens, &meta)?;
        eprintln!("qwen3 mixed step → {:?}", &out[..2]);
        for (i, &t) in out[..2].iter().enumerate() {
            assert!(
                t >= 0 && (t as usize) < vocab,
                "qwen3 mixed step seq {} token {} out of range", i, t
            );
        }

        runner.request_shutdown();
        let _ = runner_handle.join();
        Ok(())
    }
}


#[cfg(test)]
#[cfg(feature = "cuda")]
#[cfg(feature = "models")]
mod tests_perf {
    //! Decode-only CUDA Graph 性能基准。
    //!
    //! 跑两组矩阵：
    //!   * Llama3-1B、Qwen3-4B 各自；
    //!   * batch ∈ {1, 2, 4, 8}；
    //!   * 每条 seq 同 prompt prefill 一次（不计时），随后跑 256 步 batched decode；
    //!   * 跳过前 2 步消除 warm-up + capture 的一次性开销；
    //!   * 报告 `tokens/s = batch * decode_steps / wall_time`。
    //!
    //! Graph 路径默认开启（`enable_decode_graph = true`），与 eager 路径
    //! （`runner_decode_graph_compare` 里另测）的结果一并打印。
    //!
    //! 跳过条件：模型路径不可达。
    use super::*;
    use super::tests::{
        drive_step, get_model_path, make_prefill_meta,
    };
    use crate::base::DeviceType;
    use crate::model::llm::llama3::Llama3;
    use crate::model::llm::qwen3::Qwen3;
    use std::sync::Arc;
    use std::time::Instant;

    const DECODE_STEPS: usize = 256;
    const BATCH_SIZES: &[usize] = &[1, 2, 4, 8];
    const PROMPT: &str = "The capital of France is";

    fn qwen3_path() -> Option<std::path::PathBuf> {
        std::env::var("QWEN3_MODEL_PATH")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                let p = std::path::PathBuf::from(
                    "/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct",
                );
                if p.exists() { Some(p) } else { None }
            })
    }

    /// 运行 `batch` 条相同 prompt 的 prefill+decode；返回 `(total_decode_ms,
    /// tokens_per_sec)`。前 2 步 decode 不计时（吃掉 graph warm-up + capture）。
    fn bench_one_config<M: LlmModel + 'static>(
        runner: &Arc<ModelRunner<M>>,
        batch: usize,
        decode_steps: usize,
    ) -> Result<(f64, f64)> {
        let tokenizer = runner.model().tokenizer();
        let toks: Vec<i32> = tokenizer.encode(PROMPT)?;
        let p_len = toks.len();
        let cap = p_len + decode_steps + 4;

        // ── 每条 seq 各自 prefill（不计时；shape 不固定，graph 不会复用）──
        let mut last_tok = vec![0i32; batch];
        let mut kv_len = vec![0i32; batch];
        for slot in 0..batch {
            unsafe { runner.state_mut(slot).kv_cache.ensure_capacity(cap)?; }
            let meta = make_prefill_meta(slot as i32, p_len);
            let pos: Vec<i32> = (0..p_len as i32).collect();
            let out = drive_step(runner, &toks, &pos, &[0i32], &meta)?;
            last_tok[slot] = out[0];
            kv_len[slot] = p_len as i32;
        }

        // ── 构造 batched decode meta（所有 seq q_len=1）──
        let make_decode_meta = |kv_lens_now: &[i32]| -> StepMeta {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 0;
            m.num_decode = batch;
            for i in 0..batch {
                m.slot_indices[i] = i as i32;
                m.q_start_loc[i] = i as i32;
                m.positions_start[i] = kv_lens_now[i];
            }
            m.q_start_loc[batch] = batch as i32;
            m.total_q_tiles = 0;
            m
        };

        // ── 单步 batched decode helper ──
        let mut step = |last: &mut [i32], kvl: &mut [i32]| -> Result<()> {
            let meta = make_decode_meta(kvl);
            let positions: Vec<i32> = kvl.to_vec();
            let kv_lens_in: Vec<i32> = kvl.to_vec();
            let out = drive_step(runner, last, &positions, &kv_lens_in, &meta)?;
            for i in 0..batch {
                last[i] = out[i];
                kvl[i] += 1;
            }
            Ok(())
        };

        // 前 2 步 warm-up（让 graph 这个 batch_size 桶 capture 完成）
        for _ in 0..2 {
            step(&mut last_tok, &mut kv_len)?;
        }

        // 计时段
        let start = Instant::now();
        for _ in 0..decode_steps {
            step(&mut last_tok, &mut kv_len)?;
        }
        let elapsed_s = start.elapsed().as_secs_f64();
        let total_tokens = (batch * decode_steps) as f64;
        let tokens_per_sec = total_tokens / elapsed_s;
        Ok((elapsed_s * 1000.0, tokens_per_sec))
    }

    /// 在 `(model_name, model_path, batches)` 上跑两轮（eager + graph）扫描。
    /// 返回每个 batch 的 (eager_tps, graph_tps)。
    fn run_matrix<M, F>(
        model_name: &str,
        path: &std::path::Path,
        batches: &[usize],
        max_batch_tokens: usize,
        new_model: F,
    ) -> Result<()>
    where
        M: LlmModel + 'static,
        F: Fn(&std::path::Path, DeviceType) -> Result<M>,
    {
        let device = DeviceType::Cuda(0);

        eprintln!();
        eprintln!("══════════════════════════════════════════════════════════════════");
        eprintln!("  {} —— decode benchmark (steps={})", model_name, DECODE_STEPS);
        eprintln!("══════════════════════════════════════════════════════════════════");
        eprintln!(
            "{:>6}  {:>14}  {:>14}  {:>10}",
            "batch", "eager tok/s", "graph tok/s", "speedup"
        );
        eprintln!("{:>6}  {:>14}  {:>14}  {:>10}", "─────", "──────────────", "──────────────", "──────────");

        for &batch in batches {
            // (A) eager
            let model_a = new_model(path, device)?;
            let mut runner_a_owned =
                ModelRunner::new(model_a, device, max_batch_tokens, batch)?;
            runner_a_owned.set_decode_graph_enabled(false);
            let runner_a = Arc::new(runner_a_owned);
            let runner_a_loop = Arc::clone(&runner_a);
            let handle_a = std::thread::spawn(move || runner_a_loop.run());
            let (_, eager_tps) = bench_one_config(&runner_a, batch, DECODE_STEPS)?;
            runner_a.request_shutdown();
            let _ = handle_a.join();

            // (B) graph
            let model_b = new_model(path, device)?;
            let runner_b_owned =
                ModelRunner::new(model_b, device, max_batch_tokens, batch)?;
            // graph 默认 enabled
            let runner_b = Arc::new(runner_b_owned);
            let runner_b_loop = Arc::clone(&runner_b);
            let handle_b = std::thread::spawn(move || runner_b_loop.run());
            let (_, graph_tps) = bench_one_config(&runner_b, batch, DECODE_STEPS)?;
            runner_b.request_shutdown();
            let _ = handle_b.join();

            let speedup = graph_tps / eager_tps;
            eprintln!(
                "{:>6}  {:>14.1}  {:>14.1}  {:>9.2}x",
                batch, eager_tps, graph_tps, speedup
            );
        }
        eprintln!("══════════════════════════════════════════════════════════════════");
        Ok(())
    }

    #[test]
    #[ignore = "perf benchmark; requires LLAMA3_MODEL_PATH"]
    fn perf_llama3_decode_matrix() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no LLAMA3_MODEL_PATH"); return Ok(()); }
        };
        // max_batch_tokens 要 ≥ batch * 1（每步 batch 个 token）以及 prefill 的 prompt_len。
        // 取 1024 兼顾 batch=8 + prompt 长度。
        run_matrix::<Llama3, _>(
            "Llama3-1B",
            &path,
            BATCH_SIZES,
            1024,
            |p, d| Llama3::new(p, d),
        )
    }

    #[test]
    #[ignore = "perf benchmark; requires QWEN3_MODEL_PATH"]
    fn perf_qwen3_decode_matrix() -> Result<()> {
        let path = match qwen3_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no QWEN3_MODEL_PATH"); return Ok(()); }
        };
        run_matrix::<Qwen3, _>(
            "Qwen3-4B",
            &path,
            BATCH_SIZES,
            1024,
            |p, d| Qwen3::new(p, d),
        )
    }

    /// 只跑 Llama3 BS=1 graph 路径的纯 decode loop（无 eager pass）。
    /// 给 nsys profile 用，输出干净不会被多 batch 混淆。
    #[test]
    #[ignore = "for nsys profile; LLAMA3_MODEL_PATH"]
    fn perf_llama3_b1_graph_profile() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no LLAMA3_MODEL_PATH"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let max_batch_tokens = 1024usize;
        let batch = 1usize;

        let model = Llama3::new(&path, device)?;
        let runner_owned = ModelRunner::new(model, device, max_batch_tokens, batch)?;
        let runner = Arc::new(runner_owned);
        let runner_loop = Arc::clone(&runner);
        let handle = std::thread::spawn(move || runner_loop.run());

        let (_, tps) = bench_one_config(&runner, batch, DECODE_STEPS)?;
        eprintln!("[profile] Llama3 BS=1 graph: {:.1} tok/s", tps);

        runner.request_shutdown();
        let _ = handle.join();
        Ok(())
    }

    /// 只跑 Llama3 BS=1 **eager** 路径。给 nsys profile 用 —— graph 路径下
    /// nsys 看不到 graph 内每个 kernel 的真实 launch overhead，eager 路径
    /// 才能看到 per-op per-launch 的真实时间分布。
    #[test]
    #[ignore = "for nsys profile; LLAMA3_MODEL_PATH"]
    fn perf_llama3_b1_eager_profile() -> Result<()> {
        let path = match get_model_path() {
            Some(p) => p,
            None => { eprintln!("skipping: no LLAMA3_MODEL_PATH"); return Ok(()); }
        };
        let device = DeviceType::Cuda(0);
        let max_batch_tokens = 1024usize;
        let batch = 1usize;

        let model = Llama3::new(&path, device)?;
        let mut runner_owned = ModelRunner::new(model, device, max_batch_tokens, batch)?;
        runner_owned.set_decode_graph_enabled(false);
        let runner = Arc::new(runner_owned);
        let runner_loop = Arc::clone(&runner);
        let handle = std::thread::spawn(move || runner_loop.run());

        let (_, tps) = bench_one_config(&runner, batch, DECODE_STEPS)?;
        eprintln!("[profile] Llama3 BS=1 eager: {:.1} tok/s", tps);

        runner.request_shutdown();
        let _ = handle.join();
        Ok(())
    }
}
