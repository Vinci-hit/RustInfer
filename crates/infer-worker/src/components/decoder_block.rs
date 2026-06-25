use crate::components::attention::Attention;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;

/// One transformer decoder layer: a pre-norm attention sublayer followed by a
/// pre-norm FFN sublayer, both operating on the carried residual stream. Each
/// sublayer owns its own input norm (inv 7), so the block is just their
/// composition. `F` is the FFN — `DenseFfn` or `MoeFfn` (the dense↔MoE swap).
pub struct DecoderBlock<T: Dtype, D: LlmBackend, F: Component<T, D>> {
    pub attention: Attention<T, D>,
    pub ffn: F,
}

impl<T: Dtype, D: LlmBackend, F: Component<T, D>> Component<T, D> for DecoderBlock<T, D, F> {
    fn kind(&self) -> StageKind {
        StageKind::DecoderBlock
    }

    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        // Env-gated per-phase timing (RUSTINFER_PHASE_TRACE): synchronize after
        // attention and after FFN, accumulate GPU time per phase, log every 36
        // layers (one forward). Diagnostic only; off by default. The syncs
        // serialize the forward (inflating absolutes) but the attn:ffn RATIO
        // localizes a regression.
        use std::sync::atomic::{AtomicU64, Ordering};
        static ATTN_NS: AtomicU64 = AtomicU64::new(0);
        static FFN_NS: AtomicU64 = AtomicU64::new(0);
        static LAYERS: AtomicU64 = AtomicU64::new(0);
        let phase_trace = std::env::var_os("RUSTINFER_PHASE_TRACE").is_some();
        if phase_trace {
            let t0 = std::time::Instant::now();
            self.attention.run(hidden, kv, ctx)?;
            let _ = ctx.scope().synchronize();
            let t1 = std::time::Instant::now();
            self.ffn.run(hidden, None, ctx)?;
            let _ = ctx.scope().synchronize();
            let t2 = std::time::Instant::now();
            ATTN_NS.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
            FFN_NS.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
            if LAYERS.fetch_add(1, Ordering::Relaxed) % 36 == 35 {
                let a = ATTN_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                let f = FFN_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
                tracing::info!("[phase-trace] 36L attn={:.2}ms ffn={:.2}ms total={:.2}ms", a, f, a + f);
            }
            return Ok(());
        }
        self.attention.run(hidden, kv, ctx)?;
        self.ffn.run(hidden, None, ctx)
    }
}
