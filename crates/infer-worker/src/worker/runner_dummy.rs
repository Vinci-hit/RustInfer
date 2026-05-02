//! Dummy ModelRunner for CPU-only testing (no model loaded).
//! Outputs token_id = 42 for every seq.

use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::Arc;

use crate::worker::shared_buffers::SharedBuffers;

/// Dummy runner that always outputs token 42
pub struct DummyModelRunner {
    shared: Arc<SharedBuffers>,
}

impl DummyModelRunner {
    pub fn new(shared: Arc<SharedBuffers>) -> Self {
        Self { shared }
    }

    pub fn run(self) {
        loop {
            // spin wait input
            let _total_tokens = loop {
                let v = self.shared.input_meta.ready.load(Acquire);
                if v > 0 {
                    break v as usize;
                }
                std::hint::spin_loop();
            };

            let num_decode = self.shared.input_meta.num_decode_seqs.load(Acquire) as usize;
            let num_prefill = self.shared.input_meta.num_prefill_seqs.load(Acquire) as usize;
            let num_seqs = num_decode + num_prefill;

            // 写 dummy output 到 shared device/CPU tensor。
            let out = vec![42i32; num_seqs];
            let mut output_token_ids = self.shared.output_token_ids.clone();
            output_token_ids.write_from_i32_host(&out, num_seqs).expect("write dummy output");

            // signal
            self.shared.input_meta.ready.store(0, Release);
            self.shared.output_meta.ready.store(num_seqs as u32, Release);
        }
    }
}
