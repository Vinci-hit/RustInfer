//! Mandatory, bounded inference validation before the worker advertises Ready.

use super::*;

const SELF_CHECK_SEQUENCE: u64 = u64::MAX;

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    /// Exercise prefill and continuation with the final KV pool and serving
    /// graph configuration. At most two prompt tokens and two decode steps are
    /// needed, including a second decode to replay a newly captured graph.
    ///
    /// This is a bootstrap-only operation: scratch blocks have no allocator
    /// owner yet. Cleanup releases request metadata without reallocating any
    /// graph-referenced buffers. Each step is mirrored separately so all TP
    /// ranks finish capture before the following step can replay it.
    pub fn startup_self_check(&mut self) -> OpResult<()> {
        if self
            .recurrent
            .as_ref()
            .is_some_and(|state| state.has_live_sequences())
            || self.kv_pool.seq_kv_len.contains_key(&SELF_CHECK_SEQUENCE)
        {
            return Err(OpError::Shape(
                "startup self-check requires no live recurrent or synthetic sequence".into(),
            ));
        }
        // The last pool block is reserved for graph padding, so never use it
        // for the synthetic sequence's actual KV history.
        let context = self
            .max_seq_len
            .min(self.max_blocks_per_seq.saturating_mul(self.block_size))
            .min(
                self.kv_pool
                    .num_blocks
                    .saturating_sub(1)
                    .saturating_mul(self.block_size),
            );
        if context < 2 || self.cap_num_tokens == 0 || self.dims.vocab_size == 0 {
            return Err(OpError::Shape(
                "startup self-check requires capacity for a prompt and a decode token, and a nonempty vocabulary".into(),
            ));
        }
        let prompt_len = 2.min(self.cap_num_tokens).min(context - 1);
        let decode_steps = 2.min(context - prompt_len);
        let block_table: Vec<u32> = (0..(prompt_len + decode_steps).div_ceil(self.block_size))
            .map(|block| block as u32)
            .collect();
        let mut req = StepRequest {
            seqs: vec![SeqStep {
                sequence_id: SELF_CHECK_SEQUENCE,
                input_ids: vec![0; prompt_len],
                positions: (0..prompt_len as i32).collect(),
                kv_write_start: 0,
                kv_len_after: prompt_len as i32,
                block_table,
            }],
            sampling: vec![Default::default()],
            stop: StopCriteria {
                eos_ids: Vec::new(),
                generated_counts: vec![0],
                max_tokens: vec![u32::MAX],
                ignore_eos: vec![true],
            },
            draft_tokens: Vec::new(),
        };
        let result = (|| {
            let output = self.step(&req)?;
            let mut token = self.self_check_token(&output)?;
            for position in prompt_len..prompt_len + decode_steps {
                let seq = &mut req.seqs[0];
                seq.input_ids = vec![token];
                seq.positions = vec![position as i32];
                seq.kv_write_start = position as i32;
                seq.kv_len_after = (position + 1) as i32;
                req.stop.generated_counts[0] += 1;
                let output = self.step(&req)?;
                token = self.self_check_token(&output)?;
            }
            Ok(())
        })();
        // Attempt cleanup on both success and failure. A poisoned TP group may
        // reject the mirror command; local cleanup still runs before shutdown.
        let cleanup = self.finish_startup_self_check();
        result.and(cleanup)
    }

    fn self_check_token(&self, output: &StepOutput) -> OpResult<i32> {
        if let [row] = output.tokens.as_slice()
            && let [token] = row.as_slice()
            && token.token_id >= 0
            && (token.token_id as usize) < self.dims.vocab_size
        {
            return Ok(token.token_id);
        }
        Err(OpError::Kernel(
            "startup self-check did not produce one valid vocabulary token".into(),
        ))
    }

    pub(super) fn finish_startup_self_check(&mut self) -> OpResult<()> {
        let pending = self.dispatch_peer_command(RuntimePeerCommand::FinishStartupSelfCheck);
        let local = self.scope.synchronize();
        self.release_sequence(SELF_CHECK_SEQUENCE);
        self.kv_pool.seq_kv_len.remove(&SELF_CHECK_SEQUENCE);
        let followers = match pending {
            Ok(pending) => self.wait_peer_command(pending),
            Err(error) => Err(error),
        };
        complete_replicated(
            &mut self.peers,
            "finish_startup_self_check",
            local,
            followers,
        )
    }
}
