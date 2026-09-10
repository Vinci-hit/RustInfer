//! Explicit eager prefill adapter sharing ordinary planning and commit rules.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_eager_prefill<M, F>(
    cmd: &PrefillBatchCmd,
    runner: &mut Runtime<bf16, Cuda, M>,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    allocator: &mut GlobalKvAllocator,
    eos_ids: &[i32],
    forward: F,
) -> OpResult<StepOutput>
where
    M: DecoderModel<bf16, Cuda>,
    F: FnOnce(
        &mut Runtime<bf16, Cuda, M>,
        &StepRequest,
    ) -> OpResult<crate::domain::plan::StepOutput>,
{
    cmd.validate(runner.cap_num_tokens, runner.cap_batch)
        .map_err(|e| OpError::Shape(e.to_string()))?;
    let (mut plans, total) = plan_prefill_segments(cmd, prefilling);
    if plans.iter().any(|p| p.skipped) {
        return Err(OpError::Shape("stale eager prefill segment".into()));
    }
    let lease = allocator
        .lease(total)
        .map_err(|e| OpError::Shape(e.to_string()))?;
    let mut seqs = Vec::new();
    let mut sampling = Vec::new();
    let mut max_tokens = Vec::new();
    let mut ignore_eos = Vec::new();
    let mut kinds = Vec::new();
    build_prefill_steps_into(
        cmd,
        &mut plans,
        lease.as_slice(),
        &mut seqs,
        &mut sampling,
        &mut max_tokens,
        &mut ignore_eos,
        &mut kinds,
    );
    let mut req = StepRequest {
        stop: StopCriteria {
            eos_ids: eos_ids.to_vec(),
            generated_counts: vec![0; seqs.len()],
            max_tokens,
            ignore_eos,
        },
        seqs,
        sampling,
        draft_tokens: vec![],
    };
    let out = match forward(runner, &req) {
        Ok(out) => out,
        Err(error) => {
            lease.release(allocator);
            return Err(error);
        }
    };
    reclaim_prefill_step_data(&mut req.seqs, &mut plans);
    let mut output = StepOutput {
        tokens: vec![],
        prefill_done: vec![],
        assigned_indices: assigned_runs(cmd, &plans, false),
    };
    commit_prefill_outputs(
        cmd,
        &mut plans,
        active,
        prefilling,
        allocator,
        false,
        &out.tokens,
        &out.finished,
        0,
        &mut output,
    );
    let _ = lease.commit();
    Ok(output)
}
