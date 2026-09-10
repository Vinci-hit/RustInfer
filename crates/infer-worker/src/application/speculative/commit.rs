//! Commit exactly the retained target prefix to worker-owned KV leases.
use crate::application::worker_state::ActiveSeqMap;
use crate::domain::global_kv_alloc::{GlobalKvAllocator, KvLease};
use crate::domain::plan::StepOutput;
use crate::domain::ports::{OpError, OpResult};
use infer_protocol::worker_to_scheduler_data::{
    AssignedIndices, GeneratedToken, StepOutput as WireOutput,
};

pub(crate) fn commit_decode(
    active: &mut ActiveSeqMap,
    allocator: &mut GlobalKvAllocator,
    id: u64,
    mut lease: KvLease,
    output: StepOutput,
) -> OpResult<WireOutput> {
    let checked = (|| {
        let seq = active
            .get(&id)
            .ok_or_else(|| OpError::Shape("missing speculative sequence".into()))?;
        if output.tokens.len() != 1
            || output.materialized_tokens.len() != 1
            || output.finished.len() != 1
            || output.accepted_drafts.as_ref().is_none_or(|a| a.len() != 1)
        {
            return Err(OpError::Shape("invalid speculative commit rows".into()));
        }
        let kept = output.materialized_tokens[0] as usize;
        if kept == 0
            || kept != output.tokens[0].len()
            || kept > lease.len()
            || seq
                .generated_count
                .checked_add(kept)
                .is_none_or(|g| g > seq.max_tokens)
            || seq.kv_len.checked_add(kept).is_none()
        {
            return Err(OpError::Shape("invalid speculative commit length".into()));
        }
        Ok(kept)
    })();
    let kept = match checked {
        Ok(n) => n,
        Err(e) => {
            lease.release(allocator);
            return Err(e);
        }
    };
    lease.shrink_to(kept, allocator);
    let tokens = &output.tokens[0];
    let last = tokens.last().unwrap().token_id;
    if let Err(e) = active
        .get_mut(&id)
        .unwrap()
        .commit_accepted(last, kept, lease.as_slice())
    {
        lease.release(allocator);
        return Err(OpError::Shape(e.into()));
    }
    let indices = lease.commit();
    let mut assigned = Vec::new();
    let mut start = 0;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len()
            && end - start < u16::MAX as usize
            && indices[end] == indices[end - 1] + 1
        {
            end += 1;
        }
        assigned.push(AssignedIndices {
            sequence_id: id,
            base: indices[start],
            len: (end - start) as u16,
            token_ids: vec![],
        });
        start = end;
    }
    if output.finished[0] {
        allocator.release_owned(&active.remove(&id).unwrap().block_table, false);
    }
    Ok(WireOutput {
        prefill_done: vec![],
        assigned_indices: assigned,
        tokens: tokens
            .iter()
            .enumerate()
            .map(|(i, t)| GeneratedToken {
                sequence_id: id,
                token_id: t.token_id,
                finished: output.finished[0] && i + 1 == tokens.len(),
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::worker_state::ActiveSeq;
    use crate::domain::plan::SampledToken;
    #[test]
    fn malformed_commit_returns_all_provisional_slots_without_changing_the_sequence() {
        let mut allocator = GlobalKvAllocator::new(16);
        let prefix = allocator.lease(3).unwrap().commit();
        let mut active = ActiveSeqMap::new();
        active.insert(
            7,
            ActiveSeq::new(4, prefix.clone(), 10, false, Default::default()),
        );
        let lease = allocator.lease(4).unwrap();
        let result = StepOutput {
            tokens: vec![vec![]],
            materialized_tokens: vec![2],
            accepted_drafts: Some(vec![1]),
            finished: vec![false],
            hidden_tap: None,
        };
        assert!(commit_decode(&mut active, &mut allocator, 7, lease, result).is_err());
        assert_eq!(allocator.outstanding(), 3);
        assert_eq!(active[&7].block_table, prefix);
        assert_eq!(active[&7].generated_count, 1);
        assert_eq!(active[&7].last_token, 4);
    }

    #[test]
    fn rejection_releases_suffix_and_finishing_releases_retained_history_once() {
        for finished in [false, true] {
            let mut allocator = GlobalKvAllocator::new(16);
            let prefix = allocator.lease(3).unwrap().commit();
            let mut active = ActiveSeqMap::new();
            active.insert(7, ActiveSeq::new(4, prefix, 10, false, Default::default()));
            let lease = allocator.lease(4).unwrap();
            let result = StepOutput {
                tokens: vec![
                    vec![5, 6]
                        .into_iter()
                        .map(|token_id| SampledToken {
                            token_id,
                            logprob: 0.0,
                            top_logprobs: vec![],
                        })
                        .collect(),
                ],
                materialized_tokens: vec![2],
                accepted_drafts: Some(vec![1]),
                finished: vec![finished],
                hidden_tap: None,
            };
            let wire = commit_decode(&mut active, &mut allocator, 7, lease, result).unwrap();
            assert_eq!(
                wire.assigned_indices
                    .iter()
                    .map(|a| a.len as usize)
                    .sum::<usize>(),
                2
            );
            assert_eq!(
                wire.tokens.iter().map(|t| t.finished).collect::<Vec<_>>(),
                [false, finished]
            );
            assert_eq!(allocator.outstanding(), if finished { 0 } else { 5 });
            if !finished {
                assert_eq!(active[&7].generated_count, 3);
                assert_eq!(active[&7].kv_len, 5);
                assert_eq!(active[&7].last_token, 6);
            }
        }
    }
}
