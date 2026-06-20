//! Relocated from infer-backend-abi's `kv.rs`: exercises `PagedKvPool` against a
//! concrete backend type (`Cpu`, which lives in the worker) through its public
//! API.
#[cfg(test)]
mod tests {
    use infer_core::kv::*;
    use crate::infrastructure::cpu::Cpu;
    use std::collections::HashMap;

    fn empty_pool() -> PagedKvPool<f32, Cpu> {
        PagedKvPool {
            layers: Vec::new(),
            num_blocks: 0,
            block_size: 1,
            kv_dim: 0,
            quant: KvQuantTier::None,
            seq_kv_len: HashMap::new(),
        }
    }

    #[test]
    fn apply_step_appends_accepted_counts() {
        let mut pool = empty_pool();
        pool.seq_kv_len.insert(7, 5);

        pool.edit().apply_step(&[7], &[2], &[0]).unwrap();

        assert_eq!(pool.seq_kv_len.get(&7), Some(&7));
    }

    #[test]
    fn apply_step_truncates_rejected_spec_tail() {
        let mut pool = empty_pool();
        pool.seq_kv_len.insert(7, 8);

        pool.edit().apply_step(&[7], &[2], &[4]).unwrap();

        assert_eq!(pool.seq_kv_len.get(&7), Some(&6));
    }

    #[test]
    fn apply_step_keeps_fully_accepted_spec_tail() {
        let mut pool = empty_pool();
        pool.seq_kv_len.insert(7, 8);

        pool.edit().apply_step(&[7], &[4], &[4]).unwrap();

        assert_eq!(pool.seq_kv_len.get(&7), Some(&8));
    }

    #[test]
    fn apply_step_rejects_overaccepted_spec_tail() {
        let mut pool = empty_pool();
        pool.seq_kv_len.insert(7, 8);

        let err = pool.edit().apply_step(&[7], &[5], &[4]).unwrap_err();

        assert!(format!("{err:?}").contains("accepted 5 > speculative 4"));
    }

    #[test]
    fn apply_step_rejects_sid_accepted_mismatch() {
        let mut pool = empty_pool();
        let err = pool.edit().apply_step(&[1, 2], &[1], &[0]).unwrap_err();

        assert!(format!("{err:?}").contains("sids=2 accepted=1"));
    }
}
