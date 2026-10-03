//! Opt-in Mixed graph and admission experiments.
//!
//! Read once when a runtime is constructed. Unset options preserve the original
//! graph coverage and admission policy. Parsing and policy decisions are kept
//! independent of CUDA so their boundary cases can be tested without a GPU.

const EXTRA_TOKENS: &str = "RUSTINFER_MIXED_GRAPH_EXTRA_TOKENS";
const EXACT_DECODE: &str = "RUSTINFER_MIXED_GRAPH_EXACT_DECODE";
const SELECTED_READOUT: &str = "RUSTINFER_MIXED_GRAPH_SELECTED_READOUT";
const ADMISSION_TOKENS: &str = "RUSTINFER_MIXED_ADMISSION_TOKENS";
const MAX_PREFILLS: &str = "RUSTINFER_MIXED_MAX_PREFILLS";

#[derive(Debug, Default)]
pub(crate) struct MixedTuning {
    pub graph: MixedGraphTuning,
    // Admission is consumed by the CUDA scheduler, but parsed/tested on CPU too.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    pub admission: MixedAdmissionTuning,
}

#[derive(Debug, Default)]
pub(crate) struct MixedGraphTuning {
    extra_token_buckets: Vec<usize>,
    pub exact_decode: bool,
    pub selected_readout: bool,
}

#[derive(Debug, Default)]
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub(crate) struct MixedAdmissionTuning {
    token_budget: Option<usize>,
    max_prefills: Option<usize>,
}

impl MixedTuning {
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|key| match std::env::var(key) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(format!("{key}: {error}")),
        })
    }

    fn from_lookup(
        mut lookup: impl FnMut(&str) -> Result<Option<String>, String>,
    ) -> Result<Self, String> {
        let mut extra_token_buckets = Vec::new();
        if let Some(value) = lookup(EXTRA_TOKENS)? {
            for entry in value.split(',') {
                let tokens = positive(EXTRA_TOKENS, entry.trim())?;
                if tokens % 8 != 0 {
                    return Err(format!("{EXTRA_TOKENS}: buckets must be multiples of 8"));
                }
                extra_token_buckets.push(tokens);
            }
            extra_token_buckets.sort_unstable();
            extra_token_buckets.dedup();
        }
        Ok(Self {
            graph: MixedGraphTuning {
                extra_token_buckets,
                exact_decode: boolean(EXACT_DECODE, lookup(EXACT_DECODE)?)?,
                selected_readout: boolean(SELECTED_READOUT, lookup(SELECTED_READOUT)?)?,
            },
            admission: MixedAdmissionTuning {
                token_budget: lookup(ADMISSION_TOKENS)?
                    .map(|v| positive(ADMISSION_TOKENS, &v))
                    .transpose()?,
                max_prefills: lookup(MAX_PREFILLS)?
                    .map(|v| positive(MAX_PREFILLS, &v))
                    .transpose()?,
            },
        })
    }
}

fn positive(key: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{key}: expected a positive integer, got {value:?}"))
}

fn boolean(key: &str, value: Option<String>) -> Result<bool, String> {
    match value.as_deref() {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!("{key}: expected 0 or 1, got {value:?}")),
    }
}

impl MixedGraphTuning {
    pub fn prewarm_token_buckets(&self, defaults: &[usize]) -> Vec<usize> {
        let mut buckets = defaults.to_vec();
        buckets.extend_from_slice(&self.extra_token_buckets);
        buckets.sort_unstable();
        buckets.dedup();
        buckets
    }

    /// A finer bucket may reduce padding but must still cover every live token.
    pub fn token_bucket(&self, actual: usize, rounded_default: usize) -> usize {
        self.extra_token_buckets
            .iter()
            .copied()
            .find(|&bucket| bucket >= actual && bucket < rounded_default)
            .unwrap_or(rounded_default)
    }

    pub fn warmup_prefixes(
        &self,
        capture_sizes: &[usize],
        cap_batch: usize,
        max_prefix: usize,
    ) -> Vec<usize> {
        if self.exact_decode {
            (1..=cap_batch.min(max_prefix.saturating_add(1))).collect()
        } else {
            capture_sizes.to_vec()
        }
    }
}

#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
impl MixedAdmissionTuning {
    /// Select a FIFO prefix of whole scheduler commands, without allocating KV.
    /// The first command bypasses soft limits to guarantee progress. The caller's
    /// forward-group packer must still enforce hard token/sequence capacities.
    pub fn admitted_prefix(
        &self,
        commands: impl IntoIterator<Item = (usize, usize)>,
        decode_slot: usize,
        legacy_budget: Option<usize>,
        capacity: usize,
    ) -> usize {
        let budget = self
            .token_budget
            .or(legacy_budget)
            .unwrap_or(capacity)
            .min(capacity);
        let max_prefills = self.max_prefills.unwrap_or(usize::MAX);
        let mut tokens = 0usize;
        let mut rows = 0usize;
        let mut admitted = 0usize;
        for (next_tokens, next_rows) in commands {
            if admitted != 0
                && (decode_slot
                    .saturating_add(tokens)
                    .saturating_add(next_tokens)
                    > budget
                    || rows.saturating_add(next_rows) > max_prefills)
            {
                break;
            }
            tokens = tokens.saturating_add(next_tokens);
            rows = rows.saturating_add(next_rows);
            admitted += 1;
        }
        admitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(entries: &[(&str, &str)]) -> Result<MixedTuning, String> {
        MixedTuning::from_lookup(|key| {
            Ok(entries
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, value)| value.to_string()))
        })
    }

    #[test]
    fn defaults_preserve_graphs_and_legacy_admission() {
        let c = config(&[]).unwrap();
        assert!(!c.graph.exact_decode && !c.graph.selected_readout);
        assert_eq!(c.graph.token_bucket(519, 576), 576);
        assert_eq!(
            c.graph.prewarm_token_buckets(&[64, 128, 384]),
            [64, 128, 384]
        );
        assert_eq!(c.graph.warmup_prefixes(&[1, 2, 4, 8], 8, 128), [1, 2, 4, 8]);
        assert_eq!(
            c.admission
                .admitted_prefix([(512, 1); 3], 8, Some(384), 4096),
            1
        );
        assert_eq!(c.admission.admitted_prefix([(512, 1); 3], 8, None, 4096), 3);
    }

    #[test]
    fn validated_buckets_are_shared_by_warmup_and_dispatch() {
        let c = config(&[
            (EXTRA_TOKENS, "2056, 520,520"),
            (EXACT_DECODE, "1"),
            (SELECTED_READOUT, "1"),
        ])
        .unwrap();
        assert_eq!(
            c.graph.prewarm_token_buckets(&[64, 128, 520]),
            [64, 128, 520, 2056]
        );
        assert_eq!(c.graph.token_bucket(519, 576), 520);
        assert_eq!(c.graph.token_bucket(521, 576), 576);
        assert_eq!(c.graph.token_bucket(2055, 2112), 2056);
        assert_eq!(
            c.graph.warmup_prefixes(&[1, 2, 4, 8], 8, 128),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert!(c.graph.selected_readout);
    }

    #[test]
    fn malformed_options_fail_with_the_setting_name() {
        for (key, value) in [
            (EXTRA_TOKENS, ""),
            (EXTRA_TOKENS, "0"),
            (EXTRA_TOKENS, "519"),
            (EXTRA_TOKENS, "520,,2056"),
            (EXACT_DECODE, "true"),
            (SELECTED_READOUT, "2"),
            (ADMISSION_TOKENS, "0"),
            (MAX_PREFILLS, "-1"),
            (MAX_PREFILLS, "abc"),
        ] {
            assert!(config(&[(key, value)]).unwrap_err().contains(key));
        }
    }

    #[test]
    fn admission_is_independent_of_graph_coverage() {
        let c = config(&[(ADMISSION_TOKENS, "4096"), (MAX_PREFILLS, "2")]).unwrap();
        assert_eq!(
            c.admission
                .admitted_prefix([(512, 1); 3], 8, Some(384), 4096),
            2
        );
        assert_eq!(
            c.admission
                .admitted_prefix([(2048, 1); 3], 8, Some(384), 4096),
            1
        );
        assert_eq!(
            c.admission
                .admitted_prefix([(2048, 1); 3], 0, Some(384), 4096),
            2
        );
        assert_eq!(
            c.admission
                .admitted_prefix([(512, 2), (512, 1)], 8, None, 4096),
            1
        );
    }

    #[test]
    fn admission_preserves_fifo_and_atomic_first_command() {
        let c = config(&[(ADMISSION_TOKENS, "1536"), (MAX_PREFILLS, "2")]).unwrap();
        assert_eq!(
            c.admission
                .admitted_prefix([(512, 1), (2048, 1), (128, 1)], 8, None, 4096),
            1
        );
        assert_eq!(
            c.admission
                .admitted_prefix([(2048, 3), (512, 1)], 8, None, 4096),
            1
        );
        assert_eq!(c.admission.admitted_prefix([], 8, None, 4096), 0);
    }

    #[test]
    fn admission_respects_capacity_even_if_override_is_larger() {
        let c = config(&[(ADMISSION_TOKENS, "8192")]).unwrap();
        assert_eq!(
            c.admission.admitted_prefix([(2048, 1); 2], 8, None, 4096),
            1
        );
        assert_eq!(
            c.admission.admitted_prefix([(2048, 1); 2], 8, None, 4160),
            2
        );
        assert_eq!(
            c.admission
                .admitted_prefix([(usize::MAX, 1), (1, 1)], 8, None, 4096),
            1
        );
    }
}
