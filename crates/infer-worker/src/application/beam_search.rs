//! Deterministic, cached text beam search. Sampling filters do not apply to
//! beam scores: every edge uses the full model's log-softmax probability.
use crate::application::runtime::Runtime;
use crate::application::sampler_stack::GreedySampler;
use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::DecoderModel;
use crate::domain::plan::{SeqStep, StepRequest, StopCriteria};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    pub width: usize,
    pub max_context: usize,
    pub max_step_tokens: usize,
    pub max_new_tokens: usize,
    /// Score = sum(log p) / generated_length.powf(length_penalty).
    /// Generated length includes EOS, excludes the prompt.
    pub length_penalty: f64,
    pub eos_ids: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeamHypothesis {
    pub token_ids: Vec<i32>,
    pub logprob: f64,
    pub score: f64,
    pub ended_with_eos: bool,
}

/// Owns a dedicated runtime, not a scheduler's live request state. KV prefixes
/// are shared read-only through reference-counted token slots (block_size=1).
/// Branches only write fresh slots; GDN states fork via preallocated snapshots.
/// Results are returned after search, since a winning prefix may change later.
pub struct BeamSession<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> {
    runtime: Runtime<T, D, M>,
    config: BeamSearchConfig,
    ids: Tensor<i32, D>,
    logprobs: Tensor<f32, D>,
    slots: Vec<u32>,
    poisoned: bool,
}

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> BeamSession<T, D, M> {
    pub fn new(model: M, scope: D::Scope, config: BeamSearchConfig) -> OpResult<Self> {
        let vocab = model.dims().vocab_size;
        if config.width == 0
            || config.width > vocab
            || config.max_context == 0
            || config.max_context > i32::MAX as usize
            || config.max_new_tokens == 0
            || config.max_new_tokens > config.max_context
            || config.max_step_tokens < config.width
            || config.max_step_tokens > config.max_context
            || !config.length_penalty.is_finite()
            || config.length_penalty < 0.0
            || scope.topology().tp.size != 1
            || config
                .eos_ids
                .iter()
                .any(|&id| id < 0 || id as usize >= vocab)
        {
            return Err(OpError::Shape(
                "invalid beam configuration (text, TP=1 only)".into(),
            ));
        }
        let blocks = config
            .width
            .checked_mul(config.max_context)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| OpError::Shape("beam KV capacity overflow".into()))?;
        let k = config
            .width
            .saturating_mul(config.eos_ids.len().saturating_add(1))
            .min(vocab);
        let ids = Tensor::zeros([config.width * k], scope.device())?;
        let logprobs = Tensor::zeros([config.width * k], scope.device())?;
        let mut runtime = Runtime::new(
            model,
            scope,
            Box::new(GreedySampler),
            blocks,
            1,
            config.max_context,
            config.max_context,
            config.max_step_tokens,
            config.width,
            vec![],
        )?;
        if config.width > 1 {
            runtime.prepare_beam_forks()?;
        }
        Ok(Self {
            runtime,
            config,
            ids,
            logprobs,
            slots: (0..blocks as u32).collect(),
            poisoned: false,
        })
    }

    pub fn generate(&mut self, prompt: &[i32]) -> OpResult<Vec<BeamHypothesis>> {
        if self.poisoned
            || prompt.is_empty()
            || prompt.len().saturating_add(self.config.max_new_tokens) > self.config.max_context
            || prompt
                .iter()
                .any(|&id| id < 0 || id as usize >= self.runtime.dims.vocab_size)
        {
            return Err(OpError::Shape(
                "invalid beam prompt or context budget".into(),
            ));
        }
        let result = self.generate_inner(prompt);
        // A failed forward can leave partially updated recurrent bytes. Release
        // every owner and synchronize before slots can be reused by a new call.
        let sync = self.runtime.scope.synchronize();
        self.poisoned = sync.is_err();
        for id in 0..self.config.width {
            self.runtime.release_sequence(id as u64);
        }

        match result {
            Err(error) => Err(error),
            Ok(out) => {
                sync?;
                Ok(out)
            }
        }
    }

    fn generate_inner(&mut self, prompt: &[i32]) -> OpResult<Vec<BeamHypothesis>> {
        let mut search = BeamSearch::new(
            prompt.to_vec(),
            self.config.clone(),
            self.slots.clone(),
            vec![],
        )?;
        loop {
            if let Some(result) =
                search.step(&mut self.runtime, &mut self.ids, &mut self.logprobs)?
            {
                return Ok(result);
            }
        }
    }
}

/// Incremental search on a borrowed runtime. Each call executes at most one
/// forward, allowing the serving loop to process cancellation and heartbeats.
/// RadixTree owns prefix sharing and pruning; slots are reserved from the
/// serving allocator and returned by the caller after stream synchronization.
pub(crate) struct BeamSearch {
    config: BeamSearchConfig,
    prompt: Vec<i32>,
    stop_sequences: Vec<Vec<i32>>,
    tree: infer_core::radix_tree::RadixTree,
    free: Vec<u32>,
    owners: Vec<u64>,
    next_owner: u64,
    live: Vec<BeamHypothesis>,
    finished: Vec<BeamHypothesis>,
    candidates: Vec<Vec<(i32, f32)>>,
    len: usize,
    generated: usize,
}

impl BeamSearch {
    pub(crate) fn new(
        prompt: Vec<i32>,
        config: BeamSearchConfig,
        slots: Vec<u32>,
        stop_sequences: Vec<Vec<i32>>,
    ) -> OpResult<Self> {
        if prompt.is_empty()
            || config.width == 0
            || config.max_new_tokens == 0
            || config.max_step_tokens < config.width
            || prompt.len().saturating_add(config.max_new_tokens) > config.max_context
            || slots.len()
                < prompt
                    .len()
                    .saturating_add(config.width.saturating_mul(config.max_new_tokens))
        {
            return Err(OpError::Shape("invalid beam search capacity".into()));
        }
        let mut tree = infer_core::radix_tree::RadixTree::new();
        tree.lookup_prefix(&[], 0);
        Ok(Self {
            config,
            prompt,
            stop_sequences,
            tree,
            free: slots,
            owners: vec![0],
            next_owner: 1,
            live: vec![BeamHypothesis {
                token_ids: vec![],
                logprob: 0.0,
                score: 0.0,
                ended_with_eos: false,
            }],
            finished: vec![],
            candidates: vec![],
            len: 0,
            generated: 0,
        })
    }

    fn append(&mut self, owner: u64, token: i32) -> OpResult<()> {
        let slot = self
            .free
            .pop()
            .ok_or_else(|| OpError::Shape("beam KV pool exhausted".into()))?;
        self.tree.append_token(owner, token, slot);
        Ok(())
    }

    pub(crate) fn step<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>>(
        &mut self,
        runtime: &mut Runtime<T, D, M>,
        ids: &mut Tensor<i32, D>,
        logprobs: &mut Tensor<f32, D>,
    ) -> OpResult<Option<Vec<BeamHypothesis>>> {
        let tokens;
        if self.len < self.prompt.len() {
            let end = (self.len + self.config.max_step_tokens).min(self.prompt.len());
            let chunk = self.prompt[self.len..end].to_vec();
            for &token in &chunk {
                self.append(0, token)?;
            }
            tokens = vec![chunk];
        } else {
            self.generated += 1;
            let (next, parents) = expand_with_stops(
                &self.live,
                &self.candidates,
                &mut self.finished,
                &self.config,
                self.generated,
                &self.stop_sequences,
            );
            if next.is_empty() {
                if self.finished.is_empty() {
                    return Err(OpError::Shape(
                        "beam search produced no finite hypotheses".into(),
                    ));
                }
                return Ok(Some(std::mem::take(&mut self.finished)));
            }
            let mut owners = Vec::with_capacity(next.len());
            for &parent in &parents {
                let id = self.next_owner;
                self.next_owner += 1;
                if !self.tree.fork_sequence(self.owners[parent], id) {
                    return Err(OpError::Shape("beam prefix fork failed".into()));
                }
                owners.push(id);
            }
            for &owner in &self.owners {
                self.tree.mark_finished_chain(owner);
            }
            self.free.extend(self.tree.evict(usize::MAX));
            runtime.fork_beams(self.live.len(), &parents, self.len as i32)?;
            tokens = next
                .iter()
                .map(|b| vec![*b.token_ids.last().unwrap()])
                .collect::<Vec<_>>();
            for (&owner, token) in owners.iter().zip(&tokens) {
                self.append(owner, token[0])?;
            }
            self.owners = owners;
            self.live = next;
        }
        let mut tables = Vec::with_capacity(self.owners.len());
        for &owner in &self.owners {
            let mut table = Vec::with_capacity(self.len + tokens[0].len());
            if !self.tree.sequence_indices(owner, &mut table) {
                return Err(OpError::Shape("missing beam prefix".into()));
            }
            tables.push(table);
        }
        let req = request(&tokens, &tables, self.len);
        let k = self
            .config
            .width
            .saturating_mul(self.config.eos_ids.len().saturating_add(1))
            .min(runtime.dims.vocab_size);
        self.candidates = runtime.beam_step(&req, k, ids, logprobs)?;
        self.len += tokens[0].len();
        Ok(None)
    }
}

fn request(tokens: &[Vec<i32>], tables: &[Vec<u32>], start: usize) -> StepRequest {
    StepRequest {
        seqs: tokens
            .iter()
            .zip(tables)
            .enumerate()
            .map(|(id, (tokens, table))| SeqStep {
                sequence_id: id as u64,
                input_ids: tokens.clone(),
                positions: (start as i32..(start + tokens.len()) as i32).collect(),
                kv_write_start: start as i32,
                kv_len_after: (start + tokens.len()) as i32,
                block_table: table.clone(),
            })
            .collect(),
        sampling: vec![],
        draft_tokens: vec![],
        stop: StopCriteria {
            eos_ids: vec![],
            generated_counts: vec![],
            max_tokens: vec![],
            ignore_eos: vec![],
        },
    }
}

fn sort_hypotheses(beams: &mut Vec<BeamHypothesis>, width: usize) {
    beams.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.token_ids.cmp(&b.token_ids))
    });
    beams.truncate(width);
}

#[cfg(test)]
fn expand(
    live: &[BeamHypothesis],
    rows: &[Vec<(i32, f32)>],
    finished: &mut Vec<BeamHypothesis>,
    config: &BeamSearchConfig,
    generated: usize,
) -> (Vec<BeamHypothesis>, Vec<usize>) {
    expand_with_stops(live, rows, finished, config, generated, &[])
}

fn expand_with_stops(
    live: &[BeamHypothesis],
    rows: &[Vec<(i32, f32)>],
    finished: &mut Vec<BeamHypothesis>,
    config: &BeamSearchConfig,
    generated: usize,
    stops: &[Vec<i32>],
) -> (Vec<BeamHypothesis>, Vec<usize>) {
    let mut expansions = Vec::new();
    for (parent, (beam, row)) in live.iter().zip(rows).enumerate() {
        for &(token, logprob) in row {
            if !logprob.is_finite() {
                continue;
            }
            expansions.push((parent, token, beam.logprob + f64::from(logprob)));
        }
    }
    expansions.sort_unstable_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| live[a.0].token_ids.cmp(&live[b.0].token_ids))
            .then_with(|| a.1.cmp(&b.1))
    });
    // Width one follows greedy exactly, including immediate EOS termination.
    if config.width == 1 {
        expansions.truncate(1);
    }
    let mut next = Vec::with_capacity(config.width);
    let mut parents = Vec::with_capacity(config.width);
    for (parent, token, logprob) in expansions {
        let eos = config.eos_ids.contains(&token);
        let mut token_ids = live[parent].token_ids.clone();
        token_ids.push(token);
        let stopped = stops
            .iter()
            .any(|stop| !stop.is_empty() && token_ids.ends_with(stop));
        if !eos && !stopped && generated < config.max_new_tokens && next.len() == config.width {
            continue;
        }
        let beam = BeamHypothesis {
            token_ids,
            logprob,
            score: logprob / (generated as f64).powf(config.length_penalty),
            ended_with_eos: eos,
        };
        if eos || stopped || generated == config.max_new_tokens {
            finished.push(beam);
        } else {
            next.push(beam);
            parents.push(parent);
        }
    }
    sort_hypotheses(finished, config.width);
    // Conservative upper bound: future log probabilities cannot increase the
    // sum, and the maximum output length gives the largest allowed denominator.
    if finished.len() == config.width
        && next.iter().all(|b| {
            b.logprob / (config.max_new_tokens as f64).powf(config.length_penalty)
                < finished.last().unwrap().score
        })
    {
        next.clear();
        parents.clear();
    }
    (next, parents)
}

pub(crate) fn top_candidates<T: Dtype>(row: &[T], k: usize) -> OpResult<Vec<(i32, f32)>> {
    if row.is_empty() || k == 0 || k > row.len() {
        return Err(OpError::Shape("invalid candidate count".into()));
    }
    let mut scores: Vec<_> = row
        .iter()
        .enumerate()
        .map(|(id, x)| {
            let x = T::read_f64(x);
            (id as i32, if x.is_nan() { f64::NEG_INFINITY } else { x })
        })
        .collect();
    let max = scores.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    let delta = |s: f64| {
        if max == f64::INFINITY {
            if s == max { 0.0 } else { f64::NEG_INFINITY }
        } else if max == f64::NEG_INFINITY {
            0.0
        } else {
            s - max
        }
    };
    let logsum = scores.iter().map(|s| delta(s.1).exp()).sum::<f64>().ln();
    let order = |a: &(i32, f64), b: &(i32, f64)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if k < scores.len() {
        scores.select_nth_unstable_by(k, order);
        scores.truncate(k);
    }
    scores.sort_unstable_by(order);
    Ok(scores
        .into_iter()
        .map(|(id, s)| (id, (delta(s) - logsum) as f32))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(width: usize) -> BeamSearchConfig {
        BeamSearchConfig {
            width,
            max_context: 16,
            max_step_tokens: 4,
            max_new_tokens: 3,
            length_penalty: 1.0,
            eos_ids: vec![0],
        }
    }
    fn root() -> Vec<BeamHypothesis> {
        vec![BeamHypothesis {
            token_ids: vec![],
            logprob: 0.0,
            score: 0.0,
            ended_with_eos: false,
        }]
    }
    #[test]
    fn multi_token_stop_finishes_the_hypothesis() {
        let mut live = root();
        live[0].token_ids = vec![5];
        let mut finished = vec![];
        let (next, _) = expand_with_stops(
            &live,
            &[vec![(6, -0.1), (7, -0.2)]],
            &mut finished,
            &config(1),
            2,
            &[vec![5, 6]],
        );
        assert!(next.is_empty());
        assert_eq!(finished[0].token_ids, [5, 6]);
        assert!(!finished[0].ended_with_eos);
    }
    #[test]
    fn width_one_stops_at_greedy_eos() {
        let mut finished = vec![];
        let (live, parents) = expand(
            &root(),
            &[vec![(0, -0.5), (1, -0.9)]],
            &mut finished,
            &config(1),
            1,
        );
        assert!(live.is_empty() && parents.is_empty());
        assert_eq!(finished[0].token_ids, [0]);
        assert!(finished[0].ended_with_eos);
    }
    #[test]
    fn forks_parent_and_preserves_finished_eos() {
        let mut finished = vec![];
        let (live, parents) = expand(
            &root(),
            &[vec![(1, -0.4), (2, -0.6), (0, -0.8)]],
            &mut finished,
            &config(2),
            1,
        );
        assert_eq!(parents, [0, 0]);
        assert_eq!(live.len(), 2);
        let (next, _) = expand(
            &live,
            &[vec![(0, -0.1), (3, -2.0)], vec![(4, -0.2), (0, -0.3)]],
            &mut finished,
            &config(2),
            3,
        );
        assert!(next.is_empty());
        assert_eq!(finished[0].token_ids, [1, 0]);
        assert_eq!(finished.len(), 2);
    }
    #[test]
    fn candidate_probabilities_use_full_vocabulary_and_stable_ties() {
        let top = top_candidates(&[0f32, 0., 0., 0.], 2).unwrap();
        assert_eq!(top.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        assert!((top[0].1 + 4f32.ln()).abs() < 1e-6);
        let inf = top_candidates(&[f32::INFINITY, 0., f32::INFINITY], 2).unwrap();
        assert!((inf[0].1 + 2f32.ln()).abs() < 1e-6);
    }
}
