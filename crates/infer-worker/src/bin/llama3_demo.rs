//! `llama3_demo` — minimal end-to-end Llama 3 text generation.
//!
//! Loads weights from a HuggingFace-format safetensors model, tokenizes a
//! prompt, runs prefill + decode on CUDA, and prints the generated text.
//! Bypasses the ZMQ scheduler entirely — this is purely to validate the
//! forward / weight-loading path.
//!
//! Usage:
//!   llama3_demo --model-path /path/to/llama3.2-1b \
//!               --prompt "Once upon a time" \
//!               --max-new-tokens 32 \
//!               [--device cuda:0]

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use half::bf16;
use serde::Deserialize;

use infer_worker::application::model_runner::ModelRunner;
use infer_worker::infrastructure::cuda::Cuda;
use infer_worker::infrastructure::io::SafetensorsReader;
use infer_worker::models::loader::{LoadConfig, RopeScaling, WeightLoader};

#[derive(Parser, Debug)]
#[command(name = "llama3_demo", version)]
struct Args {
    /// Path to a HuggingFace Llama 3 model directory containing
    /// `config.json`, `model.safetensors`, and `tokenizer.json`.
    #[arg(long)]
    model_path: PathBuf,

    /// User prompt. Combined with `--system` if `--chat` is set, otherwise
    /// fed to the model as raw text. Ignored if `--prompt-ids-file` is set.
    #[arg(long, default_value = "")]
    prompt: String,

    /// Optional system prompt (used only when `--chat` is set).
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Wrap the input in the Llama-3 chat template (system + user headers).
    /// When false, `prompt` is encoded as raw text (no special tokens beyond
    /// the tokenizer's default BOS).
    #[arg(long, default_value_t = false)]
    chat: bool,

    /// JSON file containing a flat array of token ids (e.g. produced by
    /// `tokenizer.apply_chat_template(..., tokenize=True)`). When set,
    /// bypasses Rust-side tokenization for byte-exact reproducibility with
    /// a reference HuggingFace run.
    #[arg(long)]
    prompt_ids_file: Option<PathBuf>,

    /// JSON file containing an array of arrays of token ids — runs all
    /// prompts as a single ragged batch (one prefill step covering all
    /// sequences). For Phase 3 batched-path validation; mutually exclusive
    /// with `--prompt-ids-file`.
    #[arg(long)]
    batch_prompts_file: Option<PathBuf>,

    /// How many new tokens to generate (greedy / argmax).
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// CUDA device specifier (only `cuda:N` supported).
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// KV cache capacity (max prompt + generated tokens).
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,
}

/// Build the canonical Llama-3 chat-template string for a single
/// system + user turn, matching `tokenizer_config.json`'s template at
/// `<|start_header_id|>system<|end_header_id|>...<|eot_id|>` etc.
///
/// The string is intentionally fed to `tokenizer.encode(..., add_special_tokens=true)`
/// so the tokenizer prepends its configured BOS (`<|begin_of_text|>`) once.
fn render_llama3_chat(system: &str, user: &str) -> String {
    let date = "26 Jul 2024"; // matches the template's static fallback
    format!(
        "<|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: December 2023\n\
         Today Date: {date}\n\n\
         {system}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         {user}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n",
        date = date,
        system = system.trim(),
        user = user.trim(),
    )
}

/// Subset of HuggingFace `config.json` we need to build the model.
#[derive(Debug, Deserialize)]
struct HfConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_max_position")]
    max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    rope_scaling: Option<HfRopeScaling>,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    architectures: Vec<String>,
}

fn default_max_position() -> usize { 4096 }
fn default_rms_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f64 { 10000.0 }

#[derive(Debug, Deserialize)]
struct HfRopeScaling {
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    low_freq_factor: Option<f32>,
    #[serde(default)]
    high_freq_factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<u32>,
}

fn parse_device_id(spec: &str) -> Result<i32> {
    let suffix = spec.strip_prefix("cuda:")
        .ok_or_else(|| anyhow!("expected 'cuda:N', got '{}'", spec))?;
    suffix.parse().with_context(|| format!("invalid device id in '{}'", spec))
}

fn build_load_config(cfg: &HfConfig, max_seq_len: usize) -> LoadConfig {
    let head_dim = cfg.head_dim
        .unwrap_or_else(|| cfg.hidden_size / cfg.num_attention_heads);
    let rope_scaling = cfg.rope_scaling.as_ref().and_then(|rs| {
        // Only Llama-3 NTK rescaling is supported; fall back to bare RoPE for
        // anything else (e.g. linear, dynamic).
        let is_llama3 = rs.rope_type.as_deref() == Some("llama3");
        let factor = rs.factor?;
        let low = rs.low_freq_factor?;
        let high = rs.high_freq_factor?;
        let orig = rs.original_max_position_embeddings?;
        if !is_llama3 { return None; }
        Some(RopeScaling {
            factor,
            low_freq_factor: low,
            high_freq_factor: high,
            original_max_position_embeddings: orig,
        })
    });
    LoadConfig {
        dim: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        layer_num: cfg.num_hidden_layers,
        head_num: cfg.num_attention_heads,
        kv_head_num: cfg.num_key_value_heads,
        head_dim,
        vocab_size: cfg.vocab_size,
        // KV cache size and RoPE table size both honor the runtime cap.
        seq_len: max_seq_len.max(cfg.max_position_embeddings.min(max_seq_len)),
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        rope_scaling,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("[demo] model_path  = {}", args.model_path.display());
    eprintln!("[demo] prompt      = {:?}", args.prompt);
    eprintln!("[demo] max_new     = {}", args.max_new_tokens);
    eprintln!("[demo] device      = {}", args.device);

    // 1. Parse config.json.
    let config_path = args.model_path.join("config.json");
    let config_bytes = std::fs::read(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;
    let hf_cfg: HfConfig = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("parse {}", config_path.display()))?;
    let arch = hf_cfg.architectures.first().cloned().unwrap_or_default();
    eprintln!("[demo] architecture = {} ({} layers, dim={}, heads={}/{}, vocab={})",
        arch, hf_cfg.num_hidden_layers, hf_cfg.hidden_size,
        hf_cfg.num_attention_heads, hf_cfg.num_key_value_heads, hf_cfg.vocab_size);

    let load_cfg = build_load_config(&hf_cfg, args.max_seq_len);
    if load_cfg.rope_scaling.is_some() {
        eprintln!("[demo] rope_scaling = llama3 NTK enabled");
    }

    // 2. Initialize CUDA device.
    let device_id = parse_device_id(&args.device)?;
    let cuda = Cuda::new(device_id)
        .map_err(|e| anyhow!("Cuda::new({}): {:?}", device_id, e))?;
    eprintln!("[demo] cuda device  = id {}", device_id);

    // 3. Open weights and build the model.
    let st_path = &args.model_path;
    let reader = SafetensorsReader::open(st_path)
        .map_err(|e| anyhow!("open weights: {}", e))?;
    eprintln!("[demo] tensors      = {} (loading...)", reader.names().len());

    let loader = WeightLoader::new(&reader);
    let load_start = std::time::Instant::now();
    let model = loader.load_llama3::<bf16, Cuda>(&load_cfg, &cuda)
        .map_err(|e| anyhow!("load_llama3: {:?}", e))?;
    let _ = hf_cfg.tie_word_embeddings; // already handled inside load_llama3
    eprintln!("[demo] weights loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // 4. Build runner. If batch-prompts mode is on, allocate enough slots.
    let batch_prompts: Option<Vec<Vec<i32>>> =
        if let Some(p) = args.batch_prompts_file.as_ref() {
            let bytes = std::fs::read(p)
                .with_context(|| format!("read {}", p.display()))?;
            let v: Vec<Vec<i64>> = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {} as nested JSON array", p.display()))?;
            Some(
                v.into_iter()
                    .map(|s| s.into_iter().map(|i| i as i32).collect::<Vec<_>>())
                    .collect(),
            )
        } else {
            None
        };
    let max_batch_seqs = batch_prompts.as_ref().map(|v| v.len().max(1)).unwrap_or(1);
    // Paged KV pool sizing: block_size=1 — every token gets its own slot in
    // the global KV pool. With block_size=1 the paged scatter/attention
    // kernels degenerate to per-token gather (block_table[seq][i] == that
    // sequence's i-th token's global KV index), which is the foundation of
    // the worker-owned `GlobalKvAllocator` design.
    let block_size: usize = 1;
    let max_blocks_per_seq = (args.max_seq_len + block_size - 1) / block_size;
    let num_blocks = max_blocks_per_seq * max_batch_seqs;
    let pool_blocks = num_blocks + 1;
    eprintln!(
        "[demo] paged KV pool: block_size={} num_blocks={} pool_blocks={} max_blocks_per_seq={}",
        block_size, num_blocks, pool_blocks, max_blocks_per_seq,
    );
    // Forward workspace caps: cap_num_tokens = max_seq_len for prefill of a
    // long single prompt (single-batch demo); cap_batch = max_batch_seqs.
    let cap_num_tokens = args.max_seq_len;
    let cap_batch = max_batch_seqs;
    let flash_decode_capacity_f32 =
        infer_worker::infrastructure::cuda::kernels::attention_paged::flash_decode_workspace_capacity_f32(
            cap_batch.max(1), 128, 256,
        );
    let mut runner = ModelRunner::new(
        model, cuda, pool_blocks, block_size, max_blocks_per_seq, args.max_seq_len,
        cap_num_tokens, cap_batch, flash_decode_capacity_f32,
        vec![1, 2, 4, 8, 16, 32],
    ).map_err(|e| anyhow!("ModelRunner::new: {:?}", e))?;

    // Prime CUDA Graphs (decode-only).
    if let Err(e) = runner.prime_graphs_cuda() {
        eprintln!("[demo] CUDA Graph priming FAILED, continuing eager: {:?}", e);
    } else {
        eprintln!("[demo] CUDA Graphs primed for {:?}", runner.capture_sizes);
    }

    if let Some(prompts) = batch_prompts {
        // ─── Staggered test takes priority ───
        if std::env::var("RUSTINFER_TEST_STAGGERED").is_ok() && prompts.len() >= 2 {
            let tok_path = args.model_path.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {:?}", e))?;
            let eos_ids: &[i32] = &[128001, 128008, 128009];
            let p0 = &prompts[0];
            let p1 = &prompts[1];
            let bt0: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
            let bt1: Vec<u32> = (max_blocks_per_seq as u32..2 * max_blocks_per_seq as u32).collect();

            eprintln!("[staggered] p0 len={}, p1 len={}", p0.len(), p1.len());

            // 1. Prefill req0 alone
            let prefill0 = infer_worker::application::model_runner::SeqStep {
                input_ids: p0.clone(),
                positions: (0..p0.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p0.len() as i32,
                block_table: bt0.clone(),
            };
            let t0_first = runner.step_batch(&[prefill0])
                .map_err(|e| anyhow!("staggered prefill0: {:?}", e))?[0];
            eprintln!("[staggered] prefill0 done, first_token={}", t0_first);

            // 2. Decode req0 alone 5 steps
            let mut t0_tokens = vec![t0_first];
            let mut t0_last = t0_first;
            for i in 0..5 {
                let pos = (p0.len() + i) as i32;
                let step = infer_worker::application::model_runner::SeqStep {
                    input_ids: vec![t0_last],
                    positions: vec![pos],
                    kv_write_start: pos,
                    kv_len_after: pos + 1,
                    block_table: bt0.clone(),
                };
                t0_last = runner.step_batch(&[step])
                    .map_err(|e| anyhow!("staggered decode0 step {}: {:?}", i, e))?[0];
                t0_tokens.push(t0_last);
            }
            eprintln!("[staggered] decode0 x5: {:?}", t0_tokens);

            // 3. Prefill req1 alone
            let prefill1 = infer_worker::application::model_runner::SeqStep {
                input_ids: p1.clone(),
                positions: (0..p1.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p1.len() as i32,
                block_table: bt1.clone(),
            };
            let t1_first = runner.step_batch(&[prefill1])
                .map_err(|e| anyhow!("staggered prefill1: {:?}", e))?[0];
            eprintln!("[staggered] prefill1 done, first_token={}", t1_first);
            let mut t1_tokens = vec![t1_first];
            let mut t1_last = t1_first;

            // 4. Decode [req0, req1] together 10 steps
            for i in 0..10 {
                let pos0 = (p0.len() + 5 + i) as i32;
                let pos1 = (p1.len() + i) as i32;
                let steps = vec![
                    infer_worker::application::model_runner::SeqStep {
                        input_ids: vec![t0_last],
                        positions: vec![pos0],
                        kv_write_start: pos0,
                        kv_len_after: pos0 + 1,
                        block_table: bt0.clone(),
                    },
                    infer_worker::application::model_runner::SeqStep {
                        input_ids: vec![t1_last],
                        positions: vec![pos1],
                        kv_write_start: pos1,
                        kv_len_after: pos1 + 1,
                        block_table: bt1.clone(),
                    },
                ];
                let new = runner.step_batch(&steps)
                    .map_err(|e| anyhow!("staggered joint decode step {}: {:?}", i, e))?;
                t0_last = new[0];
                t1_last = new[1];
                t0_tokens.push(t0_last);
                t1_tokens.push(t1_last);
            }

            let txt0 = tokenizer.decode(&t0_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), false).unwrap_or_default();
            let txt1 = tokenizer.decode(&t1_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), false).unwrap_or_default();
            println!("[staggered-output] seq=0 tokens={:?}", t0_tokens);
            println!("[staggered-text]   seq=0 text={:?}", &txt0[..txt0.len().min(200)]);
            println!("[staggered-output] seq=1 tokens={:?}", t1_tokens);
            println!("[staggered-text]   seq=1 text={:?}", &txt1[..txt1.len().min(200)]);
            return Ok(());
        }

        // ─── Normal sync batch mode ───
        eprintln!("[demo] BATCH MODE: {} prompts", prompts.len());
        let tok_path = args.model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("load tokenizer: {:?}", e))?;
        let mut steps = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            eprintln!("[demo]   prompt[{}] len={} ids={:?}", i, p.len(), p);
            // Each seq gets a contiguous block_table starting at i * max_blocks_per_seq.
            let bt_start = (i * max_blocks_per_seq) as u32;
            let block_table: Vec<u32> = (0..max_blocks_per_seq as u32)
                .map(|b| bt_start + b)
                .collect();
            steps.push(infer_worker::application::model_runner::SeqStep {
                input_ids: p.clone(),
                positions: (0..p.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p.len() as i32,
                block_table,
            });
        }
        let gen_start = std::time::Instant::now();
        let first_tokens = runner.step_batch_with_graph(&steps)
            .map_err(|e| anyhow!("step_batch: {:?}", e))?;
        eprintln!("[demo] batch prefill: {:.2}s, first tokens: {:?}",
            gen_start.elapsed().as_secs_f32(), first_tokens);

        // Decode loop for batch mode
        let batch_size = prompts.len();
        let mut all_generated: Vec<Vec<i32>> = (0..batch_size)
            .map(|i| vec![first_tokens[i]])
            .collect();
        let mut last_tokens = first_tokens.clone();
        let eos_ids: &[i32] = &[128001, 128008, 128009];
        let mut active: Vec<bool> = vec![true; batch_size];

        for step in 0..args.max_new_tokens.saturating_sub(1) {
            // Check if all done
            if active.iter().all(|&a| !a) { break; }

            // Build decode steps for all active sequences
            let mut decode_steps = Vec::new();
            for i in 0..batch_size {
                if !active[i] { continue; }
                let kv_write_start = (prompts[i].len() + all_generated[i].len() - 1) as i32;
                let kv_len_after = kv_write_start + 1;
                let bt_start = (i * max_blocks_per_seq) as u32;
                let block_table: Vec<u32> = (0..max_blocks_per_seq as u32)
                    .map(|b| bt_start + b)
                    .collect();
                decode_steps.push(infer_worker::application::model_runner::SeqStep {
                    input_ids: vec![last_tokens[i]],
                    positions: vec![kv_write_start],
                    kv_write_start,
                    kv_len_after,
                    block_table,
                });
            }

            let new_tokens = runner.step_batch_with_graph(&decode_steps)
                .map_err(|e| anyhow!("step_batch decode step {}: {:?}", step, e))?;

            // Map back to active sequences
            let mut tok_idx = 0;
            for i in 0..batch_size {
                if !active[i] { continue; }
                let t = new_tokens[tok_idx];
                all_generated[i].push(t);
                last_tokens[i] = t;
                if eos_ids.contains(&t) {
                    active[i] = false;
                }
                tok_idx += 1;
            }
        }

        let elapsed = gen_start.elapsed().as_secs_f32();
        eprintln!("[demo] batch decode done in {:.2}s", elapsed);
        for i in 0..batch_size {
            let txt = tokenizer.decode(
                &all_generated[i].iter().map(|&x| x as u32).collect::<Vec<_>>(),
                false,
            ).unwrap_or_default();
            println!("[batch-output] seq={} tokens={:?}", i, all_generated[i]);
            println!("[batch-text]   seq={} text={:?}", i, &txt[..txt.len().min(200)]);
            println!();
        }
        return Ok(());
    }

    // 5. Build prompt ids — either via tokenizer or directly from a JSON file.
    let tok_path = args.model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("load tokenizer {}: {:?}", tok_path.display(), e))?;

    let prompt_ids: Vec<i32> = if let Some(ids_path) = args.prompt_ids_file.as_ref() {
        let bytes = std::fs::read(ids_path)
            .with_context(|| format!("read {}", ids_path.display()))?;
        let ids: Vec<i64> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse {} as JSON array", ids_path.display()))?;
        eprintln!("[demo] loaded {} prompt ids from {}", ids.len(), ids_path.display());
        ids.into_iter().map(|v| v as i32).collect()
    } else {
        let rendered: String = if args.chat {
            let s = render_llama3_chat(&args.system, &args.prompt);
            eprintln!("[demo] chat template enabled (system={:?})", args.system);
            s
        } else {
            args.prompt.clone()
        };
        let encoding = tokenizer.encode(rendered.as_str(), true)
            .map_err(|e| anyhow!("tokenize: {:?}", e))?;
        encoding.get_ids().iter().map(|&id| id as i32).collect()
    };
    eprintln!("[demo] prompt tokens = {} ids", prompt_ids.len());
    eprintln!("[demo] prompt_ids   = {:?}", prompt_ids);
    let id_to_str: Vec<String> = prompt_ids.iter().map(|&i| {
        tokenizer.decode(&[i as u32], false).unwrap_or_default()
    }).collect();
    eprintln!("[demo] prompt tokens (decoded per id):");
    for (i, (id, s)) in prompt_ids.iter().zip(id_to_str.iter()).enumerate() {
        eprintln!("  [{:>3}] {:>6}  {:?}", i, id, s);
    }

    // 6. Generate.
    // Llama 3.2 generation_config: eos = {128001, 128008, 128009}.
    let eos_ids: &[i32] = &[128001, 128008, 128009];
    let gen_start = std::time::Instant::now();
    let new_tokens = runner.generate_with_graph(&prompt_ids, args.max_new_tokens, eos_ids)
        .map_err(|e| anyhow!("generate: {:?}", e))?;
    let elapsed = gen_start.elapsed().as_secs_f32();
    eprintln!(
        "[demo] generated {} tokens in {:.2}s ({:.1} tok/s)",
        new_tokens.len(), elapsed, new_tokens.len() as f32 / elapsed,
    );
    if std::env::var("RUSTINFER_PROFILE_GPU").is_ok() && runner.prof_step_count > 0 {
        let n = runner.prof_step_count as f64;
        let wall_us  = runner.prof_step_wall_ns as f64 / 1000.0 / n;
        let gpu_us   = runner.prof_graph_gpu_ns as f64 / 1000.0 / n;
        let host_us  = wall_us - gpu_us;
        eprintln!(
            "[profile] decode steps={}  wall={:.1}µs/tok  gpu_graph={:.1}µs/tok  host_overhead={:.1}µs/tok ({:.1}%)",
            runner.prof_step_count, wall_us, gpu_us, host_us, 100.0 * host_us / wall_us,
        );
        eprintln!(
            "[profile] tok/s ceiling if host_overhead → 0: {:.1}",
            1.0e6 / gpu_us,
        );
    }

    // 7. Decode and print.
    let mut all_ids: Vec<u32> = prompt_ids.iter().map(|&i| i as u32).collect();
    all_ids.extend(new_tokens.iter().map(|&i| i as u32));
    let text = tokenizer.decode(&all_ids, true)
        .map_err(|e| anyhow!("decode: {:?}", e))?;
    println!("\n=== generated text ===\n{}\n======================", text);

    Ok(())
}
