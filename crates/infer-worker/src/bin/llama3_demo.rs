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

use infer_worker::app::model_runner::ModelRunner;
use infer_worker::infra::cuda::Cuda;
use infer_worker::infra::io::SafetensorsReader;
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
    // Paged KV pool sizing: pick block_size=16 (matches scheduler default),
    // allocate enough blocks to cover max_batch_seqs * max_seq_len, plus
    // one extra physical block reserved for CUDA-Graph scratch.
    let block_size: usize = 16;
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
        infer_worker::infra::cuda::kernels::attention_paged::flash_decode_workspace_capacity_f32(
            cap_batch.max(1), 128, 256,
        );
    let mut runner = ModelRunner::new(
        model, cuda, pool_blocks, block_size, max_blocks_per_seq, args.max_seq_len,
        cap_num_tokens, cap_batch, flash_decode_capacity_f32,
        vec![1, 2, 4, 8, 16],
    ).map_err(|e| anyhow!("ModelRunner::new: {:?}", e))?;

    // Prime CUDA Graphs (decode-only).
    if let Err(e) = runner.prime_graphs_cuda() {
        eprintln!("[demo] CUDA Graph priming FAILED, continuing eager: {:?}", e);
    } else {
        eprintln!("[demo] CUDA Graphs primed for {:?}", runner.capture_sizes);
    }

    if let Some(prompts) = batch_prompts {
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
            steps.push(infer_worker::app::model_runner::SeqStep {
                input_ids: p.clone(),
                positions: (0..p.len() as i32).collect(),
                kv_write_start: 0,
                kv_len_after: p.len() as i32,
                block_table,
            });
        }
        let gen_start = std::time::Instant::now();
        let first_tokens = runner.step_batch(&steps)
            .map_err(|e| anyhow!("step_batch: {:?}", e))?;
        eprintln!("[demo] batch prefill: {:.2}s, first tokens: {:?}",
            gen_start.elapsed().as_secs_f32(), first_tokens);
        for (i, &t) in first_tokens.iter().enumerate() {
            let txt = tokenizer.decode(&[t as u32], false).unwrap_or_default();
            println!("=== seq {} first token ===\nid={}  text={:?}\n", i, t, txt);
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

    // 7. Decode and print.
    let mut all_ids: Vec<u32> = prompt_ids.iter().map(|&i| i as u32).collect();
    all_ids.extend(new_tokens.iter().map(|&i| i as u32));
    let text = tokenizer.decode(&all_ids, true)
        .map_err(|e| anyhow!("decode: {:?}", e))?;
    println!("\n=== generated text ===\n{}\n======================", text);

    Ok(())
}
