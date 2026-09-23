//! Offline GGUF inspection, text tokenization and single-sequence inference.
use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use infer_worker::{
    domain::{
        dtype::Dtype,
        exec::{ExecScope, HostScope},
        model::DecoderModel,
        ports::backend::LlmBackend,
    },
    infrastructure::{
        cpu::Cpu,
        io::gguf::{GgufArray, GgufReader, GgufValue},
    },
    models::qwen3_5::gguf::{
        LoadOptions, Qwen35GgufLoader,
        probe::{GgufProbe, ProbeOutput},
        text::{ChatMessage, ChatRole, GgufText},
    },
};
use std::{path::PathBuf, time::Instant};

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Inspect,
    Cpu,
    Cuda,
}
#[derive(Clone, Copy, ValueEnum)]
enum DtypeArg {
    F32,
    F16,
    Bf16,
}
#[derive(Parser)]
#[command(about = "Inspect or run a qwen35 GGUF decoder with text or token-ID input")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "inspect")]
    backend: Backend,
    #[arg(long, value_enum, default_value = "bf16")]
    dtype: DtypeArg,
    #[arg(long, default_value_t = 4096)]
    context: usize,
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Raw IDs; no text tokenization or automatic chat template is applied.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    token_ids: Vec<i32>,
    /// User message, rendered with the chat template stored in the GGUF.
    #[arg(long, conflicts_with_all = ["token_ids", "raw_prompt"])]
    prompt: Option<String>,
    /// Completion text, without a chat template.
    #[arg(long, conflicts_with = "token_ids")]
    raw_prompt: Option<String>,
    #[arg(long, requires = "prompt", conflicts_with_all = ["raw_prompt", "token_ids"])]
    system: Option<String>,
    /// Ask the checkpoint's template to disable thinking.
    #[arg(long, requires = "prompt", conflicts_with_all = ["reasoning_effort", "raw_prompt", "token_ids"])]
    no_thinking: bool,
    #[arg(long, requires = "prompt", conflicts_with_all = ["raw_prompt", "token_ids"], value_parser = ["low", "medium", "high", "xhigh"])]
    reasoning_effort: Option<String>,
    /// Number of greedy predictions, including the first prefill prediction.
    #[arg(long, default_value_t = 128)]
    steps: usize,
    /// Check every layer for non-finite activations (adds GPU readbacks).
    #[arg(long)]
    trace: bool,
    /// Compare full-prompt prefill with tokenwise execution using fresh state.
    #[arg(long)]
    verify_chunks: bool,
    /// Write generation diagnostics, or tokenization results in inspect mode.
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_flags_are_exclusive_and_chat_flags_require_chat() {
        for flags in [
            vec!["--prompt", "hello", "--raw-prompt", "hello"],
            vec!["--prompt", "hello", "--token-ids", "1"],
            vec!["--raw-prompt", "hello", "--token-ids", "1"],
            vec!["--raw-prompt", "hello", "--no-thinking"],
            vec!["--system", "hello"],
            vec!["--reasoning-effort", "low"],
            vec![
                "--prompt",
                "hello",
                "--no-thinking",
                "--reasoning-effort",
                "low",
            ],
        ] {
            assert!(
                Args::try_parse_from(
                    ["gguf", "--model", "test.gguf"]
                        .into_iter()
                        .chain(flags.iter().copied())
                )
                .is_err(),
                "{flags:?}"
            );
        }
        let parsed = Args::try_parse_from([
            "gguf",
            "--model",
            "test.gguf",
            "--prompt",
            "你好",
            "--no-thinking",
            "--system",
            "简洁回答",
        ])
        .unwrap();
        assert_eq!(parsed.prompt.as_deref(), Some("你好"));
        assert!(parsed.no_thinking);
        assert!(parsed.token_ids.is_empty());
    }
}
struct TextInput {
    tokenizer: GgufText,
    rendered: String,
}

fn prepare_text(args: &mut Args, reader: &GgufReader) -> Result<Option<TextInput>> {
    if args.prompt.is_none() && args.raw_prompt.is_none() {
        return Ok(None);
    }
    let tokenizer = GgufText::from_gguf(reader)?;
    let rendered = if let Some(prompt) = &args.prompt {
        let mut messages = Vec::new();
        if let Some(system) = &args.system {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: system.clone(),
                reasoning_content: None,
            });
        }
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: prompt.clone(),
            reasoning_content: None,
        });
        tokenizer.render_chat(
            &messages,
            !args.no_thinking,
            args.reasoning_effort.as_deref(),
        )?
    } else {
        args.raw_prompt.as_ref().unwrap().clone()
    };
    args.token_ids = tokenizer.encode(&rendered, args.prompt.is_none())?;
    ensure!(!args.token_ids.is_empty(), "text input produced no tokens");
    Ok(Some(TextInput {
        tokenizer,
        rendered,
    }))
}

fn loaded<T: Dtype, D: LlmBackend + infer_worker::domain::ports::OpBackend>(
    l: &Qwen35GgufLoader<'_>,
    scope: D::Scope,
    args: &Args,
    reader: &GgufReader,
    text: Option<&TextInput>,
) -> Result<()> {
    if !args.token_ids.is_empty() {
        return forward::<T, D>(l, scope, args, reader, text);
    }
    let device = scope.device();
    let start = Instant::now();
    let m = l.load::<T, D>(device)?;
    ensure!(
        m.dims().num_layers == l.config().dims.num_layers,
        "layer count mismatch"
    );
    println!(
        "Loaded {} main layers: {} full attention + {} GDN; elapsed {:.2}s",
        m.dims().num_layers,
        m.cache_layout().num_full_layers(),
        m.cache_layout().linear_dims().len(),
        start.elapsed().as_secs_f64()
    );
    println!("Weights own device storage. No KV/state/vision/tokenizer has been initialized.");
    Ok(())
}
fn main() -> Result<()> {
    let mut a = Args::parse();
    let r = GgufReader::open(&a.model)?;
    let text = prepare_text(&mut a, &r)?;
    let has_input = !a.token_ids.is_empty();
    let inspect = matches!(a.backend, Backend::Inspect);
    ensure!(
        has_input || (!a.trace && !a.verify_chunks && a.dump.is_none()),
        "input is required for diagnostic options"
    );
    ensure!(
        !inspect || (!a.trace && !a.verify_chunks),
        "--trace/--verify-chunks require --backend cpu or cuda"
    );
    ensure!(
        !has_input
            || inspect
            || (a.steps > 0
                && a.token_ids
                    .len()
                    .checked_add(a.steps - 1)
                    .is_some_and(|n| n <= a.context)),
        "prompt and steps exceed context, or steps is zero"
    );
    if has_input && matches!(a.backend, Backend::Cuda) {
        ensure!(
            !matches!(a.dtype, DtypeArg::F32),
            "CUDA paged attention prefill requires bf16/f16"
        );
    }
    ensure!(
        !a.verify_chunks || a.token_ids.len() < a.context,
        "chunk verification needs one additional context position"
    );
    let l = Qwen35GgufLoader::new(
        &r,
        LoadOptions {
            context_length: a.context,
        },
    )?;
    let c = l.config();
    ensure!(
        a.token_ids
            .iter()
            .all(|&id| id >= 0 && (id as usize) < c.dims.vocab_size),
        "token ID outside vocabulary"
    );
    if let Some(text) = &text {
        ensure!(
            text.tokenizer.vocab_size() == c.dims.vocab_size,
            "tokenizer/model vocabulary mismatch"
        );
        println!("Rendered prompt: {:?}", text.rendered);
    }
    if has_input {
        println!("Input token IDs ({}): {:?}", a.token_ids.len(), a.token_ids);
    }
    let report = l.report();
    let bytes = match a.dtype {
        DtypeArg::F32 => l.device_weight_bytes::<f32>()?,
        DtypeArg::F16 => l.device_weight_bytes::<half::f16>()?,
        DtypeArg::Bf16 => l.device_weight_bytes::<half::bf16>()?,
    };
    println!(
        "qwen35: hidden={} vocab={} layers={} context={}/{}",
        c.dims.dim,
        c.dims.vocab_size,
        c.dims.num_layers,
        c.context_length,
        c.trained_context_length
    );
    println!(
        "Main tensors={} source_bytes={} estimated_device_weight_and_rope_bytes={} ({:.3} GiB)",
        report.loaded_tensors,
        report.source_weight_bytes,
        bytes,
        bytes as f64 / (1u64 << 30) as f64
    );
    println!("Formats: {:?}", report.format_counts);
    println!(
        "Skipped MTP: {} layers, {} tensors, {} bytes; tied output={}",
        c.mtp_layers,
        report.skipped_mtp_tensors.len(),
        report.skipped_mtp_bytes,
        report.tied_output
    );
    macro_rules! load {
        ($backend:ty, $dev:expr) => {
            match a.dtype {
                DtypeArg::F32 => loaded::<f32, $backend>(&l, $dev, &a, &r, text.as_ref()),
                DtypeArg::F16 => loaded::<half::f16, $backend>(&l, $dev, &a, &r, text.as_ref()),
                DtypeArg::Bf16 => loaded::<half::bf16, $backend>(&l, $dev, &a, &r, text.as_ref()),
            }
        };
    }
    match a.backend {
        Backend::Inspect => {
            if let Some(path) = &a.dump {
                let decoded = text
                    .as_ref()
                    .map(|t| t.tokenizer.decode(&a.token_ids, false))
                    .transpose()?;
                let report = serde_json::json!({
                    "input_ids": a.token_ids, "rendered_prompt": text.as_ref().map(|t| &t.rendered), "decoded_input": decoded
                });
                serde_json::to_writer(std::fs::File::create(path)?, &report)?;
                println!("Saved tokenization report to {}", path.display());
            }
            Ok(())
        }
        Backend::Cpu => load!(Cpu, HostScope::new(Cpu)),
        Backend::Cuda => {
            #[cfg(feature = "cute-dsl")]
            {
                let dev = infer_backend_cuda::Cuda::with_memory_plan(
                    a.device,
                    infer_backend_cuda::CudaMemoryPlan {
                        kernel_workspace_bytes: 1 << 20,
                        graph_arena_bytes: 1 << 20,
                        pool_retain_bytes: 4 << 20,
                    },
                )?;
                ensure!(
                    dev.config.cute_dsl_available(),
                    "CuTe DSL target does not match device"
                );
                load!(infer_backend_cuda::Cuda, dev.scope())
            }
            #[cfg(not(feature = "cute-dsl"))]
            anyhow::bail!("CUDA GGUF loading requires --features cute-dsl")
        }
    }
}

fn forward<T: Dtype, D: LlmBackend + infer_worker::domain::ports::OpBackend>(
    loader: &Qwen35GgufLoader<'_>,
    scope: D::Scope,
    args: &Args,
    reader: &GgufReader,
    text: Option<&TextInput>,
) -> Result<()> {
    ensure!(
        args.token_ids
            .iter()
            .all(|&id| id >= 0 && (id as usize) < loader.config().dims.vocab_size),
        "token ID outside vocabulary"
    );
    let start = Instant::now();
    let mut probe = GgufProbe::<T, D>::load(loader, scope, args.token_ids.len())?;
    println!(
        "Model and single-sequence caches ready in {:.2}s",
        start.elapsed().as_secs_f64()
    );
    let tokens = match reader.metadata().get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(GgufArray::String(v))) => Some(v),
        _ => None,
    };
    let describe = |out: &ProbeOutput| {
        let top: Vec<_> = out
            .top_k(5)
            .into_iter()
            .map(|(id, logit)| {
                serde_json::json!({
                    "id": id, "logit": logit,
                    "piece": tokens.and_then(|t| t.get(id))
                })
            })
            .collect();
        println!(
            "position={} vocab={} elapsed={:.3}s top={}",
            out.position,
            out.logits.len(),
            out.elapsed_seconds,
            serde_json::json!(top)
        );
        args.dump.as_ref().map(|_| {
            let layers: Vec<_> = out
                .layers
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "layer": s.layer, "max_abs": s.max_abs,
                        "last_residual": s.last_residual
                    })
                })
                .collect();
            serde_json::json!({
                "position": out.position, "elapsed_seconds": out.elapsed_seconds,
                "top": top, "layers": layers, "logits": out.logits
            })
        })
    };
    let first = probe.step(&args.token_ids, args.trace)?;
    let mut records: Vec<_> = describe(&first).into_iter().collect();
    let mut verification = serde_json::Value::Null;
    if args.verify_chunks {
        let next_id = first.top_k(1)[0].0 as i32;
        let full_next = probe.step(&[next_id], args.trace)?;
        probe.reset()?;
        let mut tokenwise = None;
        for id in &args.token_ids {
            tokenwise = Some(probe.step(&[*id], args.trace)?);
        }
        let tokenwise = tokenwise.unwrap();
        let chunk_next = probe.step(&[next_id], args.trace)?;
        verification = serde_json::json!({
            "prefill": compare(&first, &tokenwise)?,
            "continuation": compare(&full_next, &chunk_next)?
        });
        for phase in ["prefill", "continuation"] {
            let v = &verification[phase];
            println!(
                "Full vs tokenwise {phase}: relative_l2={} max_abs={} same_top1={}",
                v["relative_l2"], v["max_abs"], v["same_top1"]
            );
        }
        probe.reset()?;
        probe.step(&args.token_ids, false)?;
    }
    let mut next = first.top_k(1)[0].0 as i32;
    let mut generated = vec![next];
    let eos = reader
        .metadata()
        .get("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::as_u32);
    let is_end = |id: i32| text.map_or(Some(id as u32) == eos, |t| t.tokenizer.is_end_token(id));
    for _ in 1..args.steps {
        if is_end(next) {
            break;
        }
        let output = probe.step(&[next], args.trace)?;
        records.extend(describe(&output));
        next = output.top_k(1)[0].0 as i32;
        generated.push(next);
    }
    println!("Greedy token IDs: {generated:?}");
    let finish_reason = if is_end(next) { "eos" } else { "length" };
    let generated_text = text
        .map(|t| t.tokenizer.decode(&generated, true))
        .transpose()?;
    if let Some(decoded) = &generated_text {
        println!("Generated text ({finish_reason}):\n{decoded}");
    }
    if let Some(path) = &args.dump {
        let report = serde_json::json!({
            "input_ids": args.token_ids, "generated_ids": generated,
            "verification": &verification, "steps": records,
            "rendered_prompt": text.map(|t| &t.rendered),
            "generated_text": generated_text, "finish_reason": finish_reason
        });
        serde_json::to_writer(std::fs::File::create(path)?, &report)?;
        println!("Saved diagnostic report to {}", path.display());
    }
    ensure!(
        verification.is_null()
            || (verification["prefill"]["relative_l2"].as_f64().unwrap() < 0.02
                && verification["continuation"]["relative_l2"]
                    .as_f64()
                    .unwrap()
                    < 0.02),
        "prefill/decode comparison exceeded 2% relative L2; see diagnostic report"
    );
    Ok(())
}

fn compare(a: &ProbeOutput, b: &ProbeOutput) -> Result<serde_json::Value> {
    ensure!(
        a.logits.len() == b.logits.len()
            && a.position == b.position
            && a.layers.len() == b.layers.len(),
        "comparison geometry differs"
    );
    let error = a
        .logits
        .iter()
        .zip(&b.logits)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let energy = a.logits.iter().map(|&a| f64::from(a).powi(2)).sum::<f64>();
    let max_abs = a
        .logits
        .iter()
        .zip(&b.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let relative_l2 = (error / energy.max(1e-30)).sqrt();
    let mut layer_errors = Vec::with_capacity(a.layers.len());
    for (a, b) in a.layers.iter().zip(&b.layers) {
        ensure!(
            a.layer == b.layer && a.last_residual.len() == b.last_residual.len(),
            "layer comparison geometry differs"
        );
        let err = a
            .last_residual
            .iter()
            .zip(&b.last_residual)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>();
        let norm = a
            .last_residual
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>();
        let max = a
            .last_residual
            .iter()
            .zip(&b.last_residual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        layer_errors.push(serde_json::json!({
            "layer": a.layer, "relative_l2": (err / norm.max(1e-30)).sqrt(), "max_abs": max
        }));
    }
    Ok(
        serde_json::json!({"max_abs":max_abs,"relative_l2":relative_l2,"same_top1":a.top_k(1)[0].0==b.top_k(1)[0].0,"layer_errors":layer_errors}),
    )
}
