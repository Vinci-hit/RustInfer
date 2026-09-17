//! Offline architecture parity runner; deliberately independent of serving.
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use infer_worker::{
    domain::{
        dtype::Dtype,
        exec::{ExecScope, HostScope},
        ports::backend::LlmBackend,
    },
    infrastructure::{cpu::Cpu, io::SafetensorsReader},
    models::{
        deepseek_v4::{TinyConfig, TinyModel},
        loader::WeightLoader,
    },
};
use serde::Deserialize;

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(
    about = "Compare a tiny random V4 against saved Transformers outputs (host-assisted, single request)"
)]
struct Args {
    /// Directory created by scripts/deepseek_v4_tiny.py.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,
    /// Save Rust layer outputs and logits as F32 safetensors for diagnosis.
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    dtype: String,
    inputs: Vec<usize>,
    cases: BTreeMap<String, Vec<usize>>,
}

fn compare(actual: &[f32], expected: &[f32], dtype: &str) -> Result<serde_json::Value> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "reference shape mismatch"
    );
    ensure!(
        actual.iter().chain(expected).all(|x| x.is_finite()),
        "non-finite diagnostic"
    );
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    let energy: f64 = expected.iter().map(|&b| (b as f64).powi(2)).sum();
    let relative_l2 = (error / energy.max(1e-30)).sqrt();
    let worst = actual
        .iter()
        .zip(expected)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
        .unwrap()
        .0;
    let (abs_limit, rel_limit) = if dtype == "float32" {
        (2e-5, 2e-4)
    } else {
        (0.025, 0.03)
    };
    Ok(serde_json::json!({
        "passed":max_abs <= abs_limit && relative_l2 <= rel_limit,
        "max_abs":max_abs,"relative_l2":relative_l2,
        "abs_limit":abs_limit,"relative_l2_limit":rel_limit,
        "worst_element":worst,"worst_actual":actual[worst],"worst_reference":expected[worst],
    }))
}

fn run<T: Dtype, D: LlmBackend>(
    args: &Args,
    cfg: TinyConfig,
    manifest: &Manifest,
    scope: D::Scope,
) -> Result<()> {
    let start = Instant::now();
    let reader = SafetensorsReader::open(args.model.join("model.safetensors"))
        .map_err(anyhow::Error::msg)?;
    let reference = SafetensorsReader::open(args.model.join("reference.safetensors"))
        .map_err(anyhow::Error::msg)?;
    let _active = scope.enter();
    let model = TinyModel::<T, D>::load(&WeightLoader::new(&reader), cfg, scope.device())?;
    let mut reports = BTreeMap::new();
    let mut passed = true;
    for (case, chunks) in &manifest.cases {
        ensure!(
            !case.is_empty()
                && case
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "case names must contain only letters, digits, underscores or hyphens"
        );
        ensure!(
            !chunks.is_empty()
                && chunks.iter().all(|&n| n > 0)
                && chunks.iter().try_fold(0usize, |a, &b| a.checked_add(b))
                    == Some(manifest.inputs.len()),
            "case {case}: chunks must cover every input exactly once"
        );
        let mut cache = model.new_cache();
        let mut parts = vec![Vec::new(); model.config().num_hidden_layers + 1];
        let mut cursor = 0;
        for &n in chunks {
            let result = model.forward(&manifest.inputs[cursor..cursor + n], &mut cache, &scope)?;
            parts[0].extend(result.logits);
            for (dst, src) in parts[1..].iter_mut().zip(result.layers) {
                dst.extend(src);
            }
            cursor += n;
        }
        let mut metrics = BTreeMap::new();
        if let Some(directory) = &args.dump {
            std::fs::create_dir_all(directory)?;
            let file = directory.join(format!("{case}.safetensors"));
            ensure!(!file.exists(), "dump already exists: {}", file.display());
            let storage: Vec<_> = parts
                .iter()
                .enumerate()
                .map(|(i, data)| {
                    let name = if i == 0 {
                        "logits".to_owned()
                    } else {
                        format!("layer{}", i - 1)
                    };
                    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    (name, vec![cursor, data.len() / cursor], bytes)
                })
                .collect();
            let views = storage
                .iter()
                .map(|(name, shape, data)| {
                    Ok((
                        name.as_str(),
                        safetensors::tensor::TensorView::new(
                            safetensors::Dtype::F32,
                            shape.clone(),
                            data,
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
            safetensors::tensor::serialize_to_file(views, None, &file)?;
        }
        for i in (1..parts.len()).chain(std::iter::once(0)) {
            let actual = &parts[i];
            let part = if i == 0 {
                "logits".to_owned()
            } else {
                format!("layer{}", i - 1)
            };
            let name = format!("{case}.{part}");
            let view = reference.read_view(&name).map_err(anyhow::Error::msg)?;
            let width = if i == 0 {
                model.config().vocab_size
            } else {
                model.config().hc_mult * model.config().hidden_size
            };
            ensure!(
                view.dtype() == safetensors::Dtype::F32 && view.shape() == [cursor, width],
                "{name}: invalid reference tensor"
            );
            let expected: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            metrics.insert(
                part,
                compare(actual, &expected, &manifest.dtype).with_context(|| name)?,
            );
        }
        let case_passed = metrics.values().all(|v| v["passed"] == true);
        passed &= case_passed;
        eprintln!(
            "{case}: {cursor} tokens, {} chunks, parity {}",
            chunks.len(),
            if case_passed { "passed" } else { "FAILED" }
        );
        reports.insert(
            case,
            serde_json::json!({"metrics":metrics,"compressed_entries":cache.compressed_entries()}),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed":passed,"dtype":manifest.dtype,
            "backend":match args.backend { Backend::Cpu => "cpu", Backend::Cuda => "cuda_gemm_host_reference" },
            "elapsed_seconds":start.elapsed().as_secs_f64(),"cases":reports,
        }))?
    );
    ensure!(passed, "V4 parity failed; see the JSON metrics");
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg: TinyConfig = serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?;
    cfg.validate()?;
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(args.model.join("manifest.json"))?)?;
    ensure!(
        manifest.format == "rustinfer-v4-tiny-v1"
            && manifest.dtype == cfg.dtype
            && !manifest.cases.is_empty(),
        "incompatible or empty manifest"
    );
    match args.backend {
        Backend::Cpu => match cfg.dtype.as_str() {
            "float32" => run::<f32, Cpu>(&args, cfg, &manifest, HostScope::new(Cpu)),
            "bfloat16" => run::<half::bf16, Cpu>(&args, cfg, &manifest, HostScope::new(Cpu)),
            _ => bail!("unsupported dtype"),
        },
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use infer_worker::infrastructure::cuda::{Cuda, CudaMemoryPlan, CudaScope};
                ensure!(
                    cfg.dtype != "float32"
                        || std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0"),
                    "strict FP32 CUDA comparison requires NVIDIA_TF32_OVERRIDE=0 before starting this process"
                );
                let device = Cuda::with_memory_plan(
                    0,
                    CudaMemoryPlan {
                        kernel_workspace_bytes: 8 * 1024 * 1024,
                        graph_arena_bytes: 0,
                        pool_retain_bytes: 8 * 1024 * 1024,
                    },
                )?;
                let scope = CudaScope::new(device);
                match cfg.dtype.as_str() {
                    "float32" => run::<f32, Cuda>(&args, cfg, &manifest, scope),
                    "bfloat16" => run::<half::bf16, Cuda>(&args, cfg, &manifest, scope),
                    _ => bail!("unsupported dtype"),
                }
            }
            #[cfg(not(feature = "cuda"))]
            bail!("CUDA backend requires building with --features cuda")
        }
    }
}
