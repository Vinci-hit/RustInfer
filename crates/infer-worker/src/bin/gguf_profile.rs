//! Warm, teacher-forced profiling harness; no changes to inference arithmetic.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use infer_backend_cuda::{Cuda, CudaMemoryPlan, ffi};
use infer_core::ports::MemoryPort;
use infer_worker::{
    infrastructure::io::gguf::GgufReader,
    models::qwen3_5::gguf::{LoadOptions, Qwen35GgufLoader, probe::GgufProbe},
};
use serde::Deserialize;
use std::{
    ffi::{CString, c_char, c_void},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    nvtx_library: PathBuf,
}

#[derive(Deserialize)]
struct Workload {
    model: PathBuf,
    context: usize,
    input_ids: Vec<i32>,
    continuation_ids: Vec<i32>,
    warmups: usize,
    repeats: usize,
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

struct Nvtx {
    handle: *mut c_void,
    push: unsafe extern "C" fn(*const c_char),
    pop: unsafe extern "C" fn(),
}
impl Nvtx {
    fn load(path: &std::path::Path) -> Result<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())?;
        // The user supplies the profiling bridge built from gguf_profile_nvtx.cpp.
        unsafe {
            let handle = dlopen(path.as_ptr(), 2); // RTLD_NOW
            ensure!(!handle.is_null(), "cannot load NVTX bridge");
            let push = dlsym(handle, c"ri_nvtx_push".as_ptr());
            let pop = dlsym(handle, c"ri_nvtx_pop".as_ptr());
            if push.is_null() || pop.is_null() {
                dlclose(handle);
                anyhow::bail!("NVTX bridge is missing its exported functions");
            }
            Ok(Self {
                handle,
                push: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_char)>(push),
                pop: std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(pop),
            })
        }
    }
    fn range(&self, name: &str) -> Range<'_> {
        let name = CString::new(name).expect("static profiling label");
        unsafe { (self.push)(name.as_ptr()) };
        Range(self)
    }
}
impl Drop for Nvtx {
    fn drop(&mut self) {
        unsafe {
            dlclose(self.handle);
        }
    }
}
struct Range<'a>(&'a Nvtx);
impl Drop for Range<'_> {
    fn drop(&mut self) {
        unsafe { (self.0.pop)() }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let work: Workload = serde_json::from_reader(std::fs::File::open(&args.workload)?)?;
    ensure!(
        !work.input_ids.is_empty() && !work.continuation_ids.is_empty(),
        "empty workload"
    );
    ensure!(
        work.warmups > 0 && work.repeats > 0,
        "warmups and repeats must be positive"
    );
    ensure!(
        work.input_ids.len() + work.continuation_ids.len() - 1 <= work.context,
        "context overflow"
    );
    let nvtx = Nvtx::load(&args.nvtx_library)?;
    let start = Instant::now();
    let load = nvtx.range("load");
    let reader = GgufReader::open(&work.model)?;
    let loader = Qwen35GgufLoader::new(
        &reader,
        LoadOptions {
            context_length: work.context,
        },
    )?;
    let device = Cuda::with_memory_plan(
        0,
        CudaMemoryPlan {
            kernel_workspace_bytes: 1 << 20,
            graph_arena_bytes: 1 << 20,
            pool_retain_bytes: 4 << 20,
        },
    )?;
    let mut probe =
        GgufProbe::<half::bf16, Cuda>::load(&loader, device.scope(), work.input_ids.len())?;
    drop(loader);
    drop(reader);
    device.synchronize()?;
    drop(load);
    let load_seconds = start.elapsed().as_secs_f64();
    let mut records = Vec::new();
    for run in 0..work.warmups + work.repeats {
        let measured = run >= work.warmups;
        if run == work.warmups {
            device.synchronize()?;
            ensure!(
                unsafe { ffi::cudaProfilerStart() } == 0,
                "cudaProfilerStart failed"
            );
        }
        let label = if measured {
            format!("iteration/{}", run - work.warmups)
        } else {
            format!("warmup/{run}")
        };
        let _iteration = nvtx.range(&label);
        let reset = nvtx.range("reset");
        probe.reset()?;
        drop(reset);
        for step in 0..work.continuation_ids.len() {
            let ids = if step == 0 {
                work.input_ids.as_slice()
            } else {
                &work.continuation_ids[step - 1..step]
            };
            let phase = if step == 0 {
                "prefill".to_string()
            } else {
                format!("decode/{step}")
            };
            let range = nvtx.range(&phase);
            let started = Instant::now();
            let output = probe.step(ids, false).context("profile forward")?;
            let wall = started.elapsed().as_secs_f64();
            drop(range);
            let sample = nvtx.range("host_argmax");
            let top = output
                .logits
                .iter()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                .unwrap()
                .0;
            drop(sample);
            if measured {
                records.push(
                    serde_json::json!({"iteration": run - work.warmups, "phase": phase,
                    "position": output.position, "wall_seconds": wall, "top_id": top,
                    "expected_id": work.continuation_ids[step]}),
                );
            }
        }
    }
    device.synchronize()?;
    ensure!(
        unsafe { ffi::cudaProfilerStop() } == 0,
        "cudaProfilerStop failed"
    );
    serde_json::to_writer_pretty(
        std::fs::File::create(args.output)?,
        &serde_json::json!({"engine":"rustinfer", "load_seconds":load_seconds, "records":records}),
    )?;
    Ok(())
}
