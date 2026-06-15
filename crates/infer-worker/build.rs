use std::env;
use std::path::{Path, PathBuf};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if std::env::var("SKIP_BUILD_KERNELS").is_ok() {
            return;
        }
        let kernel_paths = find_files("src/infrastructure/cuda/kernels", "cu");

        if kernel_paths.is_empty() {
            println!(
                "cargo:warning=No CUDA kernel files (.cu) found in src/infrastructure/cuda/kernels/"
            );
            // return; // 如果你希望在这种情况下停止构建
        }
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=cublas");
        // 对应 cublasLt.h
        println!("cargo:rustc-link-lib=cublasLt");
        // cuDNN (Conv2d 等)
        println!("cargo:rustc-link-lib=cudnn");
        println!("cargo:rustc-link-lib=nvrtc");
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = PathBuf::from(manifest_dir);
        let rustinfer_root = root
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .unwrap_or_else(|| root.clone());
        let cutlass_include = root.join("src/infrastructure/cuda/kernels/third_party");
        let cudnn_frontend_include = find_cudnn_frontend_include(&rustinfer_root);
        if !cutlass_include.exists() {
            panic!(
                "Cutlass include directory not found at: {:?}",
                cutlass_include
            );
        }
        eprintln!(
            "RustInfer build: using cuDNN frontend headers from {}",
            cudnn_frontend_include.display()
        );
        if !cudnn_frontend_include.exists() {
            panic!(
                "cuDNN frontend include directory not found at: {:?}",
                cudnn_frontend_include
            );
        }
        // 自动检测 GPU 架构，无需手动修改。
        let cuda_arch = detect_cuda_arch();
        println!("cargo:rustc-env=RUSTINFER_CUDA_ARCH={}", cuda_arch);
        eprintln!("RustInfer build: detected CUDA arch {}", cuda_arch);

        let mut build = cc::Build::new();
        build
            .cuda(true)
            .opt_level(3)
            .debug(false)
            .flag("-O3")
            .flag("-w")
            .include(&cutlass_include)
            .include("/usr/include/x86_64-linux-gnu")
            .include(&cudnn_frontend_include)
            .flag("-std=c++17")
            .flag(format!("-arch={}", cuda_arch));

        for path in &kernel_paths {
            build.file(path);
            println!("cargo:rerun-if-changed={}", path.display());
        }
        for path in find_files("src/infrastructure/cuda/kernels", "cuh") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        build.compile("infer_kernels");
        println!("cargo:rustc-link-lib=static=infer_kernels");
        println!("cargo:rustc-link-lib=cudart");

        let target = env::var("TARGET").expect("TARGET environment variable not set");

        // 4. 使用 bindgen 生成 Rust FFI 绑定
        let bindings = bindgen::Builder::default()
            .header("src/infrastructure/cuda/wrapper.h")
            // 告诉 bindgen/libclang CUDA 头文件的位置
            .clang_arg(format!(
                "-I{}/include",
                env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".into())
            ))
            .clang_arg("-I/usr/include/x86_64-linux-gnu")
            // wrapper.h 所在目录（让 #include "kernels/total_head.h" 能找到）
            .clang_arg("-Isrc/infrastructure/cuda")
            // 明确告诉 bindgen 本次编译的目标架构
            .clang_arg(format!("--target={}", target))
            .clang_arg(format!("-I{}", cutlass_include.to_string_lossy()))
            // ==================== 关键的新增代码在这里 ====================
            // 强制 libclang 使用 C++ 模式解析头文件
            .clang_arg("-x")
            .clang_arg("c++")
            // =======================================================
            .allowlist_function("cudaMalloc")
            .allowlist_function("cudaFree")
            .allowlist_function("cudaMemcpy")
            .allowlist_function("cudaMemcpyAsync")
            .allowlist_function("cudaMemset")
            .allowlist_function("cudaMemsetAsync")
            .allowlist_function("cudaMemGetInfo")
            .allowlist_function("cudaProfilerStart")
            .allowlist_function("cudaProfilerStop")
            .allowlist_function("cudaGetErrorString")
            .allowlist_function("cudaGetErrorName")
            .allowlist_function("cudaGetDevice")
            .allowlist_function("cudaSetDevice")
            .allowlist_function("cudaStreamCreate")
            .allowlist_function("cudaStreamCreateWithFlags")
            .allowlist_function("cudaStreamDestroy")
            .allowlist_function("cudaStreamWaitEvent")
            .allowlist_function("cudaDeviceSynchronize")
            .allowlist_function("cudaStreamSynchronize")
            .allowlist_function("cudaEventCreate")
            .allowlist_function("cudaEventCreateWithFlags")
            .allowlist_function("cudaEventRecord")
            .allowlist_function("cudaEventSynchronize")
            .allowlist_function("cudaEventElapsedTime")
            .allowlist_function("cudaEventDestroy")
            .allowlist_type("cudaEvent_t")
            .allowlist_type("cudaError_t")
            .allowlist_type("cudaMemcpyKind")
            .allowlist_type("cudaStream_t")
            .allowlist_type("cublasLtHandle_t")
            .allowlist_type("cublasHandle_t")
            .allowlist_type("cudaGraph_t")
            .allowlist_type("cudaGraphExec_t")
            .allowlist_function("cublasLtCreate")
            .allowlist_function("cublasLtDestroy")
            .allowlist_function("cublasCreate_v2")
            .allowlist_function("cublasDestroy_v2")
            .allowlist_function("cudaStreamBeginCapture")
            .allowlist_function("cudaStreamEndCapture")
            .allowlist_function("cudaGraphInstantiate")
            .allowlist_function("cudaGraphDestroy")
            .allowlist_function("cudaGraphLaunch")
            .allowlist_function("cudaGraphExecDestroy")
            // cuDNN types
            .allowlist_type("cudnnHandle_t")
            .allowlist_type("cudnnStatus_t")
            .allowlist_type("cudnnTensorDescriptor_t")
            .allowlist_type("cudnnFilterDescriptor_t")
            .allowlist_type("cudnnConvolutionDescriptor_t")
            .allowlist_type("cudnnConvolutionFwdAlgoPerf_t")
            .allowlist_type("cudnnDataType_t")
            .allowlist_type("cudnnTensorFormat_t")
            .allowlist_type("cudnnConvolutionMode_t")
            .allowlist_type("cudnnConvolutionFwdAlgo_t")
            .allowlist_type("cudnnMathType_t")
            // cuDNN functions
            .allowlist_function("cudnnCreate")
            .allowlist_function("cudnnDestroy")
            .allowlist_function("cudnnSetStream")
            .allowlist_function("cudnnGetErrorString")
            .allowlist_function("cudnnCreateTensorDescriptor")
            .allowlist_function("cudnnSetTensor4dDescriptor")
            .allowlist_function("cudnnDestroyTensorDescriptor")
            .allowlist_function("cudnnCreateFilterDescriptor")
            .allowlist_function("cudnnSetFilter4dDescriptor")
            .allowlist_function("cudnnDestroyFilterDescriptor")
            .allowlist_function("cudnnCreateConvolutionDescriptor")
            .allowlist_function("cudnnSetConvolution2dDescriptor")
            .allowlist_function("cudnnSetConvolutionMathType")
            .allowlist_function("cudnnDestroyConvolutionDescriptor")
            .allowlist_function("cudnnGetConvolutionForwardWorkspaceSize")
            .allowlist_function("cudnnConvolutionForward")
            .allowlist_function("cudnnAddTensor")
            .allowlist_function("cudnnGetConvolutionForwardAlgorithm_v7")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .rustified_enum("cudaError_t")
            .rustified_enum("cudaMemcpyKind")
            .rustified_enum("cudnnStatus_t")
            .rustified_enum("cudnnDataType_t")
            .rustified_enum("cudnnTensorFormat_t")
            .rustified_enum("cudnnConvolutionMode_t")
            .rustified_enum("cudnnConvolutionFwdAlgo_t")
            .rustified_enum("cudnnMathType_t")
            .generate()
            .expect("Unable to generate bindings");
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}

#[cfg(feature = "cuda")]
fn find_cudnn_frontend_include(repo_root: &Path) -> PathBuf {
    println!("cargo:rerun-if-env-changed=CUDNN_FRONTEND_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUDNN_FRONTEND_ROOT");
    println!("cargo:rerun-if-env-changed=CUDNN_FRONTEND_PATH");

    let mut candidates = Vec::new();
    for var in [
        "CUDNN_FRONTEND_INCLUDE_DIR",
        "CUDNN_FRONTEND_ROOT",
        "CUDNN_FRONTEND_PATH",
    ] {
        if let Ok(value) = env::var(var) {
            let path = PathBuf::from(value);
            candidates.push(path.clone());
            candidates.push(path.join("include"));
        }
    }

    let venv_lib = repo_root.join(".venv/lib");
    if let Ok(entries) = std::fs::read_dir(&venv_lib) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("python") {
                candidates.push(entry.path().join("site-packages/include"));
            }
        }
    }

    // conda environments: CONDA_PREFIX env var, or common conda locations
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(conda_prefix);
        candidates.push(prefix.join("include"));
        // conda's own python site-packages (where pip installs cudnn_frontend)
        if let Ok(entries) = std::fs::read_dir(prefix.join("lib")) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with("python") {
                    candidates.push(entry.path().join("site-packages/include"));
                }
            }
        }
    }
    // also check CONDA_ENVS_DIR / ~/miniconda3/envs/*/include etc.
    if let Ok(home) = env::var("HOME") {
        let home = PathBuf::from(home);
        for conda_root in [
            home.join("miniconda3/envs"),
            home.join("anaconda3/envs"),
            home.join(".conda/envs"),
        ] {
            if let Ok(entries) = std::fs::read_dir(&conda_root) {
                for entry in entries.flatten() {
                    candidates.push(entry.path().join("include"));
                    // also check lib/python*/site-packages/include in each env
                    if let Ok(lib_entries) = std::fs::read_dir(entry.path().join("lib")) {
                        for lib_entry in lib_entries.flatten() {
                            let name = lib_entry.file_name();
                            if name.to_string_lossy().starts_with("python") {
                                candidates.push(lib_entry.path().join("site-packages/include"));
                            }
                        }
                    }
                }
            }
        }
    }

    candidates.extend([
        repo_root.join(".venv/lib/python3/site-packages/include"),
        PathBuf::from("/usr/local/cuda/include"),
        PathBuf::from("/usr/local/include"),
        PathBuf::from("/usr/include"),
    ]);

    for candidate in candidates {
        if candidate.join("cudnn_frontend.h").exists()
            && candidate.join("cudnn_frontend/graph_interface.h").exists()
        {
            return candidate;
        }
    }

    panic!(
        "cuDNN frontend headers not found. Set CUDNN_FRONTEND_INCLUDE_DIR to a directory containing cudnn_frontend.h"
    );
}

/// 自动检测当前 GPU 的 compute capability，返回如 "sm_90" 的字符串
/// 优先读取环境变量 CUDA_ARCH（如 CUDA_ARCH=sm_90a），否则用 nvidia-smi 自动检测
#[cfg(feature = "cuda")]
fn detect_cuda_arch() -> String {
    // 1. 环境变量优先，允许手动覆盖
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return arch;
    }

    // 2. 用 nvidia-smi 查询第一块 GPU 的 compute capability
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output();

    if let Ok(output) = output
        && output.status.success()
    {
        let cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // cap 格式如 "9.0", "8.9", "8.0"
        let sm = cap.replace('.', "");
        return format!("sm_{}", sm);
    }

    // 3. fallback
    println!("cargo:warning=Could not detect GPU arch, falling back to sm_80");
    "sm_80".to_string()
}

/// 辅助函数：递归地查找指定目录中具有特定扩展名的文件
#[cfg(feature = "cuda")]
fn find_files(dir: &str, extension: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let walker = walkdir::WalkDir::new(dir).into_iter();

    // 使用 filter_entry 可以高效跳过整个文件夹，不进入其内部扫描
    let iter = walker.filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        name != "third_party" // 跳过 third_party
    });

    for entry in iter {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == extension) {
            paths.push(path.to_path_buf());
        }
    }
    paths
}
