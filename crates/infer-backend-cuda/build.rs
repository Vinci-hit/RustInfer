use std::env;
use std::path::{Path, PathBuf};

struct CudaArchiveSpec {
    name: &'static str,
    arch_family: u32,
    codegen_arch: &'static str,
    source_dir: &'static str,
    extra_flags: &'static [&'static str],
    generic_define: Option<&'static str>,
    rust_cfg: Option<&'static str>,
}

const FP8_SM90_FLAGS: &[&str] = &[
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
];

const FA3_SM90_FLAGS: &[&str] = &[
    "--use_fast_math",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DFLASHATTENTION_DISABLE_LOCAL",
    "-DFLASHATTENTION_DISABLE_APPENDKV",
    "-DFLASHATTENTION_DISABLE_CLUSTER",
    "-DFLASHATTENTION_DISABLE_SM8x",
];

const CUDA_ARCHIVES: &[CudaArchiveSpec] = &[
    CudaArchiveSpec {
        name: "rustinfer_fp8_sm90",
        arch_family: 90,
        codegen_arch: "sm_90a",
        source_dir: "src/kernels/arch/sm90a/fp8",
        extra_flags: FP8_SM90_FLAGS,
        generic_define: Some("RUSTINFER_HAS_FP8_BLOCK_ACCELERATED"),
        rust_cfg: None,
    },
    CudaArchiveSpec {
        name: "rustinfer_fa3",
        arch_family: 90,
        codegen_arch: "sm_90a",
        source_dir: "src/kernels/third_party/fa3",
        extra_flags: FA3_SM90_FLAGS,
        generic_define: None,
        rust_cfg: Some("rustinfer_fa3"),
    },
];

fn main() {
    {
        // 1. 自动处理 libclang 环境变量 (彻底免去手动 export LIBCLANG_PATH)
        auto_configure_libclang();

        let cuda_path = find_cuda_path();

        // 收集所有可能存在 CUDA 头文件和库文件的路径 (应对 Conda 的特殊目录结构)
        let cuda_includes = get_cuda_include_paths(&cuda_path);
        let cuda_lib_paths = get_cuda_lib_paths(&cuda_path);

        // 尽早设置环境变量，供 cc-rs 查找 nvcc
        unsafe {
            env::set_var("CUDA_PATH", &cuda_path);
            let nvcc_path = cuda_path.join("bin/nvcc");
            env::set_var("NVCC", nvcc_path.to_string_lossy().to_string());
            let new_path = format!(
                "{}:{}",
                cuda_path.join("bin").display(),
                env::var("PATH").unwrap_or_default()
            );
            env::set_var("PATH", new_path);
        }

        eprintln!("RustInfer build: using CUDA from {}", cuda_path.display());

        if std::env::var("SKIP_BUILD_KERNELS").is_ok() {
            return;
        }
        // 配置 Rust 链接搜索路径 (加入动态识别的 Conda lib 路径)
        for lib_path in &cuda_lib_paths {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
        }
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

        // 链接需要的库
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cudnn");
        println!("cargo:rustc-link-lib=nvrtc");

        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = PathBuf::from(manifest_dir);
        let rustinfer_root = root
            .parent()
            .and_then(|p| p.parent())
            .map(PathBuf::from)
            .unwrap_or_else(|| root.clone());

        let cutlass_include = root.join("src/kernels/third_party");
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

        // 自动检测 GPU 架构
        let cuda_arch = detect_cuda_arch();
        println!("cargo:rustc-env=RUSTINFER_CUDA_ARCH={}", cuda_arch);
        eprintln!("RustInfer build: detected CUDA arch {}", cuda_arch);

        // Directory convention is the classification boundary:
        //   src/kernels/**             -> generic cc-rs device archive
        //   src/kernels/arch/**        -> architecture-specific whole-program archive
        //   src/kernels/third_party/** -> explicitly declared third-party archive
        // Adding another source to an existing specialized group requires no
        // build-script change; a new GPU family adds one CudaArchiveSpec.
        let kernel_paths = find_generic_kernel_files("src/kernels", "cu");
        if kernel_paths.is_empty() {
            println!("cargo:warning=No generic CUDA kernel files (.cu) found in src/kernels/");
        }
        for spec in CUDA_ARCHIVES {
            if let Some(cfg) = spec.rust_cfg {
                println!("cargo:rustc-check-cfg=cfg({cfg})");
            }
        }

        // 2. 配置 cc 编译器
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .opt_level(3)
            .debug(false)
            .flag("-O3")
            .flag("-w")
            .include(&cutlass_include)
            .include(&cudnn_frontend_include)
            .flag("-std=c++17")
            .flag(format!("-arch={}", cuda_arch));
        for spec in CUDA_ARCHIVES
            .iter()
            .filter(|spec| spec.supports(&cuda_arch))
        {
            if let Some(define) = spec.generic_define {
                build.define(define, None);
            }
        }

        // 把所有检测到的 CUDA include 路径都喂给 cc
        for inc in &cuda_includes {
            build.include(inc);
        }

        for path in &kernel_paths {
            build.file(path);
            println!("cargo:rerun-if-changed={}", path.display());
        }
        for path in find_generic_kernel_files("src/kernels", "cuh") {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        build.compile("infer_kernels");
        println!("cargo:rustc-link-lib=static=infer_kernels");
        println!("cargo:rustc-link-lib=cudart");

        // Whole-program compilation preserves architecture-specific WGMMA/TMA
        // codegen and register reallocation that cc-rs's --device-c path loses.
        for spec in CUDA_ARCHIVES
            .iter()
            .filter(|spec| spec.supports(&cuda_arch))
        {
            compile_cuda_archive(spec, &root, &cuda_path, &cuda_includes, &cutlass_include);
        }

        let target = env::var("TARGET").expect("TARGET environment variable not set");

        // 3. 配置 bindgen
        let mut bindgen_builder = bindgen::Builder::default()
            .header("src/wrapper.h")
            // .clang_arg("-I/usr/include/x86_64-linux-gnu") //Conda sysroot（系统根目录）与 Host（宿主系统）头文件冲突
            .clang_arg("-Isrc")
            .clang_arg(format!("--target={}", target))
            .clang_arg(format!("-I{}", cutlass_include.to_string_lossy()))
            .clang_arg("-D_GNU_SOURCE")
            .clang_arg("-D_POSIX_C_SOURCE=200809L")
            .clang_arg("-fms-extensions")
            .clang_arg("-x")
            .clang_arg("c++");

        // [核心修复] 将所有 CUDA include 路径传递给 bindgen/libclang
        for inc in &cuda_includes {
            bindgen_builder = bindgen_builder.clang_arg(format!("-I{}", inc.display()));
        }

        let bindings = bindgen_builder
            .enable_cxx_namespaces()
            .translate_enum_integer_types(true)
            .derive_default(true)
            // === allowlists / rustified_enum 保持不变 ===
            .allowlist_function("cudaMalloc")
            .allowlist_function("cudaFree")
            .allowlist_function("cudaMemcpy")
            .allowlist_function("cudaMemcpyAsync")
            .allowlist_function("cudaMemset")
            .allowlist_function("cudaMemsetAsync")
            .allowlist_function("cudaHostRegister")
            .allowlist_function("cudaHostUnregister")
            .allowlist_function("cudaMemGetInfo")
            .allowlist_function("cudaProfilerStart")
            .allowlist_function("cudaProfilerStop")
            .allowlist_function("cudaGetLastError")
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
            .allowlist_function("cudaStreamIsCapturing")
            .allowlist_type("cudaStreamCaptureStatus")
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

impl CudaArchiveSpec {
    fn supports(&self, cuda_arch: &str) -> bool {
        cuda_arch_family(cuda_arch) == Some(self.arch_family)
    }
}

fn cuda_arch_family(cuda_arch: &str) -> Option<u32> {
    let digits: String = cuda_arch
        .strip_prefix("sm_")?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn compile_cuda_archive(
    spec: &CudaArchiveSpec,
    root: &Path,
    cuda_path: &Path,
    cuda_includes: &[PathBuf],
    cutlass_include: &Path,
) {
    let source_dir = root.join(spec.source_dir);
    let sources = find_all_files(&source_dir, "cu");
    if sources.is_empty() {
        panic!(
            "CUDA archive '{}' has no .cu sources under {}",
            spec.name,
            source_dir.display()
        );
    }

    println!("cargo:rerun-if-changed={}", source_dir.display());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = cuda_path.join("bin/nvcc");
    let mut children = Vec::with_capacity(sources.len());
    let mut objects = Vec::with_capacity(sources.len());

    for (index, source) in sources.into_iter().enumerate() {
        let object = out_dir.join(format!("{}_{}.o", spec.name, index));
        let mut command = std::process::Command::new(&nvcc);
        command
            .args([
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-w",
                "-DNDEBUG",
                "-Xcompiler",
                "-fPIC",
            ])
            .arg(format!("-arch={}", spec.codegen_arch))
            .args(spec.extra_flags)
            .arg("-I")
            .arg(&source_dir)
            .arg("-I")
            .arg(cutlass_include);
        for include in cuda_includes {
            command.arg("-I").arg(include);
        }
        let child = command
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .spawn()
            .unwrap_or_else(|error| {
                panic!("failed to spawn nvcc for {}: {}", source.display(), error)
            });
        children.push((source, child));
        objects.push(object);
    }

    for (source, mut child) in children {
        let status = child
            .wait()
            .unwrap_or_else(|error| panic!("nvcc wait failed for {}: {}", source.display(), error));
        if !status.success() {
            panic!("nvcc failed for {} ({})", source.display(), status);
        }
    }

    let archive = out_dir.join(format!("lib{}.a", spec.name));
    let _ = std::fs::remove_file(&archive);
    let status = std::process::Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .unwrap_or_else(|error| panic!("failed to run ar for '{}': {}", spec.name, error));
    if !status.success() {
        panic!("ar failed for '{}' ({})", spec.name, status);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", spec.name);
    if let Some(cfg) = spec.rust_cfg {
        println!("cargo:rustc-cfg={cfg}");
    }
}

// ---------------------------------------------------------
// 以下是为你新增和优化的辅助工具函数
// ---------------------------------------------------------

/// 自动配置 libclang，消除找不到 shared libraries 的问题
fn auto_configure_libclang() {
    if env::var("LIBCLANG_PATH").is_ok() {
        return; // 用户已手动设置则跳过
    }
    // 尝试从 Conda 环境中找
    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let lib_path = PathBuf::from(conda_prefix).join("lib");
        if lib_path.join("libclang.so").exists() || lib_path.join("libclang.so.1").exists() {
            unsafe {
                env::set_var("LIBCLANG_PATH", lib_path.to_str().unwrap());
            }
            eprintln!(
                "RustInfer build: Auto-configured LIBCLANG_PATH={}",
                lib_path.display()
            );
        }
    }
}

/// 智能收集所有 CUDA 的 Include 路径（专门对付 Conda）
fn get_cuda_include_paths(cuda_path: &Path) -> Vec<PathBuf> {
    let mut includes = Vec::new();
    let candidates = vec![
        cuda_path.join("include"),
        cuda_path.join("targets/x86_64-linux/include"), // Conda 藏头文件的最常见位置
    ];
    for p in candidates {
        if p.exists() {
            includes.push(p);
        }
    }
    includes
}

/// 智能收集所有 CUDA 的 Lib 路径
fn get_cuda_lib_paths(cuda_path: &Path) -> Vec<PathBuf> {
    let mut libs = Vec::new();
    let candidates = vec![
        cuda_path.join("lib64"),
        cuda_path.join("lib"),
        cuda_path.join("targets/x86_64-linux/lib"),
    ];
    for p in candidates {
        if p.exists() {
            libs.push(p);
        }
    }
    libs
}

// =========================================================
// 以下为你原有代码（略作结构调整以保持干净）
// =========================================================

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

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(conda_prefix);
        candidates.push(prefix.join("include"));
        if let Ok(entries) = std::fs::read_dir(prefix.join("lib")) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with("python") {
                    candidates.push(entry.path().join("site-packages/include"));
                }
            }
        }
    }

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

    panic!("cuDNN frontend headers not found.");
}

fn detect_cuda_arch() -> String {
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return arch;
    }
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
        let sm = cap.replace('.', "");
        return format!("sm_{}", sm);
    }
    println!("cargo:warning=Could not detect GPU arch, falling back to sm_80");
    "sm_80".to_string()
}

fn find_cuda_path() -> PathBuf {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let mut candidates = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(value) = env::var(var) {
            candidates.push(PathBuf::from(value));
        }
    }

    if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
        candidates.push(PathBuf::from(conda_prefix));
    }

    if let Ok(home) = env::var("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join("miniconda3"));
        candidates.push(home.join("anaconda3"));
    }
    candidates.extend([PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")]);

    for candidate in candidates {
        // 放宽验证条件，支持 Conda 奇特的文件布局
        let has_headers = candidate.join("include").exists()
            || candidate.join("targets/x86_64-linux/include").exists();
        let has_libraries = candidate.join("lib64").exists()
            || candidate.join("lib").exists()
            || candidate.join("targets/x86_64-linux/lib").exists();
        if has_headers && has_libraries {
            return candidate;
        }
    }

    panic!("CUDA not found. Set CUDA_PATH or CUDA_HOME environment variable");
}

fn find_generic_kernel_files(dir: &str, extension: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let walker = walkdir::WalkDir::new(dir).into_iter();

    let iter = walker.filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        name != "arch" && name != "third_party"
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
    paths.sort();
    paths
}

fn find_all_files(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut paths: Vec<_> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == extension))
        .collect();
    paths.sort();
    paths
}
