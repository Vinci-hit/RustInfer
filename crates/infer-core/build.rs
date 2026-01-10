use std::env;
use std::path::PathBuf;

fn main() {
    // 检查 "cuda" feature 是否被启用
    #[cfg(feature = "cuda")]{
        let kernel_paths = find_files("src/op/kernels/cuda", "cu");
        
        if kernel_paths.is_empty() {
            // 如果没有找到任何 .cu 文件，可能是一个配置错误，可以选择性地警告或 panic
            println!("cargo:warning=No CUDA kernel files (.cu) found in src/op/kernels/cuda/");
            // return; // 如果你希望在这种情况下停止构建
        }
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cublas");
        // 对应 cublasLt.h
        println!("cargo:rustc-link-lib=cublasLt");
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = PathBuf::from(manifest_dir);
        let cutlass_include = root.join("src/op/kernels/cuda/third_party");
        if !cutlass_include.exists() {
            panic!(
                "Cutlass include directory not found at: {:?}", 
                cutlass_include
            );
        }
        // 1. 使用 cc crate 编译 CUDA 代码 (这部分保持不变)
        let mut build = cc::Build::new();
        build.cuda(true)
        .flag("-w") 
        .include(&cutlass_include)
        .flag("-std=c++17")
        .flag("-arch=sm_89");// 请根据你的 GPU 架构修改       // 禁用优化（避免变量被优化掉，调试时能看到真实值）

        for path in &kernel_paths {
            build.file(path);
            println!("cargo:rerun-if-changed={}", path.display());
        }
        build.compile("infer_kernels");
        println!("cargo:rustc-link-lib=static=infer_kernels");
        println!("cargo:rustc-link-lib=cudart");

        let target = env::var("TARGET").expect("TARGET environment variable not set");

        // 4. 使用 bindgen 生成 Rust FFI 绑定
        let bindings = bindgen::Builder::default()
            .header("src/cuda/wrapper.h")
            // 告诉 bindgen/libclang CUDA 头文件的位置
            .clang_arg(format!("-I{}/include", env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".into())))
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
            .allowlist_function("cudaGetErrorString")
            .allowlist_function("cudaGetErrorName")
            .allowlist_function("cudaGetDevice")
            .allowlist_function("cudaSetDevice")
            .allowlist_function("cudaStreamCreate")
            .allowlist_function("cudaStreamDestroy")
            .allowlist_function("cudaDeviceSynchronize")
            .allowlist_function("cudaStreamSynchronize")
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
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .rustified_enum("cudaError_t")
            .rustified_enum("cudaMemcpyKind")
            .generate()
            .expect("Unable to generate bindings");
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}

/// 辅助函数：递归地查找指定目录中具有特定扩展名的文件
#[cfg(feature = "cuda")]
fn find_files(dir: &str, extension: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(dir) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.is_file() && let Some(ext) = path.extension() && ext == extension {
            paths.push(path.to_path_buf());
        }
    }
    paths
}