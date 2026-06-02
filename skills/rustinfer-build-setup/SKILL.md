---
name: rustinfer-build-setup
description: This skill should be used when bringing RustInfer up on a fresh Ubuntu 24.04 / CUDA host — installing Rust, fixing nvcc/glibc header conflicts, installing cuDNN, and getting `cargo build --release` to succeed end-to-end. Triggers on "set up RustInfer", "install Rust and build", "nvcc rsqrt noexcept error", "cudnn.h not found", or first-time host bring-up.
---

# RustInfer Build Setup

## Purpose

Get a fresh Ubuntu 24.04 + NVIDIA host to a successful `cargo build --release` for RustInfer. Captures the gotchas hit on a real H20 host: locked-down network, glibc 2.43 ↔ CUDA 13.1 nvcc header conflict, missing cuDNN dev package.

## Environment baseline

- Ubuntu 24.04
- NVIDIA driver present (`nvidia-smi` works), but **`nvcc` not on PATH** even though `/usr/local/cuda` exists
- Rust **not installed** (`rustc` missing, `/root/.cargo/` empty)
- **No direct outbound internet.** All public endpoints (`sh.rustup.rs`, public mirror sites, `github.com`, `pypi.org`, `developer.download.nvidia.com`) return `Network is unreachable` until an HTTP proxy is configured.
- An HTTP proxy is reachable from the host (ask the user for the URL).

## Network failure rule (top priority)

**If at any step you hit `Network is unreachable`, DNS failure, or any sustained connection timeout to a public host, STOP immediately and tell the user.** Do not silently retry, swap mirrors, or hunt for offline packages — this host's network is locked down and only the user knows the right proxy/mirror to use. Report the exact failing URL and error, then ask whether they have an HTTP proxy (or internal mirror) available.

Same rule applies after the proxy is configured: if a fresh `curl` check still fails, stop and tell the user — don't keep poking.

## Step-by-step

### 1. Confirm network reachability — STOP and ask the user the moment it fails

Before doing anything else, sanity-check outbound HTTPS:

```bash
curl -sI --max-time 10 https://sh.rustup.rs | head -1
```

**If this does not return `HTTP/1.1 200 OK`, STOP immediately and tell the user** — do not retry, do not try alternate mirrors, do not full-disk-search for offline packages. Network failures on this kind of host are almost always "no outbound route, need a proxy", and only the user knows the right proxy URL. Report what you saw (exact error: `Network is unreachable`, DNS failure, timeout, etc.) and ask:

> "外网连不上(具体错误 X)。这台机器有可用的 HTTP 代理地址吗?或者其他内网镜像?"

Once the user supplies a proxy URL, export it in every shell that will run `curl` / `apt` / `cargo`:

```bash
export http_proxy=<PROXY_URL>
export https_proxy=$http_proxy
export all_proxy=$http_proxy
curl -sI --max-time 10 https://sh.rustup.rs | head -1   # must now return 200
```

If the proxy still doesn't give a 200, **stop again and tell the user** — don't keep guessing. Every later step (rustup, apt, crates.io) depends on this connectivity.

### 2. Install Rust via the official rustup script

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
rustc --version   # expect 1.96+ (RustInfer Cargo.toml uses resolver = "3", needs 1.84+)
```

Do **not** try `apt-get install rustc cargo` — the distro `rustc` package is fine version-wise, but apt fetch tends to bypass the proxy and fails with `Network is unreachable` even when the proxy export above works for `curl`. rustup over the proxy is the reliable path.

### 3. Install RustInfer's C/C++ build deps

```bash
apt-get update
apt-get install -y clang libclang-dev pkg-config libssl-dev libopenblas-dev
```

(README also lists these. `clang`/`libclang-dev` are needed for `bindgen` to parse `wrapper.h`.)

### 4. Fix the CUDA toolkit — known glibc 2.43 ↔ CUDA 13.1 conflict

The default `/usr/local/cuda` on this image points to **CUDA 13.1**, whose `crt/math_functions.h` declares `rsqrt`/`rsqrtf` without `noexcept`, while glibc 2.43's `bits/mathcalls.h` declares them with `noexcept(true)`. Building any `.cu` file produces:

```
error: exception specification is incompatible with that of previous function "rsqrtf"
   extern float rsqrtf (float __x) noexcept (true); ...
                                   ^
```

Fix: install **CUDA 13.3** toolkit. The apt alternative auto-flips `/usr/local/cuda` to point at it.

```bash
apt-get install -y cuda-toolkit-13-3
ls -l /usr/local/cuda                  # → /etc/alternatives/cuda → /usr/local/cuda-13.3
nvcc --version | tail -1               # → release 13.3, V13.3.x
```

If the symlink doesn't flip automatically:

```bash
update-alternatives --set cuda /usr/local/cuda-13.3
```

### 5. Install cuDNN (build.rs needs `cudnn.h` and links `-lcudnn`)

```bash
apt-get install -y libcudnn9-dev-cuda-13 libcudnn9-cuda-13
```

Without this, `bindgen` fails with:

```
src/infrastructure/cuda/wrapper.h:5:10: fatal error: 'cudnn.h' file not found
thread 'main' panicked at crates/infer-worker/build.rs:156: Unable to generate bindings: ClangDiagnostic
```

### 6. Build

Always export the proxy + CUDA env in the same shell that runs cargo:

```bash
. "$HOME/.cargo/env"
export http_proxy=<PROXY_URL>
export https_proxy=$http_proxy
export all_proxy=$http_proxy
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

cd /root/RustInfer
cargo build --release
```

`infer-worker`'s default features include `cuda`, so plain `cargo build --release` builds the CUDA path. Build time on a clean host is ~1–2 minutes after the proxy is up. Architecture is auto-detected from `nvidia-smi` (`sm_90` for H20); override with `CUDA_ARCH=sm_XX` if needed.

Successful build produces in `target/release/`:

- `rustinfer-worker`
- `rustinfer-scheduler`
- `rustinfer-server`
- `rustinfer-frontend`

## Verification checklist

```bash
ls /root/RustInfer/target/release/rustinfer-{worker,scheduler,server}
/root/RustInfer/target/release/rustinfer-worker --help | head -5
nvidia-smi | head -8
nvcc --version | tail -1     # must be 13.3.x or newer
```

## Common failure → fix table

| Symptom | Root cause | Fix |
|---|---|---|
| `curl: Network is unreachable` for any public host | No outbound; proxy not exported | Export `http_proxy=<PROXY_URL>` (and `https_proxy`/`all_proxy`) |
| `apt-get install` fails fetching `.deb` while proxy works for curl | apt didn't inherit env in some contexts | Ensure `http_proxy`/`https_proxy` are exported in the same shell, then re-run `apt-get update && apt-get install ...` |
| `error: exception specification is incompatible with that of previous function "rsqrtf"` | CUDA 13.1 nvcc + glibc 2.43 mismatch | `apt-get install cuda-toolkit-13-3`; ensure `/usr/local/cuda` → 13.3 |
| `fatal error: 'cudnn.h' file not found` (bindgen) | cuDNN dev package missing | `apt-get install libcudnn9-dev-cuda-13 libcudnn9-cuda-13` |
| `nvcc: command not found` even though `/usr/local/cuda` exists | PATH not exported | `export PATH=/usr/local/cuda/bin:$PATH` |
| `cargo: command not found` after rustup | shell didn't source env | `. "$HOME/.cargo/env"` |
| linker / cc-rs errors mention wrong compiler | `.cargo/config.toml` pins `/usr/bin/cc` and `/usr/local/cuda/bin/nvcc` | Confirm `cc`/`c++` exist; both should after `apt install clang` (gcc 13 is also pre-installed) |

## Why these versions specifically

- **Rust 1.96 (rustup stable)**: workspace uses `resolver = "3"` (needs 1.84+); rustup stable is the simplest fix.
- **CUDA 13.3**: 13.1's `math_functions.h` predates glibc 2.43's `noexcept(true)` math decls. 13.3 fixes the header. Don't downgrade glibc — it's the OS.
- **cuDNN 9 for CUDA 13**: matches the toolkit major; `build.rs` calls `cargo:rustc-link-lib=cudnn` and bindgen pulls in `cudnn.h`.

## Things to skip on this host

- **Don't** run `cargo build --features cuda` — `cuda` is already a default feature.
- **Don't** `apt-get install rustc cargo` — apt fetch tends to bypass the proxy and fails; rustup is the reliable path.
- **Don't** add a `[target.x86_64-unknown-linux-gnu]` rustflags override unless cc-rs actually fails — the existing `.cargo/config.toml` is sufficient.
- **Don't** try to fix the rsqrt/noexcept conflict by editing CUDA headers. Just install 13.3.

## Running after build

A successful build only gets you binaries — running them needs model files. Point the worker at whatever local model directory the host provides (e.g. a Llama-3.2-1B-Instruct checkpoint). See `skills/rustinfer-benchmark` and `skills/rustinfer-nsys-profile` for run/profile workflows.
