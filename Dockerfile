# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG RUST_VERSION=1.91.1
ARG CUDA_ARCH=sm_90
ARG CUDNN_FRONTEND_VERSION=1.18.0

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        libclang-dev \
        pkg-config \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
        --output /tmp/rustup-init \
        https://sh.rustup.rs \
    && chmod 0755 /tmp/rustup-init \
    && /tmp/rustup-init \
        --default-toolchain "${RUST_VERSION}" \
        --profile minimal \
        --no-modify-path \
        --yes \
    && rm /tmp/rustup-init \
    && rustc --version \
    && cargo --version

# RustInfer includes cuDNN Frontend headers from this wheel. They are only
# required while compiling the CUDA kernels and are not copied to the runtime.
RUN python3 -m pip install \
        --break-system-packages \
        --no-cache-dir \
        --no-deps \
        "nvidia_cudnn_frontend==${CUDNN_FRONTEND_VERSION}" \
    && frontend_include="$(python3 -c \
        'import site; print(site.getsitepackages()[0] + "/include")')" \
    && test -f "${frontend_include}/cudnn_frontend.h" \
    && ln -s "${frontend_include}" /opt/cudnn-frontend

ENV CUDA_ARCH=${CUDA_ARCH} \
    CUDNN_FRONTEND_INCLUDE_DIR=/opt/cudnn-frontend

WORKDIR /src

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates

RUN cargo build \
        --locked \
        --release \
        -p infer-worker \
        -p infer-scheduler \
        -p infer-server \
    && strip \
        target/release/rustinfer-worker \
        target/release/rustinfer-scheduler \
        target/release/rustinfer-server

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG CUDA_ARCH=sm_90
ARG VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="RustInfer" \
      org.opencontainers.image.description="RustInfer OpenAI-compatible CUDA inference server" \
      org.opencontainers.image.source="https://github.com/Vinci-hit/RustInfer" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libstdc++6 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd \
        --create-home \
        --home-dir /home/rustinfer \
        --shell /usr/sbin/nologin \
        --uid 10001 \
        --user-group \
        rustinfer \
    && install -d -o rustinfer -g rustinfer /models /opt/rustinfer

COPY --from=builder /src/target/release/rustinfer-worker /usr/local/bin/
COPY --from=builder /src/target/release/rustinfer-scheduler /usr/local/bin/
COPY --from=builder /src/target/release/rustinfer-server /usr/local/bin/
COPY docker/entrypoint.sh /usr/local/bin/rustinfer-container
COPY LICENSE /usr/share/licenses/rustinfer/LICENSE
COPY crates/infer-backend-cuda/src/kernels/third_party/fa3/LICENSE \
    /usr/share/licenses/rustinfer/FA3-LICENSE

RUN chmod 0755 /usr/local/bin/rustinfer-container

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUSTINFER_CUDA_ARCH=${CUDA_ARCH} \
    RUSTINFER_MODEL=/models/model \
    RUSTINFER_HOST=0.0.0.0 \
    RUSTINFER_PORT=8000

USER rustinfer
WORKDIR /opt/rustinfer

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=4 \
    CMD port="$(cat /tmp/rustinfer-port 2>/dev/null || printf '%s' "${RUSTINFER_PORT}")"; \
        curl --fail --silent --show-error \
        "http://127.0.0.1:${port}/ready" >/dev/null || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/rustinfer-container"]
CMD ["serve"]
