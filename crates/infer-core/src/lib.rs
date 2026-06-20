//! # `infer-core` — the GPU-free foundation **and** static backend interface
//!
//! The bottom of the static modular-backend dependency DAG. It carries the
//! value/tensor foundation (`types`/`dtype`/`error`/`device`/`storage`/`tensor`/
//! `exec`/`plan`) **and** the op-port trait surface (`ports`/`kv`/`component`)
//! that every backend crate implements — all resolved at compile time, no C ABI.
//! Builds with `rustc` alone. (The op-port traits previously lived in a separate
//! `infer-backend-abi` crate; in the static design there is no ABI boundary to
//! justify the split, so they are folded in here.)
//!
//! Foundation modules: [`types`], [`dtype`], [`error`], [`device`], [`storage`],
//! [`tensor`], [`exec`], [`plan`], [`env_flags`].
//! Backend-interface modules: [`ports`] (the op-port traits +
//! `impl_math_ops_via_core_ops!` macro + `Sampler`), [`kv`] (paged KV pool +
//! `KvView`/`LayerKv`), [`component`] (`Component`/`Hidden`/`LayerRange`).

// Allow this crate's own modules (moved in from `infer-backend-abi`) to keep
// referring to it as `infer_core::…`, and let the `impl_math_ops_via_core_ops!`
// macro use `$crate::…` paths uniformly whether expanded here or in a backend.
extern crate self as infer_core;

pub mod device;
pub mod storage;
pub mod tensor;
pub mod exec;
pub mod plan;
pub mod dtype;
pub mod error;
pub mod types;
pub mod env_flags;

// ── backend interface (folded in from infer-backend-abi) ──
pub mod component;
pub mod kv;
pub mod ports;
