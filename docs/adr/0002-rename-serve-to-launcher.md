# ADR 0002: Rename infer-serve to infer-launcher

## Status

Accepted

## Context

RustInfer had two similarly named binaries:

- `rustinfer-serve`: a launcher that starts Scheduler, Worker, and HTTP Server child processes.
- `rustinfer-server`: the actual HTTP/OpenAI-compatible API server.

The names `serve` and `server` are too close. The launcher is not itself the HTTP server, so the old name made process ownership and runtime responsibilities harder to understand.

SGLang uses clearer naming around launch entrypoints and server components: the user-facing launcher is separate from the HTTP server and scheduler/runtime internals.

## Decision

Rename the launcher crate and binary:

- crate: `infer-serve` → `infer-launcher`
- binary: `rustinfer-serve` → `rustinfer-launch`
- path: `crates/infer-serve` → `crates/infer-launcher`

Keep the HTTP server crate and binary unchanged:

- crate: `infer-server`
- binary: `rustinfer-server`

## Consequences

Positive:

- `rustinfer-launch` clearly means “start the full stack”.
- `rustinfer-server` clearly remains the HTTP API server.
- Future lifecycle/control-plane work has a clearer process map.

Negative:

- Existing users of `rustinfer-serve` need to switch to `rustinfer-launch`.
- Build and deployment scripts that reference the old package or binary name must be updated.
