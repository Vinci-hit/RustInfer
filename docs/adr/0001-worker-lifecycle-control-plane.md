# ADR 0001: Keep Autonomous Decode and Add a ZMQ Control Plane

## Status

Accepted

## Context

RustInfer currently uses a three-process serving stack: Scheduler, Worker, and HTTP Server. The Worker owns model execution and runs an in-process decode loop with worker-local continuous batching. This design is a major part of the current single-card performance profile.

We want a more modern Worker Lifecycle with registration, readiness, model loading phases, profiling, warmup, cancel, drain, and future Worker Group support. At the same time, the near-term target remains single-machine single-card, and preserving the current hot path is mandatory.

## Decision

Keep Autonomous Decode as the default and only decode execution model.

The Scheduler will not issue per-step decode commands. It will own lifecycle and request facts through an Active Request Table, while the Worker continues to own active decode execution in-process.

Add a Control Plane using the current transport style: ZMQ + MessagePack. This keeps operational and dependency complexity low and matches the existing Data Plane.

Implement the design in four phases:

1. Add Control Plane handshake, Worker readiness, progress, heartbeat, and error reporting.
2. Move model loading behind a Scheduler-issued LoadModel command.
3. Add cancel, drain, unload, and an Active Request Table in the Scheduler.
4. Introduce Worker Group as a logical abstraction for future multi-card deployments.

Explicitly do not implement scheduler-driven decode.

## Consequences

Positive:

- Preserves current Worker-local continuous batching and decode performance.
- Gives Scheduler lifecycle visibility without putting it in the decode hot path.
- Enables cancel, drain, readiness gating, profiling reports, and safer startup sequencing.
- Prepares the model for future multi-card Worker Groups without requiring multi-card execution now.
- Reuses ZMQ + MessagePack, minimizing disruption.

Negative:

- Scheduler state remains a lifecycle mirror, not the source of truth for each decode micro-step.
- Some future features, such as global per-token fairness or aggressive preemption, may require additional Worker control messages.
- The Control Plane and Data Plane must stay carefully separated to avoid accidental hot-path regressions.

## Rejected Alternatives

### Scheduler-driven decode

Rejected. It would make Scheduler responsible for every decode step and could erase the current Worker-local performance advantage. This is explicitly out of scope.

### Replace transport with gRPC immediately

Rejected for now. gRPC may become useful for multi-node operations and observability later, but ZMQ + MessagePack is sufficient for the current single-machine plan and matches the existing code.
