# RustInfer Context

## Language

**Worker**:
A process that owns model execution resources on a device. It loads a model instance, allocates runtime resources, runs the worker-local continuous batching loop, and communicates with the Scheduler over ZMQ.
_Avoid_: executor process, GPU process

**Scheduler**:
The process that owns request lifecycle, admission, Worker readiness, and control decisions. It sends requests to ready Workers and receives step outputs, but does not drive each decode step.
_Avoid_: decode driver

**Protocol Package**:
The `infer-protocol` crate that owns all wire types shared between RustInfer processes. Protocol files are named by message direction, such as `server_to_scheduler`, `scheduler_to_worker`, and `worker_to_scheduler`.
_Avoid_: defining wire types inside Worker or Scheduler crates

**Control Plane**:
The slow-path ZMQ + MessagePack protocol used for Worker lifecycle messages such as hello, ready, heartbeat, load, cancel, drain, and error reporting.
_Avoid_: management API, admin channel

**Data Plane**:
The hot-path ZMQ + MessagePack protocol used for inference batch commands and step outputs.
_Avoid_: inference API when referring to Worker communication

**Worker Lifecycle**:
The Worker state machine from startup through registration, model loading, profiling, warmup, ready, running, draining, and error states.
_Avoid_: boot flow, startup flow when discussing state semantics

**Autonomous Decode**:
The current Worker-owned decode loop where the Worker keeps active decode state in-process and continuously executes decode steps without Scheduler-driven per-step commands.
_Avoid_: unmanaged decode

**Active Request Table**:
The Scheduler-side record of requests that have been assigned to a Worker and have not finished yet. It is a lifecycle ledger, not a decode scheduler.
_Avoid_: DecodeLedger

**Worker Group**:
A logical group of one or more Worker ranks serving one model instance. Today this is a single Worker on one GPU; future multi-card support can make it multiple ranks.
_Avoid_: cluster, pool when referring to ranks of one model instance

## Relationships

- A **Scheduler** manages many **Workers**.
- A **Worker Group** contains one or more **Workers** serving one model instance.
- The **Protocol Package** defines **Control Plane** and **Data Plane** wire types.
- The **Control Plane** manages **Worker Lifecycle**.
- The **Data Plane** carries inference work.
- **Autonomous Decode** runs inside the **Worker**.
- The **Active Request Table** mirrors request lifecycle in the **Scheduler** without controlling decode steps.
