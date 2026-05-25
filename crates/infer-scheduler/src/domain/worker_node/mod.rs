//! Aggregate Root #3: `WorkerNode<S>` and its companion value objects.
//!
//! Layout:
//! - `states.rs`   — sealed `NodeState`, `Ready`, `Lost`
//! - `capacity.rs` — `Capacity` value object (NewType-typed)
//! - `node.rs`     — main `WorkerNode<S>` aggregate root
//!
//! The legacy [`crate::infrastructure::transport::control_plane::WorkerGroup`] is **not yet
//! removed**: it is still wired into bootstrap (`main.rs`), the
//! engine, and the control plane. Step 18 (engine slim-down) will
//! migrate those callers to `WorkerNode<Ready>`; this module exists
//! standalone now so subsequent steps can construct it without
//! disturbing the live code path.

mod capacity;
mod node;
mod states;

pub use capacity::Capacity;
pub use node::WorkerNode;
pub use states::{Lost, NodeState, Ready};
