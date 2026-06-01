//! Aggregate Root #3: `WorkerNode<S>` and its companion value objects.
//!
//! Layout:
//! - `states.rs`   — sealed `NodeState`, `Ready`, `Lost`
//! - `capacity.rs` — `Capacity` value object (NewType-typed)
//! - `node.rs`     — main `WorkerNode<S>` aggregate root
//!
//! The control-plane bootstrap currently builds a
//! [`crate::infrastructure::transport::control_plane::WorkerGroup`]
//! that wraps the same handshake data; `WorkerNode<Ready>` lives
//! alongside it as the typed surface for code paths that want
//! state-machine guarantees.

mod capacity;
mod node;
mod states;

pub use capacity::Capacity;
pub use node::WorkerNode;
pub use states::{Lost, NodeState, Ready};
