//! Device-independent compiler IR. Execution and physical memory planning
//! are separate from these logical graph definitions.

pub mod graph;
pub mod node;
pub mod op;
pub mod shape;

// Share the runtime dtype enum with the compiler IR.
pub use crate::types::DataType;
pub use graph::Graph;
pub use node::{Node, ValueId};
pub use op::{BinaryOp, FusionKind, Op, ReduceOp, UnaryOp};
pub use shape::Dim;
