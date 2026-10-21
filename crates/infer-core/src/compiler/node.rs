//! Device-independent computation-graph vocabulary.
//!
//! Nodes describe logical computation. Tensor metadata, physical storage,
//! kernel selection, and execution belong to other layers.

use smallvec::SmallVec;

use super::Op;

/// Identifies a node and its logical result within one graph.
///
/// All operations, including inputs, weights, and result selection, are nodes.
/// IDs are local to their graph and do not identify physical allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

/// One logical operation in a computation graph.
///
/// `Op` describes the operation and its static attributes, without owning device
/// tensors or executing kernels. Input positions are significant; their arity
/// and meaning are defined by `Op`. Each node's result is referenced by its
/// `ValueId`. Selecting a result from a multi-result operation is itself an
/// operation node. Inputs, weights, and external resources are source nodes.
///
/// This is an IR record, not a validated graph: a graph builder must check ID
/// uniqueness, operation signatures, and dependency validity.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: ValueId,
    /// Optional human-readable label, e.g. `layers.0.ffn.gate_up`.
    /// Names need not be unique; references always use `ValueId`.
    pub name: Option<String>,
    pub op: Op,
    pub inputs: SmallVec<[ValueId; 4]>,
    /// Distinct nodes whose `inputs` contain this node's ID. Graph mutations
    /// must keep this reverse index consistent with the input edges.
    pub users: SmallVec<[ValueId; 4]>,
    /// Marks a logically removed node while preserving its ID and storage slot.
    /// New nodes start alive (`false`). Live nodes must not depend on dead nodes.
    pub is_dead: bool,
}
