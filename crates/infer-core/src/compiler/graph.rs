use super::{Node, ValueId};

/// Logical graph with ordered boundary inputs and outputs.
///
/// Node IDs index `nodes`; dead nodes keep their slots. For fused regions,
/// boundary inputs correspond positionally to the enclosing node's inputs.
/// IDs inside a fused subgraph are local to that subgraph.
/// This record does not yet validate edges or operation signatures.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    /// 【图参数】记录哪些 ValueId 是模型权重（对应 SafeTensors 里的权重）
    pub params: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}
