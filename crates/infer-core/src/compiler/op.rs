use smallvec::SmallVec;

use super::{DataType, Dim, Graph};

/// Model-independent compiler IR operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // Memory and sources.
    Input {
        name: String,
    },
    Parameter {
        name: String,
        id: usize,
    },
    Constant {
        bytes: Vec<u8>,
    },

    // Element-wise operations. Fusion eligibility is decided by compiler passes.
    Unary(UnaryOp),
    Binary(BinaryOp),
    /// Element-wise `where(condition, x, y)`.
    Select,
    Cast {
        to: DataType,
    },

    // Reductions.
    Reduce {
        op: ReduceOp,
        axes: SmallVec<[usize; 4]>,
        keep_dims: bool,
    },

    // Contractions and dense computation.
    MatMul {
        transpose_a: bool,
        transpose_b: bool,
    },
    Conv2d {
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },

    // Views and movement. Whether materialization is needed depends on layout.
    Reshape {
        shape: SmallVec<[Dim; 4]>,
    },
    Transpose {
        perm: SmallVec<[usize; 4]>,
    },
    /// Half-open `[start, end)` ranges, one per axis, with unit step.
    Slice {
        ranges: SmallVec<[(usize, usize); 4]>,
    },
    Concat {
        axis: usize,
    },
    Broadcast {
        target_shape: SmallVec<[Dim; 4]>,
    },

    /// Region selected for fused lowering. Kernel and workspace requirements
    /// are determined by the backend, not guaranteed by this container.
    Fused {
        subgraph: Box<Graph>,
        kind: FusionKind,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
    Relu,
    Silu,
    Gelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Pow,
    CmpEq,
    CmpLt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionKind {
    /// Pure element-wise fusion.
    ElementWise,
    /// Matrix multiplication or convolution with a fused epilogue.
    Epilogue,
    /// A region containing reductions, such as softmax or RMSNorm.
    Reduction,
}
