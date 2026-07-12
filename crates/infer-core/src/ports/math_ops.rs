// Math ports preserve explicit matrix/tensor dimensions for backend kernels.
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use crate::ports::{OpError, OpResult};
use infer_core::dtype::Dtype;
use infer_core::dtype::quant::QuantScheme;
use infer_core::exec::ExecDevice as Device;
use infer_core::tensor::Tensor;
use infer_core::types::{Shape, Strides};

pub trait MathOps: Device {
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>> {
        Tensor::<T, Self>::zeros(shape, device)
    }

    fn add<T: Dtype>(
        scope: &Self::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn add_inplace<T: Dtype>(
        scope: &Self::Scope,
        dst: &mut Tensor<T, Self>,
        src: &Tensor<T, Self>,
    ) -> OpResult<()>;

    fn ewise_mul<T: Dtype>(
        scope: &Self::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn scalar_mul_inplace<T: Dtype>(
        scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        scalar: f64,
    ) -> OpResult<()>;

    fn broadcast_mul_inplace<T: Dtype>(
        scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()>;

    fn broadcast_add_inplace<T: Dtype>(
        scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()>;

    fn matmul<T: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<A, Self>,
        weight: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>,
        zeros: Option<&Tensor<W, Self>>,
        scheme: &QuantScheme,
    ) -> OpResult<()>;

    fn rmsnorm<T: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    fn rmsnorm_inplace<T: Dtype>(
        scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    fn silu_inplace<T: Dtype>(scope: &Self::Scope, x: &mut Tensor<T, Self>) -> OpResult<()>;

    fn softmax<T: Dtype>(
        scope: &Self::Scope,
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn rope_inplace<T: Dtype>(
        scope: &Self::Scope,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>,
        cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
    ) -> OpResult<()>;

    fn sdpa<T: Dtype>(
        scope: &Self::Scope,
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        mask: Option<&Tensor<T, Self>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()>;

    fn embedding<T: Dtype>(
        scope: &Self::Scope,
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn split_cols<T: Dtype>(
        scope: &Self::Scope,
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()>;

    fn concat_seq<T: Dtype>(
        scope: &Self::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn cast<S: Dtype, T: Dtype>(
        scope: &Self::Scope,
        src: &Tensor<S, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn bitcast<S: Dtype, T: Dtype>(
        src: &Tensor<S, Self>,
        new_shape: Shape,
    ) -> OpResult<Tensor<T, Self>> {
        let src_bytes = src.numel() * S::SIZE_BYTES;
        let dst_bytes = new_shape.numel() * T::SIZE_BYTES;
        let offset_bytes = src.offset_elems() * S::SIZE_BYTES;
        if src_bytes != dst_bytes {
            return Err(OpError::Shape(format!(
                "bitcast: byte size mismatch {} -> {}",
                src_bytes, dst_bytes
            )));
        }
        if !offset_bytes.is_multiple_of(T::SIZE_BYTES) {
            return Err(OpError::Shape(format!(
                "bitcast: offset {} is not aligned to {} bytes",
                offset_bytes,
                T::SIZE_BYTES
            )));
        }
        Ok(Tensor::from_raw_parts(
            Arc::clone(src.storage()),
            new_shape,
            Strides::contiguous_for(&new_shape),
            offset_bytes / T::SIZE_BYTES,
            src.is_contiguous(),
        ))
    }
}

#[macro_export]
macro_rules! impl_math_ops_via_core_ops {
    ($backend:ty) => {
        impl $crate::ports::MathOps for $backend {
            fn add<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                a: &infer_core::tensor::Tensor<T, Self>,
                b: &infer_core::tensor::Tensor<T, Self>,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::add(a, b, dst)
            }

            fn add_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
                src: &infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::add_inplace(dst, src)
            }

            fn ewise_mul<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                a: &infer_core::tensor::Tensor<T, Self>,
                b: &infer_core::tensor::Tensor<T, Self>,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::ewise_mul(a, b, dst)
            }

            fn scalar_mul_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                x: &mut infer_core::tensor::Tensor<T, Self>,
                scalar: f64,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::scalar_mul_inplace(x, scalar)
            }

            fn broadcast_mul_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                x: &mut infer_core::tensor::Tensor<T, Self>,
                scale: &infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::broadcast_mul_inplace(x, scale)
            }

            fn broadcast_add_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                x: &mut infer_core::tensor::Tensor<T, Self>,
                bias: &infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::broadcast_add_inplace(x, bias)
            }

            fn matmul<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                input: &infer_core::tensor::Tensor<T, Self>,
                weight: &infer_core::tensor::Tensor<T, Self>,
                output: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::matmul(input, weight, output)
            }

            fn matmul_quant<
                A: infer_core::dtype::Dtype,
                W: infer_core::dtype::Dtype,
                O: infer_core::dtype::Dtype,
            >(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                input: &infer_core::tensor::Tensor<A, Self>,
                weight: &infer_core::tensor::Tensor<W, Self>,
                output: &mut infer_core::tensor::Tensor<O, Self>,
                scales: &infer_core::tensor::Tensor<A, Self>,
                zeros: Option<&infer_core::tensor::Tensor<W, Self>>,
                scheme: &infer_core::dtype::quant::QuantScheme,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::matmul_quant(
                    input, weight, output, scales, zeros, scheme,
                )
            }

            fn rmsnorm<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                input: &infer_core::tensor::Tensor<T, Self>,
                weight: &infer_core::tensor::Tensor<T, Self>,
                output: &mut infer_core::tensor::Tensor<T, Self>,
                eps: f32,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::rmsnorm(input, weight, output, eps)
            }

            fn rmsnorm_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                x: &mut infer_core::tensor::Tensor<T, Self>,
                weight: &infer_core::tensor::Tensor<T, Self>,
                eps: f32,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::rmsnorm_inplace(x, weight, eps)
            }

            fn silu_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                x: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::silu_inplace(x)
            }

            fn softmax<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                input: &infer_core::tensor::Tensor<T, Self>,
                output: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::softmax(input, output)
            }

            fn rope_inplace<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                q: &mut infer_core::tensor::Tensor<T, Self>,
                k: &mut infer_core::tensor::Tensor<T, Self>,
                sin: &infer_core::tensor::Tensor<T, Self>,
                cos: &infer_core::tensor::Tensor<T, Self>,
                positions: &infer_core::tensor::Tensor<i32, Self>,
                head_num: usize,
                kv_head_num: usize,
                head_dim: usize,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::rope_inplace(
                    q,
                    k,
                    sin,
                    cos,
                    positions,
                    head_num,
                    kv_head_num,
                    head_dim,
                )
            }

            fn sdpa<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                q: &infer_core::tensor::Tensor<T, Self>,
                k: &infer_core::tensor::Tensor<T, Self>,
                v: &infer_core::tensor::Tensor<T, Self>,
                output: &mut infer_core::tensor::Tensor<T, Self>,
                mask: Option<&infer_core::tensor::Tensor<T, Self>>,
                num_heads: usize,
                num_kv_heads: usize,
                head_dim: usize,
                scale: f32,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                match mask {
                    Some(mask) => <Self as $crate::ports::DiffusionOps>::sdpa_masked(
                        q,
                        k,
                        v,
                        output,
                        mask,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    ),
                    None => <Self as $crate::ports::DiffusionOps>::sdpa(
                        q,
                        k,
                        v,
                        output,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    ),
                }
            }

            fn embedding<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                table: &infer_core::tensor::Tensor<T, Self>,
                indices: &infer_core::tensor::Tensor<i32, Self>,
                output: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::embedding(table, indices, output)
            }

            fn split_cols<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                src: &infer_core::tensor::Tensor<T, Self>,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
                rows: usize,
                total_cols: usize,
                col_offset: usize,
                dst_cols: usize,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::split_cols(
                    src, dst, rows, total_cols, col_offset, dst_cols,
                )
            }

            fn concat_seq<T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                a: &infer_core::tensor::Tensor<T, Self>,
                b: &infer_core::tensor::Tensor<T, Self>,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::concat_seq(a, b, dst)
            }

            fn cast<S: infer_core::dtype::Dtype, T: infer_core::dtype::Dtype>(
                scope: &<Self as infer_core::exec::ExecDevice>::Scope,
                src: &infer_core::tensor::Tensor<S, Self>,
                dst: &mut infer_core::tensor::Tensor<T, Self>,
            ) -> $crate::ports::OpResult<()> {
                let _guard = infer_core::exec::ExecScope::enter(scope);
                <Self as $crate::ports::CoreOps>::cast_dtype(src, dst)
            }
        }
    };
}
