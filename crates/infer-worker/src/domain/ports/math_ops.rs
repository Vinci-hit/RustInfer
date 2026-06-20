use std::marker::PhantomData;
use std::sync::Arc;

use crate::domain::dtype::Dtype;
use crate::domain::dtype::quant::QuantScheme;
use crate::domain::exec::Device;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Shape, Strides};

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
        Ok(Tensor {
            storage: Arc::clone(src.storage()),
            shape: new_shape,
            strides: Strides::contiguous_for(&new_shape),
            offset_elems: offset_bytes / T::SIZE_BYTES,
            numel: new_shape.numel(),
            is_contiguous: src.is_contiguous(),
            _marker: PhantomData,
        })
    }
}

#[macro_export]
macro_rules! impl_math_ops_via_core_ops {
    ($backend:ty) => {
        impl $crate::domain::ports::MathOps for $backend {
            fn add<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                a: &$crate::domain::tensor::Tensor<T, Self>,
                b: &$crate::domain::tensor::Tensor<T, Self>,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::add(a, b, dst)
            }

            fn add_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
                src: &$crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::add_inplace(dst, src)
            }

            fn ewise_mul<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                a: &$crate::domain::tensor::Tensor<T, Self>,
                b: &$crate::domain::tensor::Tensor<T, Self>,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::ewise_mul(a, b, dst)
            }

            fn scalar_mul_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                x: &mut $crate::domain::tensor::Tensor<T, Self>,
                scalar: f64,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::scalar_mul_inplace(x, scalar)
            }

            fn broadcast_mul_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                x: &mut $crate::domain::tensor::Tensor<T, Self>,
                scale: &$crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::broadcast_mul_inplace(x, scale)
            }

            fn broadcast_add_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                x: &mut $crate::domain::tensor::Tensor<T, Self>,
                bias: &$crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::broadcast_add_inplace(x, bias)
            }

            fn matmul<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                input: &$crate::domain::tensor::Tensor<T, Self>,
                weight: &$crate::domain::tensor::Tensor<T, Self>,
                output: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::matmul(input, weight, output)
            }

            fn matmul_quant<
                A: $crate::domain::dtype::Dtype,
                W: $crate::domain::dtype::Dtype,
                O: $crate::domain::dtype::Dtype,
            >(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                input: &$crate::domain::tensor::Tensor<A, Self>,
                weight: &$crate::domain::tensor::Tensor<W, Self>,
                output: &mut $crate::domain::tensor::Tensor<O, Self>,
                scales: &$crate::domain::tensor::Tensor<A, Self>,
                zeros: Option<&$crate::domain::tensor::Tensor<W, Self>>,
                scheme: &$crate::domain::dtype::quant::QuantScheme,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::matmul_quant(
                    input,
                    weight,
                    output,
                    scales,
                    zeros,
                    scheme.group,
                )
            }

            fn rmsnorm<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                input: &$crate::domain::tensor::Tensor<T, Self>,
                weight: &$crate::domain::tensor::Tensor<T, Self>,
                output: &mut $crate::domain::tensor::Tensor<T, Self>,
                eps: f32,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::rmsnorm(input, weight, output, eps)
            }

            fn rmsnorm_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                x: &mut $crate::domain::tensor::Tensor<T, Self>,
                weight: &$crate::domain::tensor::Tensor<T, Self>,
                eps: f32,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::rmsnorm_inplace(x, weight, eps)
            }

            fn silu_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                x: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::silu_inplace(x)
            }

            fn softmax<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                input: &$crate::domain::tensor::Tensor<T, Self>,
                output: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::softmax(input, output)
            }

            fn rope_inplace<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                q: &mut $crate::domain::tensor::Tensor<T, Self>,
                k: &mut $crate::domain::tensor::Tensor<T, Self>,
                sin: &$crate::domain::tensor::Tensor<T, Self>,
                cos: &$crate::domain::tensor::Tensor<T, Self>,
                positions: &$crate::domain::tensor::Tensor<i32, Self>,
                head_num: usize,
                kv_head_num: usize,
                head_dim: usize,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::rope_inplace(
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

            fn sdpa<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                q: &$crate::domain::tensor::Tensor<T, Self>,
                k: &$crate::domain::tensor::Tensor<T, Self>,
                v: &$crate::domain::tensor::Tensor<T, Self>,
                output: &mut $crate::domain::tensor::Tensor<T, Self>,
                mask: Option<&$crate::domain::tensor::Tensor<T, Self>>,
                num_heads: usize,
                num_kv_heads: usize,
                head_dim: usize,
                scale: f32,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                match mask {
                    Some(mask) => <Self as $crate::domain::ports::DiffusionOps>::sdpa_masked(
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
                    None => <Self as $crate::domain::ports::DiffusionOps>::sdpa(
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

            fn embedding<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                table: &$crate::domain::tensor::Tensor<T, Self>,
                indices: &$crate::domain::tensor::Tensor<i32, Self>,
                output: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::embedding(table, indices, output)
            }

            fn split_cols<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                src: &$crate::domain::tensor::Tensor<T, Self>,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
                rows: usize,
                total_cols: usize,
                col_offset: usize,
                dst_cols: usize,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::split_cols(
                    src, dst, rows, total_cols, col_offset, dst_cols,
                )
            }

            fn concat_seq<T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                a: &$crate::domain::tensor::Tensor<T, Self>,
                b: &$crate::domain::tensor::Tensor<T, Self>,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::concat_seq(a, b, dst)
            }

            fn cast<S: $crate::domain::dtype::Dtype, T: $crate::domain::dtype::Dtype>(
                scope: &<Self as $crate::domain::exec::Device>::Scope,
                src: &$crate::domain::tensor::Tensor<S, Self>,
                dst: &mut $crate::domain::tensor::Tensor<T, Self>,
            ) -> $crate::domain::ports::OpResult<()> {
                let _guard = $crate::domain::exec::ExecScope::enter(scope);
                <Self as $crate::domain::ports::CoreOps>::cast_dtype(src, dst)
            }
        }
    };
}
