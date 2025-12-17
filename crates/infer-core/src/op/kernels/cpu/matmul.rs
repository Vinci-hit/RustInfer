use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use ndarray::{ArrayView2, ArrayViewMut2};

/// Matmul (GEMM) 的 CPU 内核实现: C = A * B^T
///
/// # Arguments
/// * `input` (A): 输入张量, 形状 [..., M, K]
/// * `weight` (B): 权重张量, 形状 [N, K]
/// * `output` (C): 输出张量, 形状 [..., M, N]
pub fn matmul(input: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取类型化的张量和数据 slice ---

    let input_typed = input.as_f32()?;
    let weight_typed = weight.as_f32()?;
    let output_typed = output.as_f32_mut()?;

    let input_slice = input_typed.as_slice()?;
    let weight_slice = weight_typed.as_slice()?;
    let output_slice = output_typed.as_slice_mut()?;
    
    // --- 2. 形状检查和 ndarray 视图创建 ---
    let weight_shape = weight.shape();
    let out_features = weight_shape[0];
    let in_features = weight_shape[1];
    
    
    // 将权重 W [N, K] 转换为 ndarray 视图
    let weight_view: ArrayView2<f32> = ArrayView2::from_shape(
        (out_features, in_features),
        weight_slice
    ).map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // 输入可能是多维的 (e.g., [B, S, K])，我们需要处理 batch 维度
    // 我们将输入 reshape 成一个 2D 矩阵 [M, K]，其中 M 是所有前缀维度的乘积
    let input_shape = input.shape();
    let m_dim = input_shape[..input_shape.len() - 1].iter().product();
    
    let input_view: ArrayView2<f32> = ArrayView2::from_shape(
        (m_dim, in_features),
        input_slice
    ).map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // 创建输出的 2D 可变视图 [M, N]
    let mut output_view: ArrayViewMut2<f32> = ArrayViewMut2::from_shape(
        (m_dim, out_features),
        output_slice
    ).map_err(|e| Error::InvalidArgument(e.to_string()))?;
    
    // --- 3. 执行高性能矩阵乘法 ---
    // 计算 C = A * B^T。 ndarray 的 .dot() 会自动调用 BLAS GEMM。
    // weight_view.t() 是权重的转置 (transpose)，这是一个零拷贝操作。
    // println!("Performing matmul with shapes: input: {:?}, weight: {:?}, output: {:?}", input_view.dim(), weight_view.dim(), output_view.dim());
    output_view.assign(&input_view.dot(&weight_view.t()));
    Ok(())
}