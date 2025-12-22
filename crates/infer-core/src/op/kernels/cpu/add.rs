use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::bf16;
use ndarray::ArrayViewD; // 使用动态维度的视图 `ArrayViewD`
use ndarray::ArrayViewMutD;

/// Add 的 CPU 内核实现: C = A + B (按元素)
pub fn add(input_a: &Tensor, input_b: &Tensor, output: &mut Tensor) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match (input_a.dtype(), input_b.dtype(), output.dtype()) {
        (crate::base::DataType::F32, crate::base::DataType::F32, crate::base::DataType::F32) => {
            add_f32(input_a, input_b, output)
        }
        (crate::base::DataType::BF16, crate::base::DataType::BF16, crate::base::DataType::BF16) => {
            add_bf16(input_a, input_b, output)
        }
        _ => {
            // 如果输入和输出的数据类型不匹配，则返回错误
            Err(Error::InvalidArgument(format!(
                "Unsupported data type combination for add: input_a={:?}, input_b={:?}, output={:?}", 
                input_a.dtype(), input_b.dtype(), output.dtype()
            )).into())
        }
    }
}

/// F32版本的Add实现: C = A + B (按元素)
fn add_f32(input_a: &Tensor, input_b: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取类型化的张量和数据 slice ---
    let a_typed = input_a.as_f32()?;
    let b_typed = input_b.as_f32()?;
    let output_shape = output.shape().to_vec();
    let c_typed = output.as_f32_mut()?;

    let a_slice = a_typed.as_slice()?;
    let b_slice = b_typed.as_slice()?;
    let c_slice = c_typed.as_slice_mut()?;
    
    // --- 2. 创建 ndarray 视图 ---
    // 使用 ArrayViewD (D for Dynamic) 来处理任意维度的张量
    let a_view: ArrayViewD<f32> = ArrayViewD::from_shape(input_a.shape(), a_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        
    let b_view: ArrayViewD<f32> = ArrayViewD::from_shape(input_b.shape(), b_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        
    let mut c_view: ArrayViewMutD<f32> = ArrayViewMutD::from_shape(output_shape, c_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // --- 3. 执行计算 ---
    // ndarray 重载了 `+` 运算符用于按元素相加
    // `&a_view + &b_view` 会创建一个新的临时 Array，
    // 然后 .assign() 将其结果高效地复制到输出视图 c_view 中。
    c_view.assign(&(&a_view + &b_view));

    Ok(())
}

/// BF16版本的Add实现: C = A + B (按元素)
fn add_bf16(input_a: &Tensor, input_b: &Tensor, output: &mut Tensor) -> Result<()> {
    // --- 1. 获取类型化的张量和数据 slice ---
    let a_typed = input_a.as_bf16()?;
    let b_typed = input_b.as_bf16()?;
    let output_shape = output.shape().to_vec();
    let c_typed = output.as_bf16_mut()?;

    let a_slice = a_typed.as_slice()?;
    let b_slice = b_typed.as_slice()?;
    let c_slice = c_typed.as_slice_mut()?;
    
    // --- 2. 创建 ndarray 视图 ---
    // 使用 ArrayViewD (D for Dynamic) 来处理任意维度的张量
    let a_view: ArrayViewD<bf16> = ArrayViewD::from_shape(input_a.shape(), a_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        
    let b_view: ArrayViewD<bf16> = ArrayViewD::from_shape(input_b.shape(), b_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        
    let mut c_view: ArrayViewMutD<bf16> = ArrayViewMutD::from_shape(output_shape, c_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // --- 3. 执行计算 ---
    // ndarray 重载了 `+` 运算符用于按元素相加
    // `&a_view + &b_view` 会创建一个新的临时 Array，
    // 然后 .assign() 将其结果高效地复制到输出视图 c_view 中。
    c_view.assign(&(&a_view + &b_view));

    Ok(())
}


/// Add 的原地 (in-place) CPU 内核实现: A = A + B
/// 这个函数对应我们之前为 Matmul 实现的 bias add
pub fn add_inplace(tensor_a: &mut Tensor, tensor_b: &Tensor) -> Result<()> {
    // 根据数据类型自动分发到对应的实现
    match (tensor_a.dtype(), tensor_b.dtype()) {
        (crate::base::DataType::F32, crate::base::DataType::F32) => {
            add_inplace_f32(tensor_a, tensor_b)
        }
        (crate::base::DataType::BF16, crate::base::DataType::BF16) => {
            add_inplace_bf16(tensor_a, tensor_b)
        }
        _ => {
            // 如果输入的数据类型不匹配，则返回错误
            Err(Error::InvalidArgument(format!(
                "Unsupported data type combination for add_inplace: tensor_a={:?}, tensor_b={:?}", 
                tensor_a.dtype(), tensor_b.dtype()
            )).into())
        }
    }
}

/// F32版本的Add原地实现: A = A + B
fn add_inplace_f32(tensor_a: &mut Tensor, tensor_b: &Tensor) -> Result<()> {
    let a_typed = tensor_a.as_f32_mut()?;
    let b_typed = tensor_b.as_f32()?;
    let a_shape = a_typed.shape().to_vec();
    let b_shape = b_typed.shape().to_vec();
    let a_slice = a_typed.as_slice_mut()?;
    let b_slice = b_typed.as_slice()?;

    // --- 创建 ndarray 视图 ---
    let mut a_view = ArrayViewMutD::from_shape(a_shape, a_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
    let b_view = ArrayViewD::from_shape(b_shape, b_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // --- 执行原地加法 ---
    // `a_view += &b_view` 会执行按元素原地加法，支持广播
    a_view += &b_view;

    Ok(())
}

/// BF16版本的Add原地实现: A = A + B
fn add_inplace_bf16(tensor_a: &mut Tensor, tensor_b: &Tensor) -> Result<()> {
    let a_typed = tensor_a.as_bf16_mut()?;
    let b_typed = tensor_b.as_bf16()?;
    let a_shape = a_typed.shape().to_vec();
    let b_shape = b_typed.shape().to_vec();
    let a_slice = a_typed.as_slice_mut()?;
    let b_slice = b_typed.as_slice()?;

    // --- 创建 ndarray 视图 ---
    let mut a_view = ArrayViewMutD::from_shape(a_shape, a_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;
    let b_view = ArrayViewD::from_shape(b_shape, b_slice)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // --- 执行原地加法 ---
    // `a_view += &b_view` 会执行按元素原地加法，支持广播
    a_view += &b_view;

    Ok(())
}