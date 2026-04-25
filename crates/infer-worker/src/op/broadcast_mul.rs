use crate::base::DeviceType;
use crate::base::error::Result;
use crate::tensor::Tensor;

use super::kernels;

/// 广播逐元素乘法: dst[i, j] = a[i, j] * b[j]
/// a: [rows, D], b: [D], dst: [rows, D]
pub fn broadcast_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    match a.device() {
        DeviceType::Cpu => kernels::cpu::broadcast_mul(a, b, dst),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let d = *a.shape().last().unwrap() as i32;
            let rows = (a.num_elements() / d as usize) as i32;
            kernels::cuda::broadcast_mul(a, b, dst, rows, d, crate::cuda::get_current_cuda_stream())
        }
    }
}
