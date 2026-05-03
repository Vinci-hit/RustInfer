//! `ModelLoader` 的权重装载扩展。
//!
//! 这里集中所有 LLM 加载 Matmul/RMSNorm/Embedding 的通用逻辑，
//! 避免每个新模型都在自己文件里重新抄一遍 `load_matmul` / `load_fused_qkv` / ...。
//!
//! 命名约定：
//! - `load_tensor_to_device` — 基础原语：只做"加载 + 可选转 F32 + 转 device"
//! - `load_matmul` / `load_rmsnorm` / `load_embedding` — 对应算子的便捷方法
//! - `load_fused_qkv` / `load_fused_gate_up` — 逐层融合加载（原始精度）
//! - `load_awq_matmul` / `load_fused_gate_up_awq` — AWQ 量化变体（K-packed）
//!
//! 详细背景见 `doc/REFACTOR_GUIDE.md` 阶段 3。

use crate::base::{DataType, DeviceType};
use crate::base::error::Result;
use crate::model::ModelLoader;
use crate::op::embedding::Embedding;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::tensor::Tensor;

/// Q/K/V 三个投影矩阵的维度集合。
///
/// 用 struct 而非 `(usize, usize, usize)` 位置参数避免调用方搞混（原则 3）。
#[derive(Debug, Clone, Copy)]
pub struct QkvDims {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub dim: usize,
}

/// Gate+Up 两个投影矩阵的维度集合。
#[derive(Debug, Clone, Copy)]
pub struct GateUpDims {
    pub intermediate_size: usize,
    pub dim: usize,
}

impl ModelLoader {
    /// 基础原语：按 tensor 名加载到 `device`；CPU 下自动把非 F32 升到 F32。
    ///
    /// 所有其它 `load_*` 便捷方法都构建在本方法之上。
    pub fn load_tensor_to_device(&self, name: &str, device: DeviceType) -> Result<Tensor> {
        let view = self.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&view)?;
        let weight = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };
        weight.to_device(device)
    }

    /// 直接加载 tensor view（零拷贝，不转 device）——仅在调用方需要
    /// 手动拼接多个 tensor 时使用。
    pub fn load_tensor_cpu(&self, name: &str) -> Result<Tensor> {
        let view = self.get_tensor(name)?;
        Tensor::from_view_on_cpu(&view)
    }

    /// 加载一个标准 Linear weight 为 [`Matmul`]。
    pub fn load_matmul(&self, name: &str, device: DeviceType) -> Result<Matmul> {
        Ok(Matmul::from(self.load_tensor_to_device(name, device)?, None))
    }

    /// 加载 RMSNorm weight（并附 eps）。
    pub fn load_rmsnorm(&self, name: &str, device: DeviceType, eps: f32) -> Result<RMSNorm> {
        Ok(RMSNorm::from(self.load_tensor_to_device(name, device)?, eps))
    }

    /// 加载 Embedding weight。
    pub fn load_embedding(&self, name: &str, device: DeviceType) -> Result<Embedding> {
        Ok(Embedding::from(self.load_tensor_to_device(name, device)?))
    }

    /// 加载第 `layer_idx` 层的 Q/K/V，并沿 **行方向**融合为 `[q_dim + 2*kv_dim, dim]` 的 Matmul。
    ///
    /// 对应权重名：`model.layers.{i}.self_attn.{q,k,v}_proj.weight`
    pub fn load_fused_qkv(
        &self,
        layer_idx: usize,
        dims: QkvDims,
        device: DeviceType,
    ) -> Result<Matmul> {
        let wq = self.load_tensor_cpu(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))?;
        let wk = self.load_tensor_cpu(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))?;
        let wv = self.load_tensor_cpu(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))?;

        let fused = fuse_rows(
            &[&wq, &wk, &wv],
            &[dims.q_dim, dims.kv_dim, dims.kv_dim],
            dims.dim,
            wq.dtype(),
        )?;
        let fused = if device.is_cpu() && fused.dtype() != DataType::F32 {
            fused.to_dtype(DataType::F32)?
        } else {
            fused
        };
        Ok(Matmul::from(fused.to_device(device)?, None))
    }

    /// 加载第 `layer_idx` 层的 gate_proj / up_proj，沿行方向融合为 `[2 * intermediate_size, dim]`。
    ///
    /// 对应权重名：`model.layers.{i}.mlp.{gate,up}_proj.weight`
    pub fn load_fused_gate_up(
        &self,
        layer_idx: usize,
        dims: GateUpDims,
        device: DeviceType,
    ) -> Result<Matmul> {
        let w1 = self.load_tensor_cpu(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))?;
        let w3 = self.load_tensor_cpu(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))?;

        let fused = fuse_rows(
            &[&w1, &w3],
            &[dims.intermediate_size, dims.intermediate_size],
            dims.dim,
            w1.dtype(),
        )?;
        let fused = if device.is_cpu() && fused.dtype() != DataType::F32 {
            fused.to_dtype(DataType::F32)?
        } else {
            fused
        };
        Ok(Matmul::from(fused.to_device(device)?, None))
    }

    /// 加载 AWQ 量化 Linear 层（K-packed 格式）。
    ///
    /// 期望 3 个张量：
    /// - `{name_prefix}.weight_packed`  — `[N, K/8]` I32
    /// - `{name_prefix}.weight_zero_point` — `[N/8, num_groups]` I32
    /// - `{name_prefix}.weight_scale` — `[N, num_groups]` BF16
    pub fn load_awq_matmul(
        &self,
        name_prefix: &str,
        device: DeviceType,
        group_size: usize,
    ) -> Result<Matmul> {
        let weight_packed = self.load_tensor_cpu(&format!("{}.weight_packed", name_prefix))?;
        let weight_zero_point = self.load_tensor_cpu(&format!("{}.weight_zero_point", name_prefix))?;
        let weight_scale = self.load_tensor_cpu(&format!("{}.weight_scale", name_prefix))?;

        Ok(Matmul::from_awq(
            weight_packed.to_device(device)?,
            weight_zero_point.to_device(device)?,
            weight_scale.to_device(device)?,
            group_size,
            None,
        ))
    }

    /// AWQ 版的 fused Gate+Up：按行拼接 `[gate_N, K/8] + [up_N, K/8]` 等 3 组张量。
    pub fn load_fused_gate_up_awq(
        &self,
        layer_idx: usize,
        intermediate_size: usize,
        device: DeviceType,
        group_size: usize,
    ) -> Result<Matmul> {
        let gate_wp = self.load_tensor_cpu(&format!("model.layers.{}.mlp.gate_proj.weight_packed", layer_idx))?;
        let up_wp = self.load_tensor_cpu(&format!("model.layers.{}.mlp.up_proj.weight_packed", layer_idx))?;

        let gate_sc = self.load_tensor_cpu(&format!("model.layers.{}.mlp.gate_proj.weight_scale", layer_idx))?;
        let up_sc = self.load_tensor_cpu(&format!("model.layers.{}.mlp.up_proj.weight_scale", layer_idx))?;

        let gate_zp = self.load_tensor_cpu(&format!("model.layers.{}.mlp.gate_proj.weight_zero_point", layer_idx))?;
        let up_zp = self.load_tensor_cpu(&format!("model.layers.{}.mlp.up_proj.weight_zero_point", layer_idx))?;

        let k_packed = gate_wp.shape()[1]; // K/8, same for both
        let num_groups = gate_sc.shape()[1]; // num_groups, same for both

        // weight_packed: [gate_N, K/8] + [up_N, K/8] -> [2*inter, K/8] (row concat)
        let fused_wp = fuse_rows(
            &[&gate_wp, &up_wp],
            &[intermediate_size, intermediate_size],
            k_packed,
            DataType::I32,
        )?;

        // weight_scale: [gate_N, G] + [up_N, G] -> [2*inter, G] (row concat)
        let sc_dtype = gate_sc.dtype();
        let fused_sc = fuse_rows(
            &[&gate_sc, &up_sc],
            &[intermediate_size, intermediate_size],
            num_groups,
            sc_dtype,
        )?;

        // weight_zero_point: [gate_N/8, G] + [up_N/8, G] -> [2*inter/8, G] (row concat)
        let gate_n_packed = intermediate_size / 8;
        let up_n_packed = intermediate_size / 8;
        let fused_zp = fuse_rows(
            &[&gate_zp, &up_zp],
            &[gate_n_packed, up_n_packed],
            num_groups,
            DataType::I32,
        )?;

        Ok(Matmul::from_awq(
            fused_wp.to_device(device)?,
            fused_zp.to_device(device)?,
            fused_sc.to_device(device)?,
            group_size,
            None,
        ))
    }
}

/// 把多个 `[rows_i, cols]` 的 CPU 张量沿行方向拼接为 `[sum(rows_i), cols]`。
///
/// 所有输入 tensor 必须已在 CPU 上、dtype 必须一致（由调用方保证）。
fn fuse_rows(
    tensors: &[&Tensor],
    row_counts: &[usize],
    cols: usize,
    dtype: DataType,
) -> Result<Tensor> {
    debug_assert_eq!(tensors.len(), row_counts.len());
    let total_rows: usize = row_counts.iter().sum();
    let elem_size = dtype.size_in_bytes();
    let mut fused = Tensor::new(&[total_rows, cols], dtype, DeviceType::Cpu)?;
    let fused_ptr = fused.buffer_mut().as_mut_ptr();
    let mut offset = 0usize;
    for (tensor, &rows) in tensors.iter().zip(row_counts) {
        let bytes = rows * cols * elem_size;
        // SAFETY: fused 尺寸 = sum(row_counts) * cols * elem_size, offset + bytes 不会越界；
        //         tensor 数据来自 safetensors mmap 的 CPU 视图, 与 fused 不重叠。
        unsafe {
            std::ptr::copy_nonoverlapping(tensor.buffer().as_ptr(), fused_ptr.add(offset), bytes);
        }
        offset += bytes;
    }
    Ok(fused)
}
