use std::io::{self, Write};
use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::op::add_inplace::AddInplace;
use std::time::Instant;

use crate::model::common::config::RuntimeModelConfig;
use crate::model::ModelLoader;
use crate::tensor::Tensor;
use crate::model::common::tokenizer::Tokenizer;
use crate::base::error::Error::InternalError;
use std::boxed::Box;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::swiglu::SwiGLU;
use crate::op::scatter::Scatter;
use crate::model::runtime::InferenceState;


/// LlamaLayers holds all operators and weights for the model.
pub struct LlamaLayers {
    pub embedding_layer: Embedding,
    pub rmsnorm_final_layer: RMSNorm,
    pub cls_layer: Matmul,

    pub rmsnorm_attn_layers: Vec<RMSNorm>,
    pub rmsnorm_ffn_layers: Vec<RMSNorm>,
    pub wqkv_layers: Vec<Matmul>,
    pub wo_layers: Vec<Matmul>,
    pub mha_layers: Vec<FlashAttnGQA>,
    pub rope_layers: Vec<RoPEOp>,
    pub add_layers: AddInplace,
    pub scatter_layer: Scatter,

    pub w_gate_up_layers: Vec<Matmul>,
    pub w2_layers: Vec<Matmul>,
    pub swiglu_layers: Vec<SwiGLU>,
}

impl LlamaLayers {
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.embedding_layer.to_cuda(device_id)?;
        self.rmsnorm_final_layer.to_cuda(device_id)?;
        self.cls_layer.to_cuda(device_id)?;
        self.rmsnorm_attn_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.rmsnorm_ffn_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.wqkv_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.wo_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.w_gate_up_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.w2_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.mha_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.rope_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.add_layers.to_cuda(device_id)?;
        self.swiglu_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        Ok(())
    }
}

/// Llama3 model — holds only static weights and configuration.
/// Request-level mutable state lives in `InferenceState`.
pub struct Llama3 {
    pub(crate) config: RuntimeModelConfig,
    pub(crate) device_type: DeviceType,
    pub(crate) tokenizer: Box<dyn Tokenizer>,
    pub(crate) layers: LlamaLayers,
}

impl Llama3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
    ) -> Result<Self> {
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tensor_names: std::collections::HashSet<String> = loader.tensor_names().into_iter().collect();
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();

        let layer_num = config.layer_num;
        let mut rmsnorm_attn_layers = Vec::with_capacity(layer_num);
        let mut rmsnorm_ffn_layers = Vec::with_capacity(layer_num);
        let mut wqkv_layers = Vec::with_capacity(layer_num);
        let mut wo_layers = Vec::with_capacity(layer_num);
        let mut w_gate_up_layers = Vec::with_capacity(layer_num);
        let mut w2_layers = Vec::with_capacity(layer_num);

        let is_awq = config.quant_config.as_ref().is_some_and(|q|
            q.quant_method == "compressed-tensors");
        let group_size = config.quant_config.as_ref().map(|q| q.group_size).unwrap_or(128);

        for i in 0..layer_num {
            if is_awq {
                // AWQ 量化模型: 仅 MLP 层量化，attention 保持原精度
                wqkv_layers.push(Self::load_fused_qkv(i, &loader, device_type, config.q_dim, config.kv_dim, config.dim)?);
                wo_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.o_proj.weight", i), &loader, device_type)?);
                w_gate_up_layers.push(Self::load_fused_gate_up_awq(i, &loader, device_type, config.intermediate_size, group_size)?);
                w2_layers.push(Self::load_awq_matmul(&format!("model.layers.{}.mlp.down_proj", i), &loader, device_type, group_size)?);
            } else {
                // 原始精度模型
                wqkv_layers.push(Self::load_fused_qkv(i, &loader, device_type, config.q_dim, config.kv_dim, config.dim)?);
                wo_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.o_proj.weight", i), &loader, device_type)?);
                w_gate_up_layers.push(Self::load_fused_gate_up(i, &loader, device_type, config.intermediate_size, config.dim)?);
                w2_layers.push(Self::load_matmul(&format!("model.layers.{}.mlp.down_proj.weight", i), &loader, device_type)?);
            }
            rmsnorm_attn_layers.push(Self::load_rmsnorm(&format!("model.layers.{}.input_layernorm.weight", i), &loader, device_type, config.rms_norm_eps)?);
            rmsnorm_ffn_layers.push(Self::load_rmsnorm(&format!("model.layers.{}.post_attention_layernorm.weight", i), &loader, device_type, config.rms_norm_eps)?);
        }

        let embedding_layer = Self::load_embedding("model.embed_tokens.weight", &loader, device_type)?;
        let rmsnorm_final_layer = Self::load_rmsnorm("model.norm.weight", &loader, device_type, config.rms_norm_eps)?;
        let cls_layer = if tensor_names.contains("lm_head.weight") {
            Self::load_matmul("lm_head.weight", &loader, device_type)?
        } else {
            Matmul::from(embedding_layer.weight.clone(), None)
        };

        let layer_num = config.layer_num;
        let mha_layers: Result<Vec<FlashAttnGQA>> = (0..layer_num)
            .map(|_| FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size, true))
            .collect();
        let mha_layers = mha_layers?;
        let rope_layers: Result<Vec<RoPEOp>> = (0..layer_num)
            .map(|_| RoPEOp::new(config.dim, config.kv_dim, config.head_size))
            .collect();
        let rope_layers = rope_layers?;
        let add_layers = AddInplace::new();
        let swiglu_layers: Vec<SwiGLU> = (0..layer_num).map(|_| SwiGLU::new()).collect();

        if rmsnorm_attn_layers.len() != layer_num || rmsnorm_ffn_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of RMSNorm layers.".to_string()).into());
        }
        if wqkv_layers.len() != layer_num || wo_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of attention Matmul layers.".to_string()).into());
        }
        if w_gate_up_layers.len() != layer_num || w2_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of FFN Matmul layers.".to_string()).into());
        }
        if mha_layers.len() != layer_num || rope_layers.len() != layer_num || swiglu_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of non-parameterized layers.".to_string()).into());
        }

        let layers = LlamaLayers {
            embedding_layer, rmsnorm_final_layer, cls_layer,
            rmsnorm_attn_layers, rmsnorm_ffn_layers,
            wqkv_layers, wo_layers, mha_layers, rope_layers,
            add_layers, scatter_layer: Scatter::new(),
            w_gate_up_layers, w2_layers, swiglu_layers,
        };

        Ok(Self { config, device_type, tokenizer, layers })
    }

    /// Create a new InferenceState for this model.
    pub fn create_state(&self) -> Result<InferenceState> {
        InferenceState::new(&self.config, self.device_type)
    }

    /// 获取 tokenizer 引用
    pub fn tokenizer(&self) -> &dyn crate::model::common::tokenizer::Tokenizer {
        self.tokenizer.as_ref()
    }

    /// 获取模型配置引用
    pub fn config(&self) -> &RuntimeModelConfig {
        &self.config
    }

    // ---- Weight loading helpers ----

    fn load_matmul(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Matmul> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;
        let weight = if device.is_cpu() && weight.dtype() != DataType::F32 { weight.to_dtype(DataType::F32)? } else { weight };
        Ok(Matmul::from(weight.to_device(device)?, None))
    }

    fn load_fused_qkv(
        layer_idx: usize, loader: &ModelLoader, device: DeviceType,
        q_dim: usize, kv_dim: usize, dim: usize,
    ) -> Result<Matmul> {
        let wq = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))?)?;
        let wk = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))?)?;
        let wv = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))?)?;

        let dtype = wq.dtype();
        let fused_rows = q_dim + 2 * kv_dim;
        let mut fused = Tensor::new(&[fused_rows, dim], dtype, DeviceType::Cpu)?;
        let elem_size = dtype.size_in_bytes();
        let (wq_bytes, wk_bytes, wv_bytes) = (q_dim * dim * elem_size, kv_dim * dim * elem_size, kv_dim * dim * elem_size);
        let fused_ptr = fused.buffer_mut().as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(wq.buffer().as_ptr(), fused_ptr, wq_bytes);
            std::ptr::copy_nonoverlapping(wk.buffer().as_ptr(), fused_ptr.add(wq_bytes), wk_bytes);
            std::ptr::copy_nonoverlapping(wv.buffer().as_ptr(), fused_ptr.add(wq_bytes + wk_bytes), wv_bytes);
        }
        let fused = if device.is_cpu() && dtype != DataType::F32 { fused.to_dtype(DataType::F32)? } else { fused };
        Ok(Matmul::from(fused.to_device(device)?, None))
    }

    fn load_fused_gate_up(
        layer_idx: usize, loader: &ModelLoader, device: DeviceType,
        intermediate_size: usize, dim: usize,
    ) -> Result<Matmul> {
        let w1 = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))?)?;
        let w3 = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))?)?;

        let dtype = w1.dtype();
        let fused_rows = 2 * intermediate_size;
        let mut fused = Tensor::new(&[fused_rows, dim], dtype, DeviceType::Cpu)?;
        let elem_size = dtype.size_in_bytes();
        let (w1_bytes, w3_bytes) = (intermediate_size * dim * elem_size, intermediate_size * dim * elem_size);
        let fused_ptr = fused.buffer_mut().as_mut_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(w1.buffer().as_ptr(), fused_ptr, w1_bytes);
            std::ptr::copy_nonoverlapping(w3.buffer().as_ptr(), fused_ptr.add(w1_bytes), w3_bytes);
        }
        let fused = if device.is_cpu() && dtype != DataType::F32 { fused.to_dtype(DataType::F32)? } else { fused };
        Ok(Matmul::from(fused.to_device(device)?, None))
    }

    fn load_rmsnorm(name: &str, loader: &ModelLoader, device: DeviceType, eps: f32) -> Result<RMSNorm> {
        let weight = Tensor::from_view_on_cpu(&loader.get_tensor(name)?)?;
        let weight = if device.is_cpu() && weight.dtype() != DataType::F32 { weight.to_dtype(DataType::F32)? } else { weight };
        Ok(RMSNorm::from(weight.to_device(device)?, eps))
    }

    fn load_embedding(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Embedding> {
        let weight = Tensor::from_view_on_cpu(&loader.get_tensor(name)?)?;
        let weight = if device.is_cpu() && weight.dtype() != DataType::F32 { weight.to_dtype(DataType::F32)? } else { weight };
        Ok(Embedding::from(weight.to_device(device)?))
    }

    // ---- AWQ weight loading helpers (K-packed format) ----

    /// 辅助: 将多个 [rows_i, cols] 的张量纵向拼接为 [sum(rows_i), cols]
    fn fuse_tensors_vertically(tensors: &[&Tensor], row_counts: &[usize], cols: usize, dtype: DataType) -> Result<Tensor> {
        let total_rows: usize = row_counts.iter().sum();
        let elem_size = dtype.size_in_bytes();
        let mut fused = Tensor::new(&[total_rows, cols], dtype, DeviceType::Cpu)?;
        let fused_ptr = fused.buffer_mut().as_mut_ptr();
        let mut offset = 0usize;
        for (tensor, &rows) in tensors.iter().zip(row_counts) {
            let bytes = rows * cols * elem_size;
            unsafe {
                std::ptr::copy_nonoverlapping(tensor.buffer().as_ptr(), fused_ptr.add(offset), bytes);
            }
            offset += bytes;
        }
        Ok(fused)
    }

    /// 加载单个 AWQ 量化 Linear 层
    /// weight_packed: [N, K/8] (I32) — 直接加载，无需转置
    /// weight_zero_point: [N/8, num_groups] (I32)
    /// weight_scale: [N, num_groups] (BF16)
    fn load_awq_matmul(name_prefix: &str, loader: &ModelLoader, device: DeviceType, group_size: usize) -> Result<Matmul> {
        let weight_packed = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("{}.weight_packed", name_prefix))?)?;
        let weight_zero_point = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("{}.weight_zero_point", name_prefix))?)?;
        let weight_scale = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("{}.weight_scale", name_prefix))?)?;

        Ok(Matmul::from_awq(
            weight_packed.to_device(device)?,
            weight_zero_point.to_device(device)?,
            weight_scale.to_device(device)?,
            group_size,
            None,
        ))
    }

    /// AWQ fused Gate+Up 加载
    /// weight_packed 按行(N)拼接: [gate_N, K/8] + [up_N, K/8] -> [2*inter, K/8]
    /// weight_scale 按行拼接: [gate_N, G] + [up_N, G] -> [2*inter, G]
    /// weight_zero_point 按行拼接: [gate_N/8, G] + [up_N/8, G] -> [2*inter/8, G]
    fn load_fused_gate_up_awq(
        layer_idx: usize, loader: &ModelLoader, device: DeviceType,
        intermediate_size: usize, group_size: usize,
    ) -> Result<Matmul> {
        let gate_wp = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight_packed", layer_idx))?)?;
        let up_wp = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight_packed", layer_idx))?)?;

        let gate_sc = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight_scale", layer_idx))?)?;
        let up_sc = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight_scale", layer_idx))?)?;

        let gate_zp = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight_zero_point", layer_idx))?)?;
        let up_zp = Tensor::from_view_on_cpu(&loader.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight_zero_point", layer_idx))?)?;

        let k_packed = gate_wp.shape()[1]; // K/8, same for both
        let num_groups = gate_sc.shape()[1]; // num_groups, same for both

        // weight_packed: [gate_N, K/8] + [up_N, K/8] -> [2*inter, K/8] (row concat)
        let fused_wp = Self::fuse_tensors_vertically(
            &[&gate_wp, &up_wp],
            &[intermediate_size, intermediate_size],
            k_packed, DataType::I32,
        )?;

        // weight_scale: [gate_N, G] + [up_N, G] -> [2*inter, G] (row concat)
        let sc_dtype = gate_sc.dtype();
        let fused_sc = Self::fuse_tensors_vertically(
            &[&gate_sc, &up_sc],
            &[intermediate_size, intermediate_size],
            num_groups, sc_dtype,
        )?;

        // weight_zero_point: [gate_N/8, G] + [up_N/8, G] -> [2*inter/8, G] (row concat)
        let gate_n_packed = intermediate_size / 8;
        let up_n_packed = intermediate_size / 8;
        let fused_zp = Self::fuse_tensors_vertically(
            &[&gate_zp, &up_zp],
            &[gate_n_packed, up_n_packed],
            num_groups, DataType::I32,
        )?;

        Ok(Matmul::from_awq(
            fused_wp.to_device(device)?,
            fused_zp.to_device(device)?,
            fused_sc.to_device(device)?,
            group_size,
            None,
        ))
    }

    // ---- Inference methods (&self + &mut InferenceState) ----

    pub fn generate(
        &self,
        state: &mut InferenceState,
        prompt: &str,
        max_tokens: usize,
        print_output: bool,
    ) -> Result<(String, u32, u64, u64, usize)> {
        let mut stdout = io::stdout();
        if print_output {
            println!("----------------------------------------");
            println!("Prompt: {}", prompt);
            stdout.flush()?;
        }

        let prompt_tokens = self.tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidArgument("Prompt cannot be empty.".to_string()).into());
        }

        let mut workspace = crate::worker::BatchWorkspace::new(
            self.config(),
            prompt_tokens.len().max(1),
            1,
            self.device_type,
        )?;
        crate::model::runtime::compute_rope_cache(
            self.config(),
            &mut workspace.sin_cache,
            &mut workspace.cos_cache,
        )?;

        #[cfg(feature = "cuda")]
        let cuda_cfg = if self.device_type.is_cuda() {
            Some(crate::cuda::CudaConfig::new()?.with_flash_decode(
                self.config.head_num,
                self.config.head_size,
                1,
            )?)
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let cuda_ref = cuda_cfg.as_ref().map(|c| c as &crate::OpConfig);
        #[cfg(not(feature = "cuda"))]
        let cuda_ref = None;

        let prefill_start = Instant::now();
        let prefill_positions: Vec<i32> = (0..prompt_tokens.len()).map(|i| i as i32).collect();
        let prefill_q_start = [0, prompt_tokens.len() as i32];
        let slot_indices = [0];
        let prefill_meta = crate::worker::runner::WorkerBatchMeta {
            q_start_loc: &prefill_q_start,
            slot_indices: &slot_indices,
            token_ids: &prompt_tokens,
            positions: &prefill_positions,
            num_decode: 0,
            num_prefill: 1,
        };
        let mut gen_output = Tensor::new(&[1], DataType::I32, self.device_type)?;
        let first_token = {
            workspace.input_tokens.write_from_i32_host(&prompt_tokens, prompt_tokens.len())?;
            workspace.input_pos.write_from_i32_host(&prefill_positions, prefill_positions.len())?;
            workspace.kv_lens_cpu.as_i32_mut()?.as_slice_mut()?[0] = 0;
            #[cfg(feature = "cuda")]
            {
                let src = workspace.kv_lens_cpu.slice(&[0], &[1])?;
                let mut dst = workspace.kv_lens_dev.slice(&[0], &[1])?;
                dst.copy_from(&src)?;
            }
            let mut refs = vec![&mut *state];
            self.forward(refs.as_mut_slice(), &mut workspace, &prefill_meta, &mut gen_output, cuda_ref)?;
            gen_output.to_cpu()?.as_i32()?.as_slice()?[0]
        };
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let mut generated_tokens = vec![first_token];
        let mut printed_len = 0usize;
        if print_output {
            let decoded = self.tokenizer.decode(&generated_tokens)?;
            let _ = write!(stdout, "{}", &decoded[printed_len..]);
            printed_len = decoded.len();
            stdout.flush()?;
        }

        let decode_start = Instant::now();
        let mut decode_iterations = 0;
        let max_decode_end = (prompt_tokens.len() - 1 + max_tokens).min(self.config.seq_len);
        for pos in prompt_tokens.len()..max_decode_end {
            let token_ids = [0i32];
            let positions = [pos as i32];
            let q_start = [0, 1];
            let meta = crate::worker::runner::WorkerBatchMeta {
                q_start_loc: &q_start,
                slot_indices: &slot_indices,
                token_ids: &token_ids,
                positions: &positions,
                num_decode: 1,
                num_prefill: 0,
            };
            let next_token = {
                workspace.input_tokens.write_from_i32_host(&[generated_tokens[generated_tokens.len() - 1]], 1)?;
                workspace.input_pos.write_from_i32_host(&positions, 1)?;
                workspace.kv_lens_cpu.as_i32_mut()?.as_slice_mut()?[0] = positions[0];
                #[cfg(feature = "cuda")]
                {
                    let src = workspace.kv_lens_cpu.slice(&[0], &[1])?;
                    let mut dst = workspace.kv_lens_dev.slice(&[0], &[1])?;
                    dst.copy_from(&src)?;
                }
                let mut refs = vec![&mut *state];
                self.forward(refs.as_mut_slice(), &mut workspace, &meta, &mut gen_output, cuda_ref)?;
                gen_output.to_cpu()?.as_i32()?.as_slice()?[0]
            };

            if self.tokenizer.is_eos(next_token) { break; }

            generated_tokens.push(next_token);
            decode_iterations += 1;

            if print_output {
                let decoded = self.tokenizer.decode(&generated_tokens)?;
                if decoded.len() > printed_len {
                    let new_text = &decoded[printed_len..];
                    if !new_text.contains('\u{FFFD}') {
                        let _ = write!(stdout, "{}", new_text);
                        printed_len = decoded.len();
                        stdout.flush()?;
                    }
                }
            }
        }
        let decode_ms = decode_start.elapsed().as_millis() as u64;
        if print_output { println!(); }

        let generated_text = self.tokenizer.decode(&generated_tokens)?;
        Ok((generated_text, generated_tokens.len() as u32, prefill_ms, decode_ms, decode_iterations))
    }

    pub fn forward(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        batch: &crate::worker::runner::WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        let num_seqs = batch.num_seqs();
        if states.len() != num_seqs {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward states len {} != batch seqs {}",
                states.len(), num_seqs
            )).into());
        }
        if num_seqs == 0 {
            return Ok(());
        }
        let total_tokens = batch.seq_end(num_seqs - 1);
        if total_tokens > workspace.max_batch_tokens {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward total tokens {} exceeds workspace capacity {}",
                total_tokens, workspace.max_batch_tokens
            )).into());
        }
        if num_seqs > workspace.max_batch_seqs {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward seqs {} exceeds workspace capacity {}",
                num_seqs, workspace.max_batch_seqs
            )).into());
        }

        let mut kv_grew = false;
        for i in 0..num_seqs {
            if states[i].kv_cache.ensure_capacity(batch.seq_end_pos(i)?)? {
                states[i].invalidate_decode_graphs();
                kv_grew = true;
            }
        }
        if kv_grew {
            workspace.invalidate_batch_member_cache();
            #[cfg(feature = "cuda")]
            if let Some(cfg) = cuda_config {
                let cfg_ptr = cfg as *const crate::cuda::CudaConfig as *mut crate::cuda::CudaConfig;
                unsafe { (*cfg_ptr).graphs.clear(); }
            }
        }

        let can_full_graph = false
            && batch.is_decode_only()
            && self.device_type.is_cuda()
            && workspace.x.dtype() == DataType::BF16
            && self.config.head_size == 64;

        #[cfg(feature = "cuda")]
        if can_full_graph {
            let cfg = cuda_config.ok_or_else(|| Error::InvalidArgument(
                "DecodeOnly FullGraph path requires CudaConfig".into()
            ))?;
            let output_ptr = output_tokens.as_i32()?.buffer().as_ptr() as usize;
            let slot = crate::cuda::GraphSlot::LlmDecodeWithOutput { batch: num_seqs, output_ptr };
            if !cfg.graph_ready(slot) {
                cfg.sync_stream()?;
                let cfg_ptr = cfg as *const crate::cuda::CudaConfig as *mut crate::cuda::CudaConfig;
                cfg.capture_begin()?;
                self.compute_worker_batch_on_stream(states, workspace, batch, output_tokens, cuda_config)?;
                unsafe { (*cfg_ptr).capture_end(slot)?; }
            }
            cfg.launch(slot)?;
            cfg.sync_stream()?;
        } else {
            self.compute_worker_batch_on_stream(states, workspace, batch, output_tokens, cuda_config)?;
        }
        Ok(())
    }

    fn compute_worker_batch_on_stream(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        batch: &crate::worker::runner::WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config_ref: Option<&crate::OpConfig>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(cfg) = cuda_config_ref {
            return crate::cuda::with_cuda_stream(cfg.stream, || {
                self.compute_worker_batch(states, workspace, batch, output_tokens, cuda_config_ref)
            });
        }
        self.compute_worker_batch(states, workspace, batch, output_tokens, cuda_config_ref)
    }

    fn compute_worker_batch(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        batch: &crate::worker::runner::WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config_ref: Option<&crate::OpConfig>,
    ) -> Result<()> {
        let total_tokens = batch.seq_end(batch.num_seqs() - 1);
        let num_seqs = batch.num_seqs();
        let dim = self.config.dim;
        let q_dim = self.config.q_dim;
        let kv_dim = self.config.kv_dim;
        let inter = self.config.intermediate_size;
        let qkv_cols = q_dim + 2 * kv_dim;

        let input_tokens_view = workspace.input_tokens.slice(&[0], &[total_tokens])?;
        let input_pos_view = workspace.input_pos.slice(&[0], &[total_tokens])?;
        let mut x = workspace.x.slice(&[0, 0], &[total_tokens, dim])?;
        self.layers.embedding_layer.forward(&input_tokens_view, &mut x, cuda_config_ref)?;

        #[cfg(feature = "cuda")]
        let split_stream = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);

        for layer_idx in 0..self.config.layer_num {
            let mut attn_norm_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
            if layer_idx == 0 || !self.device_type.is_cuda() {
                self.layers.rmsnorm_attn_layers[layer_idx].forward(&x, &mut attn_norm_out, cuda_config_ref)?;
            }

            let mut qkv = workspace.qkv_out.slice(&[0, 0], &[total_tokens, qkv_cols])?;
            self.layers.wqkv_layers[layer_idx].forward(&attn_norm_out, &mut qkv, cuda_config_ref)?;

            let mut q = workspace.q_out.slice(&[0, 0], &[total_tokens, q_dim])?;
            let mut k = workspace.k_out.slice(&[0, 0], &[total_tokens, kv_dim])?;
            let mut v = workspace.v_out.slice(&[0, 0], &[total_tokens, kv_dim])?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut q, total_tokens, qkv_cols, 0, q_dim, #[cfg(feature = "cuda")] split_stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut k, total_tokens, qkv_cols, q_dim, kv_dim, #[cfg(feature = "cuda")] split_stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut v, total_tokens, qkv_cols, q_dim + kv_dim, kv_dim, #[cfg(feature = "cuda")] split_stream)?;
            self.layers.rope_layers[layer_idx].forward(&input_pos_view, &workspace.sin_cache, &workspace.cos_cache, &mut q, &mut k, cuda_config_ref)?;

            #[cfg(feature = "cuda")]
            if false && batch.is_decode_only() && self.device_type.is_cuda() {
                let k_ptrs_layer = unsafe { workspace.k_cache_ptrs_dev.add(layer_idx * num_seqs) };
                let v_ptrs_layer = unsafe { workspace.v_cache_ptrs_dev.add(layer_idx * num_seqs) };
                crate::op::kernels::cuda::scatter_kv_batch_launch_ready(
                    k.dtype(), kv_dim, num_seqs,
                    &k, &v,
                    kv_dim, kv_dim,
                    0, 0,
                    &input_pos_view,
                    k_ptrs_layer, v_ptrs_layer,
                    cuda_config_ref,
                )?;
            } else {
                for seq_idx in 0..num_seqs {
                    let start = batch.seq_start(seq_idx);
                    let len = batch.seq_len(seq_idx);
                    let pos = batch.seq_pos(seq_idx);
                    let (mut k_dst, mut v_dst) = states[seq_idx].kv_cache.slice_kv_cache(layer_idx, pos, len, kv_dim)?;
                    let k_src = k.slice(&[start, 0], &[len, kv_dim])?;
                    let v_src = v.slice(&[start, 0], &[len, kv_dim])?;
                    k_dst.copy_from_on_current_stream(&k_src)?;
                    v_dst.copy_from_on_current_stream(&v_src)?;
                }
            }

            let attn_all = workspace.intermediate.slice(&[0, 0], &[total_tokens, q_dim])?;
            for seq_idx in 0..num_seqs {
                let start = batch.seq_start(seq_idx);
                let len = batch.seq_len(seq_idx);
                let q_seq = q.slice(&[start, 0], &[len, q_dim])?;
                let (k_hist, v_hist) = states[seq_idx].kv_cache.get(layer_idx)?;
                let mut out_seq = attn_all.slice(&[start, 0], &[len, q_dim])?;
                let kv_len = if self.device_type.is_cuda() {
                    workspace.kv_lens_dev.slice(&[seq_idx], &[1])?
                } else {
                    workspace.kv_lens_cpu.slice(&[seq_idx], &[1])?
                };
                self.layers.mha_layers[layer_idx].forward(&q_seq, k_hist, v_hist, &kv_len, &mut out_seq, cuda_config_ref)?;
            }

            let mut wo_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
            self.layers.wo_layers[layer_idx].forward(&attn_all, &mut wo_out, cuda_config_ref)?;

            let mut ffn_norm_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
            if self.device_type.is_cuda() {
                crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                    &mut ffn_norm_out, &mut x, &wo_out,
                    &self.layers.rmsnorm_ffn_layers[layer_idx].weight,
                    self.config.rms_norm_eps, cuda_config_ref,
                )?;
            } else {
                self.layers.add_layers.forward(&wo_out, &mut x, cuda_config_ref)?;
                self.layers.rmsnorm_ffn_layers[layer_idx].forward(&x, &mut ffn_norm_out, cuda_config_ref)?;
            }

            let mut gate_up = workspace.gate_up_out.slice(&[0, 0], &[total_tokens, 2 * inter])?;
            self.layers.w_gate_up_layers[layer_idx].forward(&ffn_norm_out, &mut gate_up, cuda_config_ref)?;

            let mut w1_out = workspace.w1_out.slice(&[0, 0], &[total_tokens, inter])?;
            let mut w3_out = workspace.w3_out.slice(&[0, 0], &[total_tokens, inter])?;
            crate::op::split_cols::split_cols_tensor(&gate_up, &mut w1_out, total_tokens, 2 * inter, 0, inter, #[cfg(feature = "cuda")] split_stream)?;
            crate::op::split_cols::split_cols_tensor(&gate_up, &mut w3_out, total_tokens, 2 * inter, inter, inter, #[cfg(feature = "cuda")] split_stream)?;
            self.layers.swiglu_layers[layer_idx].forward(&w3_out, &mut w1_out, cuda_config_ref)?;

            let mut w2_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
            self.layers.w2_layers[layer_idx].forward(&w1_out, &mut w2_out, cuda_config_ref)?;
            if self.device_type.is_cuda() {
                let next_norm_weight = if layer_idx + 1 < self.config.layer_num {
                    &self.layers.rmsnorm_attn_layers[layer_idx + 1].weight
                } else {
                    &self.layers.rmsnorm_final_layer.weight
                };
                let mut next_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
                crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                    &mut next_out, &mut x, &w2_out,
                    next_norm_weight, self.config.rms_norm_eps, cuda_config_ref,
                )?;
            } else {
                self.layers.add_layers.forward(&w2_out, &mut x, cuda_config_ref)?;
            }
        }

        let mut final_norm_all = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
        if !self.device_type.is_cuda() {
            self.layers.rmsnorm_final_layer.forward(&x, &mut final_norm_all, cuda_config_ref)?;
        }

        let sample_hidden = workspace.intermediate.slice(&[0, 0], &[num_seqs, dim])?;
        for seq_idx in 0..num_seqs {
            let last = batch.seq_end(seq_idx) - 1;
            let src = final_norm_all.slice(&[last, 0], &[1, dim])?;
            let mut dst = sample_hidden.slice(&[seq_idx, 0], &[1, dim])?;
            dst.copy_from_on_current_stream(&src)?;
        }

        let mut logits = workspace.logits.slice(&[0, 0], &[num_seqs, self.config.vocab_size])?;
        self.layers.cls_layer.forward(&sample_hidden, &mut logits, cuda_config_ref)?;

        let tok_vocab = self.config.tokenizer_vocab_size;
        #[cfg(feature = "cuda")]
        let use_batched_argmax = self.device_type.is_cuda() && logits.dtype() == DataType::BF16;
        #[cfg(not(feature = "cuda"))]
        let use_batched_argmax = false;

        if use_batched_argmax {
            #[cfg(feature = "cuda")]
            {
                let mut out_view = output_tokens.slice(&[0], &[num_seqs])?;
                crate::op::kernels::cuda::argmax_batch_strided(&logits, tok_vocab, self.config.vocab_size, 0, num_seqs, &mut out_view, cuda_config_ref)?;
            }
        } else {
            for i in 0..num_seqs {
                let logits_row = logits.slice(&[i, 0], &[1, self.config.vocab_size])?;
                let logits_trimmed = logits_row.slice(&[0, 0], &[1, tok_vocab])?;
                let logits_1d = logits_trimmed.reshape(&[tok_vocab])?;
                let mut dst = output_tokens.slice(&[i], &[1])?;
                states[i].sampler.sample(&logits_1d, &mut dst, cuda_config_ref)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
//  Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::error::Result;

    fn generate_and_measure(
        model: &Llama3, state: &mut InferenceState,
        prompt: &str, max_tokens: usize, verbose: bool,
    ) -> Result<(String, u64, u32, u64, u64, usize)> {
        let start = Instant::now();
        let (text, n_tok, prefill_ms, decode_ms, decode_iter) = model.generate(state, prompt, max_tokens, verbose)?;
        Ok((text, start.elapsed().as_millis() as u64, n_tok, prefill_ms, decode_ms, decode_iter))
    }

    /// Pre-run a tiny `generate()` on the same state the benchmark will
    /// use, so the real call sees a hot CUDA context:
    ///
    /// - kernel module load / PTX→SASS JIT for every prefill + decode
    ///   kernel (embedding, rmsnorm, qkv, rope, scatter_kv, flash-attn,
    ///   wo, silu, sampler, …).
    /// - cuBLASLt algorithm heuristics for every `(M,N,K)` shape hit.
    /// - the decode-path CUDA Graph: captured here, replayed for free
    ///   from the real benchmark's first decode step onward.
    ///
    /// After warmup returns, the real `generate(prompt, N)` call
    /// unconditionally overwrites `kv_cache[..prompt_len]` from pos=0
    /// (see [`Llama3::generate`]), so warmup has no correctness impact.
    ///
    /// ## Why the filler prompt
    ///
    /// The flash-attention prefill kernel processes the sequence in
    /// fixed-size tiles (~64 tokens). Feeding it a prompt of 1–2 tokens
    /// causes out-of-range reads inside the tile, putting the CUDA
    /// context into a sticky-error state that surfaces later as
    /// `CUBLAS_STATUS_EXECUTION_FAILED (13)` on the next cuBLASLt call.
    /// We pick a ~10-token filler string to stay above that floor while
    /// keeping the warmup cheap.
    fn warmup(model: &Llama3, state: &mut InferenceState) -> Result<()> {
        // ~10 tokens after BPE — safely above the flash-attn prefill
        // tile floor. A few decode steps also capture the CUDA Graph.
        let prompt = "The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog.";
        let _ = model.generate(state, prompt, 4, false)?;
        Ok(())
    }

    #[test]
    #[ignore = "Long running test"]
    fn test_llama3_cpu_loading_and_generation() -> Result<()> {
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Model not found.");

        let model = Llama3::new(model_path, DeviceType::Cpu)?;
        let mut state = model.create_state()?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_, _, n_tok, prefill_ms, decode_ms, decode_iter) = generate_and_measure(&model, &mut state, prompt, 150, true)?;
        assert!(n_tok > 0, "No tokens generated.");

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== CPU: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    #[test]
    #[ignore = "Long running test"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_performance() -> Result<()> {
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Model not found.");

        let model = Llama3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        warmup(&model, &mut state)?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_, _, n_tok, prefill_ms, decode_ms, decode_iter) = generate_and_measure(&model, &mut state, prompt, 2000, false)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== BF16 CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    fn get_dummy_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b")
    }

    fn get_awq_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/vllm_test/llama3.2-1b-AWQ-mlp3")
    }

    #[test]
    #[ignore = "Long running test"]
    #[cfg(feature = "cuda")]
    fn test_llama3_awq_cuda() -> Result<()> {
        let model_path = get_awq_model_path();
        assert!(model_path.exists(), "AWQ model not found.");

        let model = Llama3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        warmup(&model, &mut state)?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello, who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_text, _, n_tok, prefill_ms, decode_ms, decode_iter) = generate_and_measure(&model, &mut state, prompt, 2000, false)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== K-packed INT4 CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }
}

// ══════════════════════════════════════════════════════════════════
// LlmModel trait 实现
// ══════════════════════════════════════════════════════════════════

impl crate::model::llm::LlmModel for Llama3 {
    fn config(&self) -> &crate::model::common::config::RuntimeModelConfig {
        Llama3::config(self)
    }

    fn tokenizer(&self) -> &dyn crate::model::common::tokenizer::Tokenizer {
        Llama3::tokenizer(self)
    }

    fn create_state(&self) -> Result<crate::model::runtime::InferenceState> {
        Llama3::create_state(self)
    }

    fn forward(
        &self,
        states: &mut [&mut crate::model::runtime::InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        batch: &crate::worker::runner::WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        Llama3::forward(self, states, workspace, batch, output_tokens, cuda_config)
    }

    fn fill_rope_cache(
        &self,
        dst_sin: &mut Tensor,
        dst_cos: &mut Tensor,
    ) -> Result<()> {
        crate::model::runtime::compute_rope_cache(
            Llama3::config(self), dst_sin, dst_cos,
        )
    }
}
