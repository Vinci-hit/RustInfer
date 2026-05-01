use std::io::{self, Write};
use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::cuda::CudaConfig;
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
use crate::model::BufferType;
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

        // Prefill
        let prefill_start = Instant::now();
        let first_token = self.forward_prefill(state, &prompt_tokens, 0, prompt_tokens.len())?;
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let mut generated_tokens = vec![first_token];
        let mut printed_len = 0usize;
        if print_output {
            let decoded = self.tokenizer.decode(&generated_tokens)?;
            let _ = write!(stdout, "{}", &decoded[printed_len..]);
            printed_len = decoded.len();
            stdout.flush()?;
        }

        // Decode
        let decode_start = Instant::now();
        let mut decode_iterations = 0;
        for pos in prompt_tokens.len()..(prompt_tokens.len() - 1 + max_tokens) {
            let next_token = self.forward_decoding(state, pos as i32)?;

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

    /// 单步 decode（B=1）。
    ///
    /// * `pos` — 当前待生成 token 的绝对位置（**不是** 上一步生成的 token 的 pos）。
    ///
    /// 输入 token 隐含为 `state.output_token`（graph 自闭环：上一步 sampler 写入，本步 embedding 读）；
    /// 调用方需确保 state 已完成 prefill 且 `state.output_token` 持有正确值。
    pub fn forward_decoding(&self, state: &mut InferenceState, pos: i32) -> Result<i32> {
        // state.input_pos 容量 [max_seq_len]，decode 只写前 1 个元素
        state.input_pos.write_from_i32_host(&[pos], 1)?;
        let input_tokens_view = &state.output_token;

        // CUDA Graph
        if self.device_type.is_cuda() {
            let cfg = state.cuda_config.as_mut().expect("CudaConfig should be initialized");
            let slot = crate::cuda::GraphSlot::LlmDecode(1);
            if cfg.graph_ready(slot) {
                cfg.launch(slot)?;
                cfg.sync_stream()?;
                return Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0]);
            } else {
                cfg.capture_begin()?;
            }
        }

        let cuda_config_ref = if self.device_type.is_cuda() { state.cuda_config.as_ref() } else { None };

        let x_buffer = state.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.embedding_layer.forward(input_tokens_view, &mut x, cuda_config_ref)?;

        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            if i == 0 || !self.device_type.is_cuda() {
                self.layers.rmsnorm_attn_layers[i].forward(&x, &mut attn_norm_out, cuda_config_ref)?;
            }

            let qkv_cols = self.config.q_dim + 2 * self.config.kv_dim;
            let qkv_buffer = state.workspace.get_mut(&BufferType::QkvOutput).unwrap();
            let mut qkv = qkv_buffer.slice(&[0, 0], &[1, qkv_cols])?;
            self.layers.wqkv_layers[i].forward(&attn_norm_out, &mut qkv, cuda_config_ref)?;

            let mut q = qkv.slice(&[0, 0], &[1, self.config.q_dim])?;
            let mut k = qkv.slice(&[0, self.config.q_dim], &[1, self.config.kv_dim])?;
            let v = qkv.slice(&[0, self.config.q_dim + self.config.kv_dim], &[1, self.config.kv_dim])?;
            let (k_cache_full, v_cache_full) = state.kv_cache.get_mut(i)?;

            let sin_cache = state.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = state.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&state.input_pos, sin_cache, cos_cache, &mut q, &mut k, cuda_config_ref)?;

            crate::op::scatter::scatter_kv(k_cache_full, &k, v_cache_full, &v, &state.input_pos, cuda_config_ref)?;

            let (k_hist, v_hist) = state.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out;
            self.layers.mha_layers[i].forward(&q, k_hist, v_hist, &state.input_pos, &mut attn_out, cuda_config_ref)?;
            let mut wo_out = q;
            self.layers.wo_layers[i].forward(&attn_out, &mut wo_out, cuda_config_ref)?;

            let mut ffn_norm_out = attn_out;
            if self.device_type.is_cuda() {
                crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                    &mut ffn_norm_out,
                    &mut x,
                    &wo_out,
                    &self.layers.rmsnorm_ffn_layers[i].weight,
                    self.config.rms_norm_eps,
                    cuda_config_ref,
                )?;
            } else {
                self.layers.add_layers.forward(&wo_out, &mut x, cuda_config_ref)?;
                self.layers.rmsnorm_ffn_layers[i].forward(&x, &mut ffn_norm_out, cuda_config_ref)?;
            }

            let inter = self.config.intermediate_size;
            let gu_buffer = state.workspace.get_mut(&BufferType::GateUpOutput).unwrap();
            let mut gate_up = gu_buffer.slice(&[0, 0], &[1, 2 * inter])?;
            self.layers.w_gate_up_layers[i].forward(&ffn_norm_out, &mut gate_up, cuda_config_ref)?;
            let mut w1_out = gate_up.slice(&[0, 0], &[1, inter])?;
            let w3_out = gate_up.slice(&[0, inter], &[1, inter])?;
            self.layers.swiglu_layers[i].forward(&w3_out, &mut w1_out, cuda_config_ref)?;

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[i].forward(&w1_out, &mut w2_out, cuda_config_ref)?;

            if self.device_type.is_cuda() {
                if i + 1 < self.config.layer_num {
                    let buf = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
                    let mut next_out = buf.slice(&[0, 0], &[1, self.config.dim])?;
                    crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                        &mut next_out,
                        &mut x,
                        &w2_out,
                        &self.layers.rmsnorm_attn_layers[i + 1].weight,
                        self.config.rms_norm_eps,
                        cuda_config_ref,
                    )?;
                } else {
                    let buf = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
                    let mut final_out = buf.slice(&[0, 0], &[1, self.config.dim])?;
                    crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                        &mut final_out,
                        &mut x,
                        &w2_out,
                        &self.layers.rmsnorm_final_layer.weight,
                        self.config.rms_norm_eps,
                        cuda_config_ref,
                    )?;
                }
            } else {
                self.layers.add_layers.forward(&w2_out, &mut x, cuda_config_ref)?;
            }
        }

        let final_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        if !self.device_type.is_cuda() {
            self.layers.rmsnorm_final_layer.forward(&x, &mut final_norm_out, cuda_config_ref)?;
        }

        let logits = state.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
        self.layers.cls_layer.forward(&final_norm_out, logits, cuda_config_ref)?;
        let logits_full = state.workspace.get(&BufferType::ForwardOutput).unwrap();
        let logits_ref = logits_full.slice(&[0], &[self.config.tokenizer_vocab_size])?;
        state.sampler.sample(&logits_ref, &mut state.output_token, cuda_config_ref)?;

        if self.device_type.is_cuda() {
            let cfg = state.cuda_config.as_mut().expect("CudaConfig should be initialized");
            let slot = crate::cuda::GraphSlot::LlmDecode(1);
            if !cfg.graph_ready(slot) {
                cfg.capture_end(slot)?;
                // capture 期间 kernel 只被记录不被执行，所以 step 0 必须手动 launch
                // 一次，否则 state.output_token 仍是上一步（prefill 最后一次）的值。
                cfg.launch(slot)?;
                cfg.sync_stream()?;
            }
        }

        Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0])
    }

    /// 一段 prefill，把 `tokens[0..seq_len]` 投喂模型并一次性生成（采样）下一个 token。
    ///
    /// * `tokens`   — host 侧 i32 token 数组，长度 ≥ seq_len；内部做 H2D
    /// * `start_pos`— `tokens[0]` 对应的 KV cache 绝对位置（continuation 场景可 > 0）
    pub fn forward_prefill(&self, state: &mut InferenceState, tokens: &[i32], start_pos: i32, seq_len: usize) -> Result<i32> {
        assert!(tokens.len() >= seq_len, "tokens slice shorter than seq_len");
        let pos = start_pos as usize;

        let cuda_config_ref = if self.device_type.is_cuda() { state.cuda_config.as_ref() } else { None };

        // 把本段 prefill 的 seq_len 个绝对位置 [pos, pos+1, ..., pos+seq_len-1]
        // 写到 state.input_pos 前 seq_len 个元素，供 RoPE 按 per-row 语义消费。
        let positions_host: Vec<i32> = (0..seq_len).map(|i| (pos + i) as i32).collect();
        state.input_pos.write_from_i32_host(&positions_host, seq_len)?;

        // MHA.forward 的 kv_len 参数：prefill 场景下为 start_pos（历史长度），
        // 存在一个 host 侧单元素 tensor 中。
        let mut kv_len_tensor = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        kv_len_tensor.as_i32_mut()?.as_slice_mut()?[0] = start_pos;

        // tokens H2D 到 InputTokens buffer 的前 seq_len 个元素
        let input_tokens_buffer = state.workspace.get_mut(&BufferType::InputTokens).unwrap();
        input_tokens_buffer.write_from_i32_host(tokens, seq_len)?;
        let input_tokens_view = input_tokens_buffer.slice(&[0], &[seq_len])?;

        let x_buffer = state.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
        self.layers.embedding_layer.forward(&input_tokens_view, &mut x, cuda_config_ref)?;

        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&x, &mut attn_norm_out, cuda_config_ref)?;

            let q_buffer = state.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[seq_len, self.config.q_dim])?;
            let (mut k, mut v) = state.kv_cache.slice_kv_cache(i, pos as i32, seq_len, self.config.kv_dim)?;

            let qkv_cols = self.config.q_dim + 2 * self.config.kv_dim;
            let qkv_buffer = state.workspace.get_mut(&BufferType::QkvOutput).unwrap();
            let mut qkv = qkv_buffer.slice(&[0, 0], &[seq_len, qkv_cols])?;
            self.layers.wqkv_layers[i].forward(&attn_norm_out, &mut qkv, cuda_config_ref)?;
            let stream = CudaConfig::resolve_stream(cuda_config_ref);
            crate::op::split_cols::split_cols_tensor(&qkv, &mut q, seq_len, qkv_cols, 0, self.config.q_dim, stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut k, seq_len, qkv_cols, self.config.q_dim, self.config.kv_dim, stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut v, seq_len, qkv_cols, self.config.q_dim + self.config.kv_dim, self.config.kv_dim, stream)?;

            let sin_cache = state.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = state.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&state.input_pos, sin_cache, cos_cache, &mut q, &mut k, cuda_config_ref)?;

            let (k_hist, v_hist) = state.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out;
            self.layers.mha_layers[i].forward(&q, k_hist, v_hist, &kv_len_tensor, &mut attn_out, cuda_config_ref)?;
            let mut wo_out = q;
            self.layers.wo_layers[i].forward(&attn_out, &mut wo_out, cuda_config_ref)?;

            self.layers.add_layers.forward(&wo_out, &mut x, cuda_config_ref)?;

            // FFN
            let mut ffn_norm_out = attn_out;
            self.layers.rmsnorm_ffn_layers[i].forward(&x, &mut ffn_norm_out, cuda_config_ref)?;
            let w1_buffer = state.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;
            let w3_buffer = state.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;

            let inter = self.config.intermediate_size;
            let gu_buffer = state.workspace.get_mut(&BufferType::GateUpOutput).unwrap();
            let mut gate_up = gu_buffer.slice(&[0, 0], &[seq_len, 2 * inter])?;
            self.layers.w_gate_up_layers[i].forward(&ffn_norm_out, &mut gate_up, cuda_config_ref)?;
            let stream = CudaConfig::resolve_stream(cuda_config_ref);
            crate::op::split_cols::split_cols_tensor(&gate_up, &mut w1_out, seq_len, 2 * inter, 0, inter, stream)?;
            crate::op::split_cols::split_cols_tensor(&gate_up, &mut w3_out, seq_len, 2 * inter, inter, inter, stream)?;
            self.layers.swiglu_layers[i].forward(&w3_out, &mut w1_out, cuda_config_ref)?;

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[i].forward(&w1_out, &mut w2_out, cuda_config_ref)?;

            self.layers.add_layers.forward(&w2_out, &mut x, cuda_config_ref)?;
        }

        // Extract last token
        let last_hidden = x.slice(&[seq_len - 1, 0], &[1, self.config.dim])?;
        let buf = state.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
        let mut final_norm_input = buf.slice(&[0, 0], &[1, self.config.dim])?;
        final_norm_input.copy_from(&last_hidden)?;

        let final_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.rmsnorm_final_layer.forward(&final_norm_input, &mut final_norm_out, cuda_config_ref)?;

        let logits = state.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
        self.layers.cls_layer.forward(&final_norm_out, logits, cuda_config_ref)?;

        let logits_full = state.workspace.get(&BufferType::ForwardOutput).unwrap();
        let logits_ref = logits_full.slice(&[0], &[self.config.tokenizer_vocab_size])?;
        state.sampler.sample(&logits_ref, &mut state.output_token, cuda_config_ref)?;

        Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0])
    }

    // ════════════════════════════════════════════════════════════════
    // Batch Forward Methods
    // ════════════════════════════════════════════════════════════════

    /// Batch decode forward: B 个 seq 各 seq_len=1，共享 matmul/rmsnorm 等 op 调用。
    ///
    /// 与 `forward_decoding` 的区别:
    /// - 所有 [1, dim] 变为 [B, dim]，一次 GEMM 调用
    /// - RoPE 用 positions[B] 为每行不同 pos
    /// - scatter_kv_batch: 写 B 行到 B 个独立 cache
    /// - FlashAttn: 循环 per-seq (Phase 1)
    /// - Sampler: 循环 per-seq (Phase 1)
    ///
    /// # Arguments
    /// * `states` - B 个 InferenceState (各自有独立 KV cache)
    /// * `workspace` - 共享的 batch workspace (预分配 [max_batch, dim] 等)
    /// * `positions` - [B] CPU i32, 每个 seq 的 decode position
    /// Batched decode 入口。
    ///
    /// `states` 采用 `&mut [&mut InferenceState]`（可变引用切片）而非 `&mut [InferenceState]`，
    /// 这样调用者可以从一个大 slot 池子里用 `get_disjoint_mut` / 手动 split 拿出任意子集，
    /// 不要求它们在内存里连续——对 continuous batching 的 scheduler 是必需的。
    pub fn forward_batch_decode(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        positions: &[i32],
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Vec<i32>> {
        let batch_size = states.len();
        assert_eq!(positions.len(), batch_size);
        assert!(batch_size > 0);

        // ═══ B=1 fast path: 直接转发到 serial `forward_decoding` ═══
        // forward_decoding 已经实现了高度优化的 CUDA Graph + fused_add_rmsnorm + hgemv 路径，
        // 性能最优。batch 路径的 "_batch" kernel 族只在 B>1 时才有必要。
        #[cfg(feature = "cuda")]
        if batch_size == 1 && self.device_type.is_cuda() {
            let tok = self.forward_decoding(&mut *states[0], positions[0])?;
            return Ok(vec![tok]);
        }

        let dim = self.config.dim;
        let q_dim = self.config.q_dim;
        let kv_dim = self.config.kv_dim;
        let inter = self.config.intermediate_size;
        let qkv_cols = q_dim + 2 * kv_dim;

        let cuda_config_ref = cuda_config;

        // ═══════════════════════════════════════════════════════════════
        // Stage 1: Host prelude (capture 之外) — 更新每步 H2D buffer + 首次填 KV 指针
        // ═══════════════════════════════════════════════════════════════

        // 1a. 首次调用：把每个 state.output_token → workspace.output_tokens (D2D 一次性)
        //     之后的调用：embedding 直接从 workspace.output_tokens 读（graph 自闭环），
        //     sampler 写回同一 buffer，prelude 不再需要每步 D2D input_tokens。
        #[cfg(feature = "cuda")]
        if !workspace.cache_ptrs_filled {
            let stream = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);
            crate::cuda::with_cuda_stream(stream, || -> Result<()> {
                let out_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
                for i in 0..batch_size {
                    let mut dst = out_view.slice(&[i], &[1])?;
                    let src = &states[i].output_token;
                    dst.copy_from_on_current_stream(src)?;
                }
                Ok(())
            })?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            // CPU 路径：每步从 state.output_token 搬到 workspace.output_tokens
            // （CPU 不走 graph 闭环）
            let out_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
            for i in 0..batch_size {
                let mut dst = out_view.slice(&[i], &[1])?;
                let src = &states[i].output_token;
                dst.copy_from(src)?;
            }
        }

        // 1b. 收集 positions → input_pos_cpu → input_pos(device) (H2D async)
        //     kv_lens 与 positions 值相同（见 serial forward_decoding 语义），复用同一块 device buffer
        {
            let pos_slice = workspace.input_pos_cpu.as_i32_mut()?.as_slice_mut()?;
            pos_slice[..batch_size].copy_from_slice(positions);
        }
        #[cfg(feature = "cuda")]
        {
            let stream = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);
            crate::cuda::with_cuda_stream(stream, || -> Result<()> {
                let src = workspace.input_pos_cpu.slice(&[0], &[batch_size])?;
                let mut dst = workspace.input_pos.slice(&[0], &[batch_size])?;
                dst.copy_from_on_current_stream(&src)?;
                Ok(())
            })?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let src = workspace.input_pos_cpu.slice(&[0], &[batch_size])?;
            let mut dst = workspace.input_pos.slice(&[0], &[batch_size])?;
            dst.copy_from(&src)?;
        }
        // (原来需要单独做 kv_lens H2D，现在直接让 capture 里的 flash 用 input_pos 作为 kv_lens)

        // 1d. 首次调用：把所有层 × B 个 seq 的 K/V cache 指针一次性填入 workspace 的
        //     device 指针数组。之后 graph replay 复用同一个数组，不再 H2D。
        #[cfg(feature = "cuda")]
        if self.device_type.is_cuda() && !workspace.cache_ptrs_filled {
            let layer_num = self.config.layer_num;
            let mut k_host: Vec<u64> = Vec::with_capacity(layer_num * batch_size);
            let mut v_host: Vec<u64> = Vec::with_capacity(layer_num * batch_size);
            for layer_idx in 0..layer_num {
                for i in 0..batch_size {
                    let (kc, vc) = states[i].kv_cache.get(layer_idx).unwrap();
                    let k_ptr = match kc.dtype() {
                        DataType::BF16 => kc.as_bf16()?.buffer().as_ptr() as u64,
                        DataType::F16 => kc.as_f16()?.buffer().as_ptr() as u64,
                        other => return Err(crate::base::error::Error::InvalidArgument(
                            format!("unsupported kv cache dtype: {:?}", other)).into()),
                    };
                    let v_ptr = match vc.dtype() {
                        DataType::BF16 => vc.as_bf16()?.buffer().as_ptr() as u64,
                        DataType::F16 => vc.as_f16()?.buffer().as_ptr() as u64,
                        other => return Err(crate::base::error::Error::InvalidArgument(
                            format!("unsupported kv cache dtype: {:?}", other)).into()),
                    };
                    k_host.push(k_ptr);
                    v_host.push(v_ptr);
                }
            }
            let stream = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);
            let bytes = layer_num * batch_size * std::mem::size_of::<u64>();
            unsafe {
                crate::cuda_check!(crate::cuda::ffi::cudaMemcpyAsync(
                    workspace.k_cache_ptrs_dev as *mut _,
                    k_host.as_ptr() as *const _,
                    bytes,
                    crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream,
                ))?;
                crate::cuda_check!(crate::cuda::ffi::cudaMemcpyAsync(
                    workspace.v_cache_ptrs_dev as *mut _,
                    v_host.as_ptr() as *const _,
                    bytes,
                    crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream,
                ))?;
                crate::cuda_check!(crate::cuda::ffi::cudaStreamSynchronize(stream))?;
            }
            workspace.cache_ptrs_filled = true;
        }

        // ═══════════════════════════════════════════════════════════════
        // Stage 2: CUDA Graph capture/replay fast-path (首次 capture，之后 replay)
        // ═══════════════════════════════════════════════════════════════
        #[cfg(feature = "cuda")]
        let use_graph = self.device_type.is_cuda()
            && workspace.input_tokens.dtype() == DataType::I32
            && matches!(workspace.x.dtype(), DataType::BF16)
            && self.config.head_size == 64;
        #[cfg(not(feature = "cuda"))]
        let use_graph = false;

        #[cfg(feature = "cuda")]
        if use_graph {
            let cuda_cfg = cuda_config_ref.ok_or_else(||
                crate::base::error::Error::InvalidArgument(
                    "forward_batch_decode graph path requires CudaConfig".into()
                ))?;

            // 按实际 batch_size 分桶 cache graph
            let slot = crate::cuda::GraphSlot::LlmDecode(batch_size);

            if cuda_cfg.graph_ready(slot) {
                // Fast path: replay
                cuda_cfg.launch(slot)?;
                cuda_cfg.sync_stream()?;
            } else {
                // Slow path:
                //   1) 先跑一次 non-graph forward (让 cuBLASLt algorithm cache / kernel JIT 预热)
                //   2) warmup 会把 sampler 新输出写到 workspace.output_tokens，把它重置回
                //      每个 state.output_token 的原始值（= prefill/上一步 forward 的真实 input）
                //   3) begin_capture → 第二次 forward (这次仅被记录) → end_capture
                //   4) 立即 launch 一次，让本次 caller step 真正执行并返回正确 token
                //
                // KV cache 会被写两次（同 pos），第二次覆盖第一次 — 不影响正确性。
                self.forward_batch_decode_capture(
                    states, workspace, batch_size, dim, q_dim, kv_dim, inter, qkv_cols,
                    cuda_config_ref,
                )?;
                cuda_cfg.sync_stream()?;

                // reset workspace.output_tokens[i] = state[i].output_token (原始 input token)
                {
                    let stream = cuda_cfg.stream;
                    crate::cuda::with_cuda_stream(stream, || -> Result<()> {
                        let out_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
                        for i in 0..batch_size {
                            let mut dst = out_view.slice(&[i], &[1])?;
                            let src = &states[i].output_token;
                            dst.copy_from_on_current_stream(src)?;
                        }
                        Ok(())
                    })?;
                    cuda_cfg.sync_stream()?;
                }

                let cfg_ptr = cuda_cfg as *const crate::cuda::CudaConfig
                    as *mut crate::cuda::CudaConfig;
                cuda_cfg.capture_begin()?;
                self.forward_batch_decode_capture(
                    states, workspace, batch_size, dim, q_dim, kv_dim, inter, qkv_cols,
                    cuda_config_ref,
                )?;
                unsafe { (*cfg_ptr).capture_end(slot)?; }
                cuda_cfg.launch(slot)?;
                cuda_cfg.sync_stream()?;
            }
        } else {
            // Non-graph 路径（CPU 或者首次未达到 graph 条件）
            self.forward_batch_decode_capture(
                states, workspace, batch_size, dim, q_dim, kv_dim, inter, qkv_cols,
                cuda_config_ref,
            )?;
        }
        #[cfg(not(feature = "cuda"))]
        self.forward_batch_decode_capture(
            states, workspace, batch_size, dim, q_dim, kv_dim, inter, qkv_cols,
            cuda_config_ref,
        )?;

        // ═══════════════════════════════════════════════════════════════
        // Stage 3: Postlude (capture 外) — D2D copy 每个 state.output_token + 单次 D2H 读 tokens
        // ═══════════════════════════════════════════════════════════════
        let output_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
        for i in 0..batch_size {
            let src_view = workspace.output_tokens.slice(&[i], &[1])?;
            states[i].output_token.copy_from_on_current_stream(&src_view)?;
        }
        let out_cpu = output_view.to_cpu()?;
        let out_slice = out_cpu.as_i32()?.as_slice()?;
        Ok(out_slice.to_vec())
    }

    /// Pure GPU compute 部分，不含任何 host-side 读写/同步，可以被 CUDA Graph 完整 capture。
    /// 输入读自 `workspace.input_tokens / input_pos / kv_lens_dev / sin_cache / cos_cache
    /// / k_cache_ptrs_dev / v_cache_ptrs_dev`，输出写入 `workspace.output_tokens`。
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_decode_capture(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        batch_size: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter: usize,
        qkv_cols: usize,
        cuda_config_ref: Option<&crate::OpConfig>,
    ) -> Result<()> {
        let input_pos_view = workspace.input_pos.slice(&[0], &[batch_size])?;

        // 2. Embedding — 输入直接读 workspace.output_tokens（与 sampler 写的是同一块
        //    buffer），形成 graph 自闭环：当前步的 sampler 输出就是下一步的 embedding 输入。
        //    这样 prelude 里不需要做 D2D copy "state.output_token → workspace.input_tokens"。
        let input_tokens_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
        let mut x = workspace.x.slice(&[0, 0], &[batch_size, dim])?;
        self.layers.embedding_layer.forward(&input_tokens_view, &mut x, cuda_config_ref)?;

        // 3. Transformer layers
        for layer_idx in 0..self.config.layer_num {
            let mut attn_norm_out = workspace.rms_out.slice(&[0, 0], &[batch_size, dim])?;
            if layer_idx == 0 || !self.device_type.is_cuda() {
                self.layers.rmsnorm_attn_layers[layer_idx].forward(&x, &mut attn_norm_out, cuda_config_ref)?;
            }

            let mut qkv = workspace.qkv_out.slice(&[0, 0], &[batch_size, qkv_cols])?;
            self.layers.wqkv_layers[layer_idx].forward(&attn_norm_out, &mut qkv, cuda_config_ref)?;

            #[cfg(feature = "cuda")]
            let split_stream = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);

            // RoPE (strided)：直接从 qkv 的 q/k 段寻址，不做 split_cols
            #[cfg(feature = "cuda")]
            if self.device_type.is_cuda() && qkv.dtype() == DataType::BF16 {
                crate::op::kernels::cuda::rope_strided(
                    dim, kv_dim, self.config.head_size,
                    &mut qkv, &mut workspace.qkv_out, // k 也写在 qkv 上（同一块内存）
                    qkv_cols, qkv_cols,
                    0, q_dim,                          // q_col_offset, k_col_offset
                    &input_pos_view,
                    &workspace.sin_cache, &workspace.cos_cache,
                    batch_size,
                    cuda_config_ref,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                // CPU fallback: 仍走 split → per-row rope
                let qkv_runtime_dtype = qkv.dtype();
                let mut q = Tensor::new(&[batch_size, q_dim], qkv_runtime_dtype, qkv.device())?;
                let mut k = Tensor::new(&[batch_size, kv_dim], qkv_runtime_dtype, qkv.device())?;
                crate::op::split_cols::split_cols_tensor(
                    &qkv, &mut q, batch_size, qkv_cols, 0, q_dim,
                )?;
                crate::op::split_cols::split_cols_tensor(
                    &qkv, &mut k, batch_size, qkv_cols, q_dim, kv_dim,
                )?;
                self.layers.rope_layers[layer_idx].forward(
                    &input_pos_view, &workspace.sin_cache, &workspace.cos_cache,
                    &mut q, &mut k, cuda_config_ref,
                )?;
            }

            // scatter_kv_batch (strided)：从 qkv 的 k/v 段直接寻址
            #[cfg(feature = "cuda")]
            if self.device_type.is_cuda() {
                let k_ptrs_layer = unsafe {
                    workspace.k_cache_ptrs_dev.add(layer_idx * batch_size)
                };
                let v_ptrs_layer = unsafe {
                    workspace.v_cache_ptrs_dev.add(layer_idx * batch_size)
                };
                crate::op::kernels::cuda::scatter_kv_batch_launch_ready(
                    qkv.dtype(), kv_dim, batch_size,
                    &qkv, &qkv,
                    qkv_cols, qkv_cols,
                    q_dim, q_dim + kv_dim,
                    &input_pos_view,
                    k_ptrs_layer, v_ptrs_layer,
                    cuda_config_ref,
                )?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                // CPU: 先 split_cols k/v 出来，再 per-seq scatter
                let qkv_dtype = qkv.dtype();
                let mut k = Tensor::new(&[batch_size, kv_dim], qkv_dtype, qkv.device())?;
                let mut v = Tensor::new(&[batch_size, kv_dim], qkv_dtype, qkv.device())?;
                crate::op::split_cols::split_cols_tensor(
                    &qkv, &mut k, batch_size, qkv_cols, q_dim, kv_dim,
                )?;
                crate::op::split_cols::split_cols_tensor(
                    &qkv, &mut v, batch_size, qkv_cols, q_dim + kv_dim, kv_dim,
                )?;
                let mut k_caches: Vec<&mut Tensor> = Vec::with_capacity(batch_size);
                let mut v_caches: Vec<&mut Tensor> = Vec::with_capacity(batch_size);
                for state in states.iter_mut() {
                    let (kc, vc) = state.kv_cache.get_mut(layer_idx)?;
                    k_caches.push(kc);
                    v_caches.push(vc);
                }
                crate::op::scatter::scatter_kv_batch(
                    &mut k_caches, &mut v_caches, &k, &v,
                    workspace.input_pos_cpu.as_i32()?.as_slice()?,
                    cuda_config_ref,
                )?;
            }

            // Attention (batched): attn 输出写到 workspace.q_out（用作 wo 输入，且前 q_dim 连续）
            let mut attn_out = workspace.q_out.slice(&[0, 0], &[batch_size, q_dim])?;
            #[cfg(feature = "cuda")]
            let cuda_flash_path = self.device_type.is_cuda()
                && qkv.dtype() == DataType::BF16
                && self.config.head_size == 64;
            #[cfg(not(feature = "cuda"))]
            let cuda_flash_path = false;

            if cuda_flash_path {
                #[cfg(feature = "cuda")]
                {
                    // kv_lens 和 positions 值相同，直接复用 input_pos_view
                    let kv_lens_view = input_pos_view.clone();
                    let cuda_cfg = cuda_config_ref.ok_or_else(||
                        crate::base::error::Error::InvalidArgument(
                            "flash_decoding_batch_bf16 需要 CudaConfig".into()
                        ))?;
                    let k_ptrs_layer = unsafe {
                        workspace.k_cache_ptrs_dev.add(layer_idx * batch_size)
                    };
                    let v_ptrs_layer = unsafe {
                        workspace.v_cache_ptrs_dev.add(layer_idx * batch_size)
                    };
                    unsafe {
                        crate::op::kernels::cuda::flash_decoding_batch_bf16_launch_ready(
                            &qkv, &mut attn_out,
                            &kv_lens_view,
                            k_ptrs_layer, v_ptrs_layer,
                            cuda_cfg,
                            batch_size,
                            self.config.head_num,
                            self.config.kv_head_num,
                            self.config.head_size,
                            qkv_cols, q_dim, // q_row_stride = qkv_cols, o_row_stride = q_dim (连续)
                            0, 0,            // q_col_offset = 0, o_col_offset = 0
                        )?;
                    }
                }
            } else {
                // CPU fallback: 先 split q 再调原 forward_batch_decode
                let qkv_dtype = qkv.dtype();
                let mut q = Tensor::new(&[batch_size, q_dim], qkv_dtype, qkv.device())?;
                crate::op::split_cols::split_cols_tensor(
                    &qkv, &mut q, batch_size, qkv_cols, 0, q_dim,
                    #[cfg(feature = "cuda")] split_stream,
                )?;
                let kv_lens: Vec<i32> = workspace.input_pos_cpu.as_i32()?.as_slice()?[..batch_size].to_vec();
                let k_cache_refs: Vec<&Tensor> = states.iter()
                    .map(|s| { let (k, _) = s.kv_cache.get(layer_idx).unwrap(); k })
                    .collect();
                let v_cache_refs: Vec<&Tensor> = states.iter()
                    .map(|s| { let (_, v) = s.kv_cache.get(layer_idx).unwrap(); v })
                    .collect();
                self.layers.mha_layers[layer_idx].forward_batch_decode(
                    &q, &k_cache_refs, &v_cache_refs, &kv_lens,
                    &mut attn_out, cuda_config_ref,
                )?;
            }

            // WO projection: [B, q_dim] → [B, dim]
            // wo_out 用独立连续 buffer（workspace.intermediate），避免和 attn_out 地址冲突
            let mut wo_out = workspace.intermediate.slice(&[0, 0], &[batch_size, dim])?;
            self.layers.wo_layers[layer_idx].forward(&attn_out, &mut wo_out, cuda_config_ref)?;

            let mut ffn_norm_out = attn_out;
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

            let mut gate_up = workspace.gate_up_out.slice(&[0, 0], &[batch_size, 2 * inter])?;
            self.layers.w_gate_up_layers[layer_idx].forward(&ffn_norm_out, &mut gate_up, cuda_config_ref)?;

            // w1 必须连续（后续 w2 GEMM 要连续输入）。
            //   - B=1 时，gate_up.slice(col=0, inner=inter) 的物理布局就是前 inter 连续，
            //     可以直接当 w1 使用，省掉一次 split_cols kernel launch。
            //   - B>1 时，需要 split_cols 到独立连续 buffer。
            let mut w1_out = if batch_size == 1 {
                gate_up.slice(&[0, 0], &[batch_size, inter])?
            } else {
                let mut w1 = workspace.w1_out.slice(&[0, 0], &[batch_size, inter])?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut w1, batch_size, 2 * inter, 0, inter,
                    #[cfg(feature = "cuda")] split_stream,
                )?;
                w1
            };
            // w3 不 split，直接用 strided swiglu 从 gate_up 第 inter 列读，in-place 写到 w1_out
            //   B=1: w1_out 实际是 gate_up.slice，row stride = 2*inter
            //   B>1: w1_out 是独立 buffer，row stride = inter
            let x_row_stride_w1 = if batch_size == 1 { 2 * inter } else { inter };
            #[cfg(feature = "cuda")]
            if self.device_type.is_cuda() && w1_out.dtype() == DataType::BF16 {
                unsafe {
                    crate::op::kernels::cuda::swiglu_inplace_strided_bf16(
                        &mut w1_out, &gate_up,
                        batch_size, inter,
                        x_row_stride_w1, // w1 的 row stride (B=1 时 = 2*inter, B>1 时 = inter)
                        2 * inter,       // y_row_stride = gate_up 是 [B, 2*inter]
                        0,               // x_col_offset
                        inter,           // y_col_offset = w3 的起点
                        cuda_config_ref,
                    )?;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                // CPU: split w3 后调原 SwiGLU
                let mut w3_out = workspace.w3_out.slice(&[0, 0], &[batch_size, inter])?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut w3_out, batch_size, 2 * inter, inter, inter,
                )?;
                self.layers.swiglu_layers[layer_idx].forward(&w3_out, &mut w1_out, cuda_config_ref)?;
            }

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[layer_idx].forward(&w1_out, &mut w2_out, cuda_config_ref)?;

            if self.device_type.is_cuda() {
                if layer_idx + 1 < self.config.layer_num {
                    let buf = &mut workspace.rms_out;
                    let mut next_out = buf.slice(&[0, 0], &[batch_size, dim])?;
                    crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                        &mut next_out, &mut x, &w2_out,
                        &self.layers.rmsnorm_attn_layers[layer_idx + 1].weight,
                        self.config.rms_norm_eps, cuda_config_ref,
                    )?;
                } else {
                    let buf = &mut workspace.rms_out;
                    let mut final_out = buf.slice(&[0, 0], &[batch_size, dim])?;
                    crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                        &mut final_out, &mut x, &w2_out,
                        &self.layers.rmsnorm_final_layer.weight,
                        self.config.rms_norm_eps, cuda_config_ref,
                    )?;
                }
            } else {
                self.layers.add_layers.forward(&w2_out, &mut x, cuda_config_ref)?;
            }
        }

        // 4. Final RMSNorm + LM head
        let mut final_norm_out = workspace.rms_out.slice(&[0, 0], &[batch_size, dim])?;
        if !self.device_type.is_cuda() {
            self.layers.rmsnorm_final_layer.forward(&x, &mut final_norm_out, cuda_config_ref)?;
        }

        let mut logits = workspace.logits.slice(&[0, 0], &[batch_size, self.config.vocab_size])?;
        self.layers.cls_layer.forward(&final_norm_out, &mut logits, cuda_config_ref)?;

        // 5. Sample → workspace.output_tokens（D2D/GPU，可被 graph capture）
        let tok_vocab = self.config.tokenizer_vocab_size;
        #[cfg(feature = "cuda")]
        let use_batched_argmax = self.device_type.is_cuda() && logits.dtype() == DataType::BF16;
        #[cfg(not(feature = "cuda"))]
        let use_batched_argmax = false;

        if use_batched_argmax {
            #[cfg(feature = "cuda")]
            {
                let split_stream2 = crate::cuda::CudaConfig::resolve_stream(cuda_config_ref);
                let mut out_view = workspace.output_tokens.slice(&[0], &[batch_size])?;
                // 直接走 strided argmax_batch：从 logits [B, vocab_size] 取前 tok_vocab 列
                crate::op::kernels::cuda::argmax_batch_strided(
                    &logits, tok_vocab, self.config.vocab_size, 0,
                    batch_size, &mut out_view, cuda_config_ref,
                )?;
            }
        } else {
            for i in 0..batch_size {
                let logits_row = logits.slice(&[i, 0], &[1, self.config.vocab_size])?;
                let logits_trimmed = logits_row.slice(&[0, 0], &[1, tok_vocab])?;
                let logits_1d = logits_trimmed.reshape(&[tok_vocab])?;
                states[i].sampler.sample(&logits_1d, &mut states[i].output_token, cuda_config_ref)?;
                // 同时把结果写一份到 workspace.output_tokens[i]，让 postlude 统一走 D2H
                let mut dst = workspace.output_tokens.slice(&[i], &[1])?;
                dst.copy_from(&states[i].output_token)?;
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
        let prompt = "The quick brown fox jumps over the lazy dog.";
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

    fn forward_prefill(
        &self,
        state: &mut crate::model::runtime::InferenceState,
        tokens: &[i32],
        start_pos: i32,
        seq_len: usize,
    ) -> Result<i32> {
        Llama3::forward_prefill(self, state, tokens, start_pos, seq_len)
    }

    fn forward_decoding(
        &self,
        state: &mut crate::model::runtime::InferenceState,
        pos: i32,
    ) -> Result<i32> {
        Llama3::forward_decoding(self, state, pos)
    }

    fn forward_batch_decode(
        &self,
        states: &mut [&mut crate::model::runtime::InferenceState],
        workspace: &mut crate::worker::batch_workspace::BatchWorkspace,
        positions: &[i32],
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Vec<i32>> {
        Llama3::forward_batch_decode(self, states, workspace, positions, cuda_config)
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
