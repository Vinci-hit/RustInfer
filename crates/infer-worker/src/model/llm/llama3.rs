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
            .map(|_| FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size))
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

        let mut input_pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        input_pos.as_i32_mut()?.as_slice_mut()?[0] = 0;
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidArgument("Prompt cannot be empty.".to_string()).into());
        }
        let mut input_tokens_cpu = Tensor::new(&[prompt_tokens.len()], DataType::I32, DeviceType::Cpu)?;
        input_tokens_cpu.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&prompt_tokens);

        // Prefill
        let prefill_start = Instant::now();
        let mut current_token = self.forward_prefill(state, &input_tokens_cpu, &input_pos, prompt_tokens.len())?;
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let mut generated_tokens = vec![current_token];
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
        let mut input_tokens_cpu = input_tokens_cpu.slice(&[0], &[1])?;
        for pos in prompt_tokens.len()..(prompt_tokens.len() - 1 + max_tokens) {
            input_pos.as_i32_mut()?.as_slice_mut()?[0] = pos as i32;
            input_tokens_cpu.as_i32_mut()?.as_slice_mut()?[0] = current_token;
            let next_token = self.forward_decoding(state, &input_tokens_cpu, &input_pos)?;

            if self.tokenizer.is_eos(next_token) { break; }

            generated_tokens.push(next_token);
            current_token = next_token;
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

    fn forward_decoding(&self, state: &mut InferenceState, _tokens: &Tensor, pos_cpu: &Tensor) -> Result<i32> {
        state.input_pos.copy_from(pos_cpu)?;
        let input_tokens_view = &state.output_token;

        // CUDA Graph
        if self.device_type.is_cuda() {
            let cfg = state.cuda_config.as_mut().expect("CudaConfig should be initialized");
            if cfg.cuda_graph.is_none() {
                cfg.capture_graph_begin()?;
            } else {
                cfg.launch_graph()?;
                cfg.sync_stream()?;
                return Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0]);
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
            if cfg.cuda_graph.is_none() { cfg.capture_graph_end()?; }
        }

        Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0])
    }

    fn forward_prefill(&self, state: &mut InferenceState, tokens: &Tensor, pos_cpu: &Tensor, seq_len: usize) -> Result<i32> {
        let pos = pos_cpu.as_i32()?.as_slice()?[0] as usize;

        let cuda_config_ref = if self.device_type.is_cuda() { state.cuda_config.as_ref() } else { None };

        state.input_pos.copy_from(pos_cpu)?;

        let input_tokens_buffer = state.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[seq_len])?;
        input_tokens_view.copy_from(tokens)?;

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
            let stream = cuda_config_ref.map_or(std::ptr::null_mut(), |c| c.stream);
            crate::op::split_cols::split_cols_tensor(&qkv, &mut q, seq_len, qkv_cols, 0, self.config.q_dim, stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut k, seq_len, qkv_cols, self.config.q_dim, self.config.kv_dim, stream)?;
            crate::op::split_cols::split_cols_tensor(&qkv, &mut v, seq_len, qkv_cols, self.config.q_dim + self.config.kv_dim, self.config.kv_dim, stream)?;

            let sin_cache = state.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = state.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&state.input_pos, sin_cache, cos_cache, &mut q, &mut k, cuda_config_ref)?;

            let (k_hist, v_hist) = state.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out;
            self.layers.mha_layers[i].forward(&q, k_hist, v_hist, pos_cpu, &mut attn_out, cuda_config_ref)?;
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
            let stream = cuda_config_ref.map_or(std::ptr::null_mut(), |c| c.stream);
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

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello, who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (text, _, n_tok, prefill_ms, decode_ms, decode_iter) = generate_and_measure(&model, &mut state, prompt, 2000, false)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== K-packed INT4 CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }
}
