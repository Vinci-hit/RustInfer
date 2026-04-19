use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::base::error::Error::InternalError;
use crate::model::BufferType;
use crate::op::{Op, OpContext};
use crate::op::add_inplace::AddInplace;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::scatter::Scatter;
use crate::op::swiglu::SwiGLU;
use super::runtime::InferenceState;
use crate::tensor::Tensor;
use super::common::config::RuntimeModelConfig;
use super::ModelLoader;
use super::common::tokenizer::Tokenizer;


/// Qwen3Layers holds all operators and weights for the model.
/// Compared to Llama3, Qwen3 adds optional per-head QK-norm layers.
pub struct Qwen3Layers {
    pub embedding_layer: Embedding,
    pub rmsnorm_final_layer: RMSNorm,
    pub cls_layer: Matmul,

    pub rmsnorm_attn_layers: Vec<RMSNorm>,
    pub rmsnorm_ffn_layers: Vec<RMSNorm>,

    pub qnorm_layers: Option<Vec<RMSNorm>>,
    pub knorm_layers: Option<Vec<RMSNorm>>,

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

impl Qwen3Layers {
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.embedding_layer.to_cuda(device_id)?;
        self.rmsnorm_final_layer.to_cuda(device_id)?;
        self.cls_layer.to_cuda(device_id)?;
        self.rmsnorm_attn_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        self.rmsnorm_ffn_layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        if let Some(ref mut layers) = self.qnorm_layers {
            layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        }
        if let Some(ref mut layers) = self.knorm_layers {
            layers.iter_mut().try_for_each(|l| l.to_cuda(device_id))?;
        }
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

/// Qwen3 model — holds only static weights and configuration.
/// Request-level mutable state lives in `InferenceState`.
pub struct Qwen3 {
    pub(crate) config: RuntimeModelConfig,
    pub(crate) device_type: DeviceType,
    pub(crate) tokenizer: Box<dyn Tokenizer>,
    pub(crate) layers: Qwen3Layers,
}

impl Qwen3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
    ) -> Result<Self> {

        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tensor_names: std::collections::HashSet<String> = loader.tensor_names().into_iter().collect();
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();

        let layer_num = config.layer_num;
        let has_qnorm = tensor_names.iter().any(|n| n.contains("q_norm"));
        let has_knorm = tensor_names.iter().any(|n| n.contains("k_norm"));

        let mut rmsnorm_attn_layers = Vec::with_capacity(layer_num);
        let mut rmsnorm_ffn_layers = Vec::with_capacity(layer_num);
        let mut qnorm_layers_opt = Vec::with_capacity(layer_num);
        let mut knorm_layers_opt = Vec::with_capacity(layer_num);
        let mut wqkv_layers = Vec::with_capacity(layer_num);
        let mut wo_layers = Vec::with_capacity(layer_num);
        let mut w_gate_up_layers = Vec::with_capacity(layer_num);
        let mut w2_layers = Vec::with_capacity(layer_num);

        let is_awq = config.quant_config.as_ref().map_or(false, |q|
            q.quant_method == "compressed-tensors" || q.quant_method == "awq");
        let group_size = config.quant_config.as_ref().map(|q| q.group_size).unwrap_or(128);

        for i in 0..layer_num {
            wqkv_layers.push(Self::load_fused_qkv(i, &loader, device_type, config.q_dim, config.kv_dim, config.dim)?);
            wo_layers.push(Matmul::from(Self::load_weight(&format!("model.layers.{}.self_attn.o_proj.weight", i), &loader, device_type)?, None));

            if is_awq {
                w_gate_up_layers.push(Self::load_fused_gate_up_awq(i, &loader, device_type, config.intermediate_size, group_size)?);
                w2_layers.push(Self::load_awq_matmul(&format!("model.layers.{}.mlp.down_proj", i), &loader, device_type, group_size)?);
            } else {
                w_gate_up_layers.push(Self::load_fused_gate_up(i, &loader, device_type, config.intermediate_size, config.dim)?);
                w2_layers.push(Matmul::from(Self::load_weight(&format!("model.layers.{}.mlp.down_proj.weight", i), &loader, device_type)?, None));
            }
            rmsnorm_attn_layers.push(RMSNorm::from(Self::load_weight(&format!("model.layers.{}.input_layernorm.weight", i), &loader, device_type)?, config.rms_norm_eps));
            rmsnorm_ffn_layers.push(RMSNorm::from(Self::load_weight(&format!("model.layers.{}.post_attention_layernorm.weight", i), &loader, device_type)?, config.rms_norm_eps));
            if has_qnorm {
                qnorm_layers_opt.push(RMSNorm::from(Self::load_weight(&format!("model.layers.{}.self_attn.q_norm.weight", i), &loader, device_type)?, config.rms_norm_eps));
            }
            if has_knorm {
                knorm_layers_opt.push(RMSNorm::from(Self::load_weight(&format!("model.layers.{}.self_attn.k_norm.weight", i), &loader, device_type)?, config.rms_norm_eps));
            }
        }

        let embedding_layer = Embedding::from(Self::load_weight("model.embed_tokens.weight", &loader, device_type)?);
        let rmsnorm_final_layer = RMSNorm::from(Self::load_weight("model.norm.weight", &loader, device_type)?, config.rms_norm_eps);
        let cls_layer = if tensor_names.contains("lm_head.weight") {
            Matmul::from(Self::load_weight("lm_head.weight", &loader, device_type)?, None)
        } else {
            Matmul::from(embedding_layer.weight.clone(), None)
        };

        let mha_layers = (0..layer_num)
            .map(|_| FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size))
            .collect::<Result<Vec<_>>>()?;
        let rope_layers = (0..layer_num)
            .map(|_| RoPEOp::new(config.q_dim, config.kv_dim, config.head_size))
            .collect::<Result<Vec<_>>>()?;
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
        if let Some(ref q) = has_qnorm.then_some(&qnorm_layers_opt) {
            if q.len() != layer_num { return Err(InternalError("Incorrect number of Q-norm layers.".to_string()).into()); }
        }
        if let Some(ref k) = has_knorm.then_some(&knorm_layers_opt) {
            if k.len() != layer_num { return Err(InternalError("Incorrect number of K-norm layers.".to_string()).into()); }
        }
        if mha_layers.len() != layer_num || rope_layers.len() != layer_num || swiglu_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of non-parameterized layers.".to_string()).into());
        }

        let layers = Qwen3Layers {
            embedding_layer, rmsnorm_final_layer, cls_layer,
            rmsnorm_attn_layers, rmsnorm_ffn_layers,
            qnorm_layers: has_qnorm.then_some(qnorm_layers_opt),
            knorm_layers: has_knorm.then_some(knorm_layers_opt),
            wqkv_layers, wo_layers, mha_layers, rope_layers,
            add_layers, scatter_layer: Scatter::new(),
            w_gate_up_layers, w2_layers, swiglu_layers,
        };

        Ok(Self { config, device_type, tokenizer, layers })
    }

    /// Create a new InferenceState for this model, including Qwen3-specific workspace buffers.
    pub fn create_state(&self) -> Result<InferenceState> {
        let mut state = InferenceState::new(&self.config, self.device_type)?;

        let float_dtype = self.config.runtime_float_dtype(self.device_type)?;
        let max_seq_len = self.config.seq_len;

        // Override Query buffer: Qwen3 needs [max_seq_len, q_dim], not [max_seq_len, dim]
        state.workspace.insert(BufferType::Query,
            Tensor::new(&[max_seq_len, self.config.q_dim], float_dtype, self.device_type)?);
        // Attention output: [max_seq_len, q_dim] — separate from RmsOutput to avoid aliasing
        state.workspace.insert(BufferType::AttnOutput,
            Tensor::new(&[max_seq_len, self.config.q_dim], float_dtype, self.device_type)?);
        // QK-norm buffers (per-head reshape)
        if self.layers.qnorm_layers.is_some() {
            state.workspace.insert(BufferType::QNormBuffer,
                Tensor::new(&[max_seq_len * self.config.head_num, self.config.head_size], float_dtype, self.device_type)?);
        }
        if self.layers.knorm_layers.is_some() {
            state.workspace.insert(BufferType::KNormBuffer,
                Tensor::new(&[max_seq_len * self.config.kv_head_num, self.config.head_size], float_dtype, self.device_type)?);
        }

        Ok(state)
    }

    // ---- Weight loading helpers ----

    /// Load a raw tensor, optionally casting to F32 on CPU, and move to `device`.
    fn load_weight(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Tensor> {
        let weight = Tensor::from_view_on_cpu(&loader.get_tensor(name)?)?;
        let weight = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };
        weight.to_device(device)
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

        let k_packed = gate_wp.shape()[1];
        let num_groups = gate_sc.shape()[1];

        let fused_wp = Self::fuse_tensors_vertically(
            &[&gate_wp, &up_wp],
            &[intermediate_size, intermediate_size],
            k_packed, DataType::I32,
        )?;

        let sc_dtype = gate_sc.dtype();
        let fused_sc = Self::fuse_tensors_vertically(
            &[&gate_sc, &up_sc],
            &[intermediate_size, intermediate_size],
            num_groups, sc_dtype,
        )?;

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
        self.layers.embedding_layer.forward(&mut OpContext::new(&[&state.output_token], &mut [&mut x], cuda_config_ref))?;

        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            if i == 0 || !self.device_type.is_cuda() {
                self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;
            }

            let qkv_cols = self.config.q_dim + 2 * self.config.kv_dim;
            let qkv_buffer = state.workspace.get_mut(&BufferType::QkvOutput).unwrap();
            let mut qkv = qkv_buffer.slice(&[0, 0], &[1, qkv_cols])?;
            self.layers.wqkv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut qkv], cuda_config_ref))?;

            let mut q = qkv.slice(&[0, 0], &[1, self.config.q_dim])?;
            let mut k_view = qkv.slice(&[0, self.config.q_dim], &[1, self.config.kv_dim])?;
            let v_view = qkv.slice(&[0, self.config.q_dim + self.config.kv_dim], &[1, self.config.kv_dim])?;

            // QK-norm (Qwen3-specific)
            let mut q = if let Some(ref qnorm_layers) = self.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[self.config.head_num, self.config.head_size])?;
                let qnorm_buffer = state.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qnorm_out = qnorm_buffer.slice(&[0, 0], &[self.config.head_num, self.config.head_size])?;
                qnorm_layers[i].forward(&mut OpContext::new(&[&q_reshaped], &mut [&mut qnorm_out], cuda_config_ref))?;
                qnorm_out.reshape(&[1, self.config.q_dim])?
            } else {
                q
            };
            let mut k_active = if let Some(ref knorm_layers) = self.layers.knorm_layers {
                let k_reshaped = k_view.reshape(&[self.config.kv_head_num, self.config.head_size])?;
                let knorm_buffer = state.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut knorm_out = knorm_buffer.slice(&[0, 0], &[self.config.kv_head_num, self.config.head_size])?;
                knorm_layers[i].forward(&mut OpContext::new(&[&k_reshaped], &mut [&mut knorm_out], cuda_config_ref))?;
                knorm_out.reshape(&[1, self.config.kv_dim])?
            } else {
                k_view.reshape(&[1, self.config.kv_dim])?
            };

            let (k_cache_full, v_cache_full) = state.kv_cache.get_mut(i)?;
            let sin_cache = state.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = state.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&state.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k_active], cuda_config_ref))?;

            crate::op::kernels::scatter_kv(k_cache_full, &k_active, v_cache_full, &v_view, &state.input_pos, cuda_config_ref)?;

            let (k_hist, v_hist) = state.kv_cache.get(i).unwrap();
            let attn_out_buffer = state.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_out_buffer.slice(&[0, 0], &[1, self.config.q_dim])?;
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_hist, v_hist, &state.input_pos], &mut [&mut attn_out], cuda_config_ref))?;

            let wo_buffer = state.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut wo_out = wo_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;

            let mut ffn_norm_out = attn_norm_out;
            if self.device_type.is_cuda() {
                crate::op::kernels::fused_add_rmsnorm(
                    &mut ffn_norm_out, &mut x, &wo_out,
                    &self.layers.rmsnorm_ffn_layers[i].weight,
                    self.config.rms_norm_eps, cuda_config_ref,
                )?;
            } else {
                self.layers.add_layers.forward(&mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref))?;
                self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref))?;
            }

            let inter = self.config.intermediate_size;
            let gu_buffer = state.workspace.get_mut(&BufferType::GateUpOutput).unwrap();
            let mut gate_up = gu_buffer.slice(&[0, 0], &[1, 2 * inter])?;
            self.layers.w_gate_up_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut gate_up], cuda_config_ref))?;
            let mut w1_out = gate_up.slice(&[0, 0], &[1, inter])?;
            let w3_out = gate_up.slice(&[0, inter], &[1, inter])?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;

            if self.device_type.is_cuda() {
                let next_norm_weight = if i + 1 < self.config.layer_num {
                    &self.layers.rmsnorm_attn_layers[i + 1].weight
                } else {
                    &self.layers.rmsnorm_final_layer.weight
                };
                let buf = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
                let mut next_out = buf.slice(&[0, 0], &[1, self.config.dim])?;
                crate::op::kernels::fused_add_rmsnorm(
                    &mut next_out, &mut x, &w2_out,
                    next_norm_weight, self.config.rms_norm_eps, cuda_config_ref,
                )?;
            } else {
                self.layers.add_layers.forward(&mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref))?;
            }
        }

        let final_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        if !self.device_type.is_cuda() {
            self.layers.rmsnorm_final_layer.forward(&mut OpContext::new(&[&x], &mut [&mut final_norm_out], cuda_config_ref))?;
        }

        let logits = state.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
        self.layers.cls_layer.forward(&mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref))?;
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
        self.layers.embedding_layer.forward(&mut OpContext::new(&[&input_tokens_view], &mut [&mut x], cuda_config_ref))?;

        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;

            let qkv_cols = self.config.q_dim + 2 * self.config.kv_dim;
            let qkv_buffer = state.workspace.get_mut(&BufferType::QkvOutput).unwrap();
            let mut qkv = qkv_buffer.slice(&[0, 0], &[seq_len, qkv_cols])?;
            self.layers.wqkv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut qkv], cuda_config_ref))?;

            let q_buffer = state.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[seq_len, self.config.q_dim])?;
            let (mut k, mut v) = state.kv_cache.slice_kv_cache(i, pos as i32, seq_len, self.config.kv_dim)?;

            let stream = cuda_config_ref.map_or(std::ptr::null_mut(), |c| c.stream);
            crate::op::kernels::split_cols_tensor(&qkv, &mut q, seq_len, qkv_cols, 0, self.config.q_dim, stream)?;
            crate::op::kernels::split_cols_tensor(&qkv, &mut k, seq_len, qkv_cols, self.config.q_dim, self.config.kv_dim, stream)?;
            crate::op::kernels::split_cols_tensor(&qkv, &mut v, seq_len, qkv_cols, self.config.q_dim + self.config.kv_dim, self.config.kv_dim, stream)?;

            // QK-norm (Qwen3-specific)
            if let Some(ref qnorm_layers) = self.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[seq_len * self.config.head_num, self.config.head_size])?;
                let qnorm_buffer = state.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qnorm_out = qnorm_buffer.slice(&[0, 0], &[seq_len * self.config.head_num, self.config.head_size])?;
                qnorm_layers[i].forward(&mut OpContext::new(&[&q_reshaped], &mut [&mut qnorm_out], cuda_config_ref))?;
                q.copy_from(&qnorm_out.reshape(&[seq_len, self.config.q_dim])?)?;
            }
            if let Some(ref knorm_layers) = self.layers.knorm_layers {
                let k_reshaped = k.reshape(&[seq_len * self.config.kv_head_num, self.config.head_size])?;
                let knorm_buffer = state.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut knorm_out = knorm_buffer.slice(&[0, 0], &[seq_len * self.config.kv_head_num, self.config.head_size])?;
                knorm_layers[i].forward(&mut OpContext::new(&[&k_reshaped], &mut [&mut knorm_out], cuda_config_ref))?;
                k.copy_from(&knorm_out.reshape(&[seq_len, self.config.kv_dim])?)?;
            }

            let sin_cache = state.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = state.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&state.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k], cuda_config_ref))?;

            let (k_hist, v_hist) = state.kv_cache.get(i).unwrap();
            let attn_out_buffer = state.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_out_buffer.slice(&[0, 0], &[seq_len, self.config.q_dim])?;
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_hist, v_hist, pos_cpu], &mut [&mut attn_out], cuda_config_ref))?;

            let wo_buffer = state.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut wo_out = wo_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref))?;

            let ffn_norm_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut ffn_norm_out = ffn_norm_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref))?;

            let inter = self.config.intermediate_size;
            let gu_buffer = state.workspace.get_mut(&BufferType::GateUpOutput).unwrap();
            let mut gate_up = gu_buffer.slice(&[0, 0], &[seq_len, 2 * inter])?;
            self.layers.w_gate_up_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut gate_up], cuda_config_ref))?;

            let w1_buffer = state.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[seq_len, inter])?;
            let w3_buffer = state.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[seq_len, inter])?;

            let stream = cuda_config_ref.map_or(std::ptr::null_mut(), |c| c.stream);
            crate::op::kernels::split_cols_tensor(&gate_up, &mut w1_out, seq_len, 2 * inter, 0, inter, stream)?;
            crate::op::kernels::split_cols_tensor(&gate_up, &mut w3_out, seq_len, 2 * inter, inter, inter, stream)?;

            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;
            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref))?;
        }

        // Extract last token
        let last_hidden = x.slice(&[seq_len - 1, 0], &[1, self.config.dim])?;
        let buf = state.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
        let mut final_norm_input = buf.slice(&[0, 0], &[1, self.config.dim])?;
        final_norm_input.copy_from(&last_hidden)?;

        let final_norm_out_buffer = state.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.rmsnorm_final_layer.forward(&mut OpContext::new(&[&final_norm_input], &mut [&mut final_norm_out], cuda_config_ref))?;

        let logits = state.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
        self.layers.cls_layer.forward(&mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref))?;
        let logits_full = state.workspace.get(&BufferType::ForwardOutput).unwrap();
        let logits_ref = logits_full.slice(&[0], &[self.config.tokenizer_vocab_size])?;
        state.sampler.sample(&logits_ref, &mut state.output_token, cuda_config_ref)?;

        Ok(state.output_token.to_cpu()?.as_i32()?.as_slice()?[0])
    }
}

// ============================================================================
//  Model trait
// ============================================================================
use super::Model;

impl Model for Qwen3 {
    fn init(&mut self, _device_type: DeviceType) -> Result<()> {
        if let DeviceType::Cuda(device_id) = _device_type {
            self.layers.to_cuda(device_id)?;
        }
        Ok(())
    }

    fn forward(&mut self, _input: &Tensor, _pos: &Tensor) -> Result<Tensor> {
        Err(Error::InvalidArgument("forward not yet implemented for Qwen3".to_string()).into())
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    fn is_eos_token(&self, token_id: u32) -> bool {
        self.tokenizer.is_eos(token_id as i32)
    }

    fn slice_kv_cache(&self, _layer_idx: usize, _start_pos: usize, _end_pos: usize) -> Result<(Tensor, Tensor)> {
        Err(Error::InvalidArgument("slice_kv_cache not yet implemented for Qwen3".to_string()).into())
    }
}

// ============================================================================
//  Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;
    use crate::base::error::Result;

    fn generate_and_measure(
        model: &Qwen3, state: &mut InferenceState,
        prompt: &str, max_tokens: usize, verbose: bool,
    ) -> Result<(String, u64, u32, u64, u64, usize)> {
        let start = Instant::now();
        let (text, n_tok, prefill_ms, decode_ms, decode_iter) = model.generate(state, prompt, max_tokens, verbose)?;
        Ok((text, start.elapsed().as_millis() as u64, n_tok, prefill_ms, decode_ms, decode_iter))
    }

    fn get_qwen3_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/models/checkpoint-800-1")
    }

    #[test]
    #[ignore = "需要 Qwen3 模型权重，请单独运行。"]
    #[cfg(feature = "cuda")]
    fn test_qwen3_cuda_performance() -> Result<()> {
        let model_path = get_qwen3_model_path();
        assert!(model_path.exists(), "Qwen3 model directory not found at {:?}", model_path);

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n写一段C++代码，实现一个简单的中序遍历函数。<|im_end|>\n<|im_start|>assistant\n";

        let model = Qwen3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        let (_text, _dur, n_tok, prefill_ms, decode_ms, decode_iter) =
            generate_and_measure(&model, &mut state, prompt, 2000, true)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    #[test]
    #[ignore = "需要 Qwen3 模型权重，请单独运行。"]
    fn test_qwen3_cpu_loading_and_generation() -> Result<()> {
        let model_path = get_qwen3_model_path();
        assert!(model_path.exists(), "Qwen3 model directory not found at {:?}", model_path);

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n写一段C++代码，实现一个简单的中序遍历函数。<|im_end|>\n<|im_start|>assistant\n";

        let model = Qwen3::new(model_path, DeviceType::Cpu)?;
        let mut state = model.create_state()?;
        let (_text, _dur, n_tok, prefill_ms, decode_ms, decode_iter) =
            generate_and_measure(&model, &mut state, prompt, 150, true)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== CPU: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    fn get_qwen3_awq_model_path() -> &'static Path {
        Path::new("/data/home/vinciiliu/models/qwen3-4b-instruct-AWQ-mlp3")
    }

    #[test]
    #[ignore = "Long running test"]
    #[cfg(feature = "cuda")]
    fn test_qwen3_awq_cuda() -> Result<()> {
        let model_path = get_qwen3_awq_model_path();
        assert!(model_path.exists(), "Qwen3 AWQ model not found.");

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n";

        let model = Qwen3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        let (_text, _dur, n_tok, prefill_ms, decode_ms, decode_iter) =
            generate_and_measure(&model, &mut state, prompt, 2000, true)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (prefill_ms + decode_ms) as f64;
        println!("\n=== Qwen3 AWQ CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            n_tok, total_ms,
            (prompt_len + n_tok as f64) / (total_ms / 1000.0),
            if decode_ms > 0 { decode_iter as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }
}
