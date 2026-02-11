use std::path::Path;
use std::sync::Arc;

use crate::base::DeviceType;
use crate::base::error::Result;
use std::time::Instant;
use crate::cuda::CudaConfig;
use crate::op::{Op, OpContext};
use crate::op::attn_backend::AttentionBackend;
use crate::model::config::RuntimeModelConfig;
use crate::model::ModelLoader;
use crate::tensor::Tensor;
use crate::model::{Workspace, Model};
use crate::model::layers::DecoderLayers;
use crate::model::WeightMappingAdapter;

pub struct Llama3 {
    config: Arc<RuntimeModelConfig>,
    device_type: DeviceType,
    layers: DecoderLayers,
    workspace: Workspace,
    cuda_config: CudaConfig,
    attn_backend: AttentionBackend,
}

// ======================= Llama3 Weight Mapping =======================

/// Llama3 model's weight mapping implementation
///
/// Defines how Llama3 maps safetensor weight names to internal layers.
/// Llama3 uses the following naming convention:
/// - model.embed_tokens.weight
/// - model.layers.{layer}.self_attn.q_proj.weight
/// - model.layers.{layer}.mlp.gate_proj.weight
/// - model.norm.weight
/// - lm_head.weight
#[derive(Debug, Clone)]
struct Llama3WeightMapping;

impl WeightMappingAdapter for Llama3WeightMapping {
    fn embedding(&self) -> &'static str {
        "model.embed_tokens.weight"
    }

    fn rmsnorm_final(&self) -> &'static str {
        "model.norm.weight"
    }

    fn cls(&self) -> &'static str {
        "lm_head.weight"
    }

    fn format_layer_weight(&self, layer_idx: usize, weight_name: &str) -> String {
        format!("model.layers.{}.{}", layer_idx, weight_name)
    }

    fn attn_q(&self) -> &'static str {
        "self_attn.q_proj.weight"
    }

    fn attn_k(&self) -> &'static str {
        "self_attn.k_proj.weight"
    }

    fn attn_v(&self) -> &'static str {
        "self_attn.v_proj.weight"
    }

    fn attn_o(&self) -> &'static str {
        "self_attn.o_proj.weight"
    }

    fn ffn_gate(&self) -> &'static str {
        "mlp.gate_proj.weight"
    }

    fn ffn_up(&self) -> &'static str {
        "mlp.up_proj.weight"
    }

    fn ffn_down(&self) -> &'static str {
        "mlp.down_proj.weight"
    }

    fn rmsnorm_attn(&self) -> &'static str {
        "input_layernorm.weight"
    }

    fn rmsnorm_ffn(&self) -> &'static str {
        "post_attention_layernorm.weight"
    }
}

// ======================= Llama3 Implementation =======================

// Llama3 implementation

impl Llama3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
        max_batch_size: usize,
        block_size: usize,
    ) -> Result<Self> {
        let start_time = Instant::now();
        println!("Start calculate time, Loading Llama3 model from directory: {:?}", model_dir.as_ref());
        let loader = ModelLoader::load(model_dir.as_ref())?;
        let config = loader.config.clone();
        let cuda_config = CudaConfig::new()?;

        println!("Creating Llama3 decoder layers...");

        // Use DecoderLayers with Llama3 weight mapping
        let weight_mapping = Llama3WeightMapping;
        let layers = DecoderLayers::from_loader(
            &loader,
            &config,
            &weight_mapping,
            device_type,
        )?;

        // --- 初始化工作区 ---
        let workspace = Self::init_workspace(&config, &device_type, max_batch_size, block_size)?;

        // --- 初始化 Attention Backend (FlashInfer) ---
        let attn_backend = AttentionBackend::new(
            config.head_num as u32,
            config.kv_head_num as u32,
            config.head_size as u32,
            block_size as u32,
            max_batch_size as u32,
            config.seq_len as u32,
        )?;

        let mut model = Self {
            config: config.into(),
            device_type,
            layers,
            workspace,
            cuda_config,
            attn_backend,
        };
        println!("Calculating RoPE sin/cos cache...");
        Self::calculate_rope_cache(&model.config, &mut model.workspace)?;
        let duration = start_time.elapsed();
        println!("Model successfully initialized. Loading time: {:.2} seconds", duration.as_secs_f64());
        Ok(model)
    }

    fn calculate_rope_cache(
        config: &RuntimeModelConfig,
        workspace: &mut Workspace,
    ) -> Result<()> {
        let sin_cache = &mut workspace.sin_cache;
        let cos_cache = &mut workspace.cos_cache;

        match sin_cache.device() {
            DeviceType::Cpu => {
                crate::op::kernels::cpu::rope_sin_cos_cache_calc(
                    config.head_size,
                    config.seq_len,
                    500000.0, // Llama3 rope_base
                    sin_cache,
                    cos_cache,
                )?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                let stream = crate::cuda::config::CudaConfig::new()?;
                crate::op::kernels::cuda::rope_sin_cos_cache_calc_cuda(
                    config.head_size,
                    config.seq_len,
                    500000.0, // Llama3 rope_base
                    sin_cache,
                    cos_cache,
                    Some(&stream),
                )?;
            }
        }
        Ok(())
    }

    /// 预分配前向传播所需的所有中间张量。
    ///
    /// # Arguments
    /// * `config` - 模型运行时配置
    /// * `device` - 目标设备
    /// * `max_batch_size` - 最大批处理大小
    /// * `block_size` - KV Cache 块大小
    fn init_workspace(
        config: &RuntimeModelConfig,
        device: &DeviceType,
        max_batch_size: usize,
        block_size: usize,
    ) -> Result<Workspace> {
        let max_blocks_per_req = (config.seq_len + block_size - 1) / block_size;
        Workspace::new(config, *device, max_batch_size, max_blocks_per_req)
    }

    /// Decode forward: 批量处理，每个请求 1 个 token
    ///
    /// Supports CudaGraph capture/replay: on the first call for a given bucket
    /// size the full pipeline is captured into a graph; subsequent calls with
    /// the same bucket replay the captured graph.
    ///
    /// # Arguments
    /// * `input_tokens` - [batch_size] token ids
    /// * `positions` - [batch_size] position ids
    /// * `kv_indices` - [nnz_tokens] 历史 KV cache 的物理槽位索引 (CSR-style)
    /// * `kv_indptr` - [batch_size + 1] CSR row pointers, kv_len[i] = kv_indptr[i+1] - kv_indptr[i]
    /// * `new_slots` - [batch_size] 写入新 K/V 的物理槽位
    /// * `kv_cache` - KVCachePool 用于存储K/V
    fn forward_decoding(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        kv_indices: &Tensor,
        kv_indptr: &Tensor,
        new_slots: &Tensor,
        kv_cache: &mut crate::model::KVCachePool,
    ) -> Result<()> {
        let batch_size = input_tokens.shape()[0];

        // ---- CudaGraph: find bucket and check for cached graph ----
        let bucket = self.workspace.find_bucket(batch_size)
            .ok_or_else(|| crate::base::error::Error::InvalidArgument(
                format!("batch_size {} exceeds max bucket", batch_size)
            ))?;

        if let Some(graph) = self.workspace.get_graph(bucket) {
            graph.launch(self.cuda_config.stream)?;
            self.cuda_config.sync_stream()?;
            return Ok(());
        }

        // ---- CudaGraph: begin capture ----
        self.cuda_config.capture_graph_begin()?;

        // Run the full forward pipeline (will be captured into graph)
        self.forward_pipeline(input_tokens, positions, kv_indices, kv_indptr, new_slots, kv_cache, batch_size)?;

        // ---- CudaGraph: end capture and store graph ----
        let graph = self.cuda_config.capture_graph_end()?;
        self.workspace.insert_graph(bucket, graph);

        Ok(())
    }

    /// The actual forward computation pipeline, factored out so that
    /// `forward_decoding` can wrap it with CudaGraph capture/replay.
    fn forward_pipeline(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        kv_indices: &Tensor,
        kv_indptr: &Tensor,
        new_slots: &Tensor,
        kv_cache: &mut crate::model::KVCachePool,
        batch_size: usize,
    ) -> Result<()> {
        let cuda_config = Some(&self.cuda_config);

        // Step 1: Token Embedding [batch_size] -> [batch_size, dim]
        let mut hidden = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;

        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[input_tokens], &mut [&mut hidden], cuda_config)
        )?;

        // Step 2: Transformer Layers
        for i in 0..self.config.layer_num {
            // 2.1 Pre-Attention RMSNorm: attn_output → rms_output
            let mut norm_out = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let hidden_view = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(
                &mut OpContext::new(&[&hidden_view], &mut [&mut norm_out], cuda_config)
            )?;

            // 2.2 Fused QKV Projection: rms_output → qkv_output
            let mut qkv = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;
            let norm_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.wqkv_layers[i].forward(
                &mut OpContext::new(&[&norm_view], &mut [&mut qkv], cuda_config)
            )?;

            // 2.3 RoPE on Q and K (in-place on qkv_output)
            let qkv_mut = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;
            let mut q_mut = qkv_mut.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut k_mut = qkv_mut.slice(&[0, self.config.dim], &[batch_size, self.config.dim + self.config.kv_dim])?;

            self.layers.rope_layers[i].forward(
                &mut OpContext::new(
                    &[positions, &self.workspace.sin_cache, &self.workspace.cos_cache],
                    &mut [&mut q_mut, &mut k_mut],
                    cuda_config
                )
            )?;

            // 2.4 Scatter KV: write new K,V to paged KV cache via new_slots
            let qkv_view = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;
            let k_updated = qkv_view.slice(&[0, self.config.dim], &[batch_size, self.config.dim + self.config.kv_dim])?;
            let v_updated = qkv_view.slice(&[0, self.config.dim + self.config.kv_dim], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;

            let (k_cache_layer, v_cache_layer) = kv_cache.get_mut(i)?;
            self.layers.scatter_layer.forward(
                &mut OpContext::new(&[&k_updated, &v_updated, new_slots], &mut [k_cache_layer, v_cache_layer], cuda_config)
            )?;

            // 2.5 Flash Attention GQA: Q @ K^T -> softmax -> @ V
            let q_view = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let (k_cache_ref, v_cache_ref) = kv_cache.get_mut(i)?;
            let mut attn_out = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mha_inputs = [&q_view as &Tensor, k_cache_ref as &Tensor, v_cache_ref as &Tensor, kv_indices, kv_indptr];
            let mut mha_outputs = [&mut attn_out];
            let mut mha_ctx = OpContext::new(
                &mha_inputs,
                &mut mha_outputs,
                cuda_config
            );
            mha_ctx.attn_backend = Some(&self.attn_backend);
            self.layers.mha_layers[i].forward(&mut mha_ctx)?;

            // 2.6 Wo projection: rms_output → qkv_output[:, :dim] (reuse as temp)
            let attn_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut wo_out = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.wo_layers[i].forward(
                &mut OpContext::new(&[&attn_view], &mut [&mut wo_out], cuda_config)
            )?;

            // 2.7 Residual Add: attn_output += wo_out
            let wo_view = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut hidden_mut = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&wo_view], &mut [&mut hidden_mut], cuda_config)
            )?;

            // 2.8 FFN RMSNorm: attn_output → rms_output
            let hidden_view = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut ffn_norm = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.rmsnorm_ffn_layers[i].forward(
                &mut OpContext::new(&[&hidden_view], &mut [&mut ffn_norm], cuda_config)
            )?;

            // 2.9 W1 (gate proj): rms_output → w1_output
            let ffn_norm_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut w1_out = self.workspace.w1_output.slice(&[0, 0], &[batch_size, self.config.intermediate_size])?;
            self.layers.w1_layers[i].forward(
                &mut OpContext::new(&[&ffn_norm_view], &mut [&mut w1_out], cuda_config)
            )?;

            // 2.10 W3 (up proj): rms_output → w3_output
            let ffn_norm_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut w3_out = self.workspace.w3_output.slice(&[0, 0], &[batch_size, self.config.intermediate_size])?;
            self.layers.w3_layers[i].forward(
                &mut OpContext::new(&[&ffn_norm_view], &mut [&mut w3_out], cuda_config)
            )?;

            // 2.11 SwiGLU: silu(w1_output) * w3_output → w1_output (in-place)
            let w3_view = self.workspace.w3_output.slice(&[0, 0], &[batch_size, self.config.intermediate_size])?;
            let mut w1_mut = self.workspace.w1_output.slice(&[0, 0], &[batch_size, self.config.intermediate_size])?;
            self.layers.swiglu_layers[i].forward(
                &mut OpContext::new(&[&w3_view], &mut [&mut w1_mut], cuda_config)
            )?;

            // 2.12 W2 (down proj): w1_output → rms_output (reuse as temp)
            let w1_view = self.workspace.w1_output.slice(&[0, 0], &[batch_size, self.config.intermediate_size])?;
            let mut down_out = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.w2_layers[i].forward(
                &mut OpContext::new(&[&w1_view], &mut [&mut down_out], cuda_config)
            )?;

            // 2.13 Residual Add: attn_output += rms_output (down_out)
            let down_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut hidden_mut = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&down_view], &mut [&mut hidden_mut], cuda_config)
            )?;
        }

        // Step 3: Final RMSNorm: attn_output → rms_output
        let hidden_view = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
        let mut final_norm = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
        self.layers.rmsnorm_final_layer.forward(
            &mut OpContext::new(&[&hidden_view], &mut [&mut final_norm], cuda_config)
        )?;

        // Step 4: LM Head (cls): rms_output → forward_output [batch_size, vocab_size]
        let norm_view = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
        let mut logits = self.workspace.forward_output.slice(&[0, 0], &[batch_size, self.config.vocab_size])?;
        self.layers.cls_layer.forward(
            &mut OpContext::new(&[&norm_view], &mut [&mut logits], cuda_config)
        )?;

        Ok(())
    }

    /// Prefill forward: 处理变长输入序列
    fn forward_prefill(
        &mut self,
        _input_tokens: &Tensor,
        _positions: &Tensor,
        _slot_mapping: &Tensor,
    ) -> Result<()> {
        // TODO: Prefill 需要单独的 workspace（支持变长序列）
        // 当前设计是 decode-only
        Ok(())
    }

    /// Get the device type used by this model
    pub fn device(&self) -> DeviceType {
        self.device_type
    }
}

// ============================================================================ //  Model Trait Implementation // ============================================================================

impl Model for Llama3 {
    fn config(&self) -> &RuntimeModelConfig {
        &self.config
    }

    fn forward_paged(
            &mut self,
            input_tokens_cpu: &Tensor,      // CPU Tensor [total_tokens]
            positions_cpu: &Tensor,          // CPU Tensor [total_tokens]
            kv_indices_cpu: &Tensor,         // CPU Tensor [nnz_tokens] CSR-style slot indices
            kv_indptr_cpu: &Tensor,          // CPU Tensor [batch_size + 1] CSR row pointers
            new_slots_cpu: &Tensor,          // CPU Tensor [total_tokens] slots for new K/V
            decode_tokens: usize,
            kv_cache: &mut crate::model::KVCachePool,
        ) -> Result<Tensor> {
        let total_tokens = input_tokens_cpu.shape()[0];
        let prefill_tokens = total_tokens - decode_tokens;
        let batch_size = kv_indptr_cpu.shape()[0] - 1;  // kv_indptr has batch_size + 1 elements
        let nnz_tokens = kv_indices_cpu.shape()[0];

        // ======================= CPU -> Device Copy =======================
        // 拷贝到静态 buffer（CudaGraph 绑定地址）
        {
            let mut input_dst = self.workspace.static_input_ids().slice(&[0], &[total_tokens])?;
            input_dst.copy_from(input_tokens_cpu)?;
        }
        {
            let mut pos_dst = self.workspace.static_positions().slice(&[0], &[total_tokens])?;
            pos_dst.copy_from(positions_cpu)?;
        }
        {
            let mut slots_dst = self.workspace.static_slot_mapping().slice(&[0], &[total_tokens])?;
            slots_dst.copy_from(new_slots_cpu)?;
        }
        // Copy kv_indices and kv_indptr to static buffers (fixed addresses for CudaGraph)
        {
            let mut kv_idx_dst = self.workspace.static_kv_indices().slice(&[0], &[nnz_tokens])?;
            kv_idx_dst.copy_from(kv_indices_cpu)?;
        }
        {
            let indptr_len = kv_indptr_cpu.shape()[0]; // batch_size + 1
            let mut kv_indptr_dst = self.workspace.static_kv_indptr().slice(&[0], &[indptr_len])?;
            kv_indptr_dst.copy_from(kv_indptr_cpu)?;
        }

        // 获取 device tensor 引用
        let input_tokens = self.workspace.static_input_ids().slice(&[0], &[total_tokens])?;
        let positions = self.workspace.static_positions().slice(&[0], &[total_tokens])?;
        let new_slots = self.workspace.static_slot_mapping().slice(&[0], &[total_tokens])?;
        let kv_indices = self.workspace.static_kv_indices().slice(&[0], &[nnz_tokens])?;
        let kv_indptr = self.workspace.static_kv_indptr().slice(&[0], &[kv_indptr_cpu.shape()[0]])?;

        // ======================= Plan Attention Backend =======================
        // Convert token-level kv_indptr to page-level metadata for FlashInfer
        {
            let indptr_host = kv_indptr_cpu.as_i32()?.as_slice()?;
            let stream = self.cuda_config.stream;
            self.attn_backend.plan(indptr_host, batch_size as u32, false, stream)?;
        }

        // ======================= Phase 1: Decode =======================
        if decode_tokens > 0 {
            let decode_input = input_tokens.slice(&[0], &[decode_tokens])?;
            let decode_pos = positions.slice(&[0], &[decode_tokens])?;
            let decode_new_slots = new_slots.slice(&[0], &[decode_tokens])?;

            self.forward_decoding(&decode_input, &decode_pos, &kv_indices, &kv_indptr, &decode_new_slots, kv_cache)?;
        }

        // ======================= Phase 2: Prefill =======================
        if prefill_tokens > 0 {
            let prefill_input = input_tokens.slice(&[decode_tokens], &[total_tokens])?;
            let prefill_pos = positions.slice(&[decode_tokens], &[total_tokens])?;
            let prefill_new_slots = new_slots.slice(&[decode_tokens], &[total_tokens])?;

            self.forward_prefill(&prefill_input, &prefill_pos, &prefill_new_slots)?;
        }

        // 返回 logits [batch_size, vocab_size] from forward_output
        let output = self.workspace.forward_output.slice(&[0, 0], &[batch_size, self.config.vocab_size])?;
        Ok(output)
    }

    fn device_type(&self) -> DeviceType {
        self.device_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;
    use crate::base::DataType;
    use crate::model::{KVCachePool, KVCacheConfig, Model};
    use crate::op::sampler::{ArgmaxSampler, Sampler};

    fn get_dummy_model_path() -> &'static Path {
        Path::new("/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct")
    }

    /// Autoregressive decode loop that measures prefill and decode performance.
    ///
    /// Returns (generated_text, num_generated, prefill_ms, decode_ms, decode_iterations).
    fn generate_and_measure(
        model: &mut Llama3,
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<(String, usize, f64, f64, usize)> {
        let device = model.device();

        // Extract config values upfront to avoid borrow conflicts with forward_paged
        let layer_num = model.config().layer_num;
        let kv_head_num = model.config().kv_head_num;
        let head_size = model.config().head_size;
        let vocab_size = model.config().vocab_size;

        // --- Setup ---
        let kv_config = KVCacheConfig::new(
            layer_num,
            kv_head_num,
            head_size,
            DataType::BF16,
            16,    // block_size
            1000,  // num_blocks
        );
        let mut kv_cache = KVCachePool::new(kv_config, device)?;

        let sampler = ArgmaxSampler::new(device);
        let mut output_token_ids = Tensor::new(&[1], DataType::I32, device)?;

        // Encode prompt
        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| crate::base::error::Error::InvalidArgument(format!("Tokenizer error: {}", e)))?;
        let prompt_token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let prompt_len = prompt_token_ids.len();
        println!("Prompt tokens: {} tokens", prompt_len);

        // --- Prefill phase (process prompt tokens one-by-one in decode mode) ---
        let prefill_start = Instant::now();
        for t in 0..prompt_len {
            let input_tokens = Tensor::from_slice(&[prompt_token_ids[t]], DeviceType::Cpu)?;
            let positions = Tensor::from_slice(&[t as i32], DeviceType::Cpu)?;

            // CSR: kv_indices = [0..t-1], kv_indptr = [0, t]
            let kv_indices_data: Vec<i32> = (0..t as i32).collect();
            let kv_indices = if kv_indices_data.is_empty() {
                Tensor::from_slice(&[0i32], DeviceType::Cpu)? // dummy, won't be used when t=0
            } else {
                Tensor::from_slice(&kv_indices_data, DeviceType::Cpu)?
            };
            let kv_indptr = Tensor::from_slice(&[0i32, t as i32], DeviceType::Cpu)?;
            let new_slots = Tensor::from_slice(&[t as i32], DeviceType::Cpu)?;

            let _logits = model.forward_paged(
                &input_tokens,
                &positions,
                &kv_indices,
                &kv_indptr,
                &new_slots,
                1, // decode_tokens
                &mut kv_cache,
            )?;
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Sample first token from last prefill logits
        let logits_view = model.workspace.forward_output.slice(
            &[0, 0],
            &[1, vocab_size],
        )?;
        sampler.sample(&logits_view, &mut output_token_ids, Some(&model.cuda_config))?;

        let first_token_cpu = output_token_ids.to_cpu()?;
        let first_token = first_token_cpu.as_i32()?.as_slice()?[0];

        let mut generated_ids: Vec<u32> = vec![first_token as u32];
        let mut current_pos = prompt_len;

        // --- Decode phase ---
        let decode_start = Instant::now();
        let mut decode_iterations = 0;
        let eos_token_id: i32 = 128009; // Llama3 EOS

        loop {
            if generated_ids.len() >= max_tokens {
                break;
            }
            let last_token = *generated_ids.last().unwrap() as i32;
            if last_token == eos_token_id {
                break;
            }

            let input_tokens = Tensor::from_slice(&[last_token], DeviceType::Cpu)?;
            let positions = Tensor::from_slice(&[current_pos as i32], DeviceType::Cpu)?;

            // CSR: all past slots [0..current_pos-1], one request
            let kv_indices_data: Vec<i32> = (0..current_pos as i32).collect();
            let kv_indices = Tensor::from_slice(&kv_indices_data, DeviceType::Cpu)?;
            let kv_indptr = Tensor::from_slice(&[0i32, current_pos as i32], DeviceType::Cpu)?;
            let new_slots = Tensor::from_slice(&[current_pos as i32], DeviceType::Cpu)?;

            let _logits = model.forward_paged(
                &input_tokens,
                &positions,
                &kv_indices,
                &kv_indptr,
                &new_slots,
                1,
                &mut kv_cache,
            )?;

            // Sample next token
            let logits_view = model.workspace.forward_output.slice(
                &[0, 0],
                &[1, vocab_size],
            )?;
            sampler.sample(&logits_view, &mut output_token_ids, Some(&model.cuda_config))?;

            let token_cpu = output_token_ids.to_cpu()?;
            let token_id = token_cpu.as_i32()?.as_slice()?[0];

            generated_ids.push(token_id as u32);
            current_pos += 1;
            decode_iterations += 1;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        // --- Detokenize ---
        let text = tokenizer.decode(&generated_ids, true)
            .map_err(|e| crate::base::error::Error::InvalidArgument(format!("Decode error: {}", e)))?;

        Ok((text, generated_ids.len(), prefill_ms, decode_ms, decode_iterations))
    }

    #[test]
    fn test_llama3_cuda_forward() -> Result<()> {
        let model_path = get_dummy_model_path();
        if !model_path.exists() {
            eprintln!("Skipping test: model not found at {:?}", model_path);
            return Ok(());
        }

        // Load model
        println!("=== Loading Llama3 model ===");
        let mut model = Llama3::new(model_path, DeviceType::Cuda(0), 128, 16)?;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| crate::base::error::Error::InvalidArgument(format!("Tokenizer error: {}", e)))?;

        let prompt = "Write a quick sort algorithm in C++.";
        let max_tokens = 512;

        println!("=== Generating (prompt: \"{}\") ===", prompt);
        let (text, num_generated, prefill_ms, decode_ms, decode_iters) =
            generate_and_measure(&mut model, &tokenizer, prompt, max_tokens)?;

        // --- Print results ---
        println!("\n========== Generation Results ==========");
        println!("Generated text:\n{}", text);
        println!("=========================================");
        println!("Tokens generated: {}", num_generated);

        // Prefill metrics
        let prompt_encoding = tokenizer.encode(prompt, true)
            .map_err(|e| crate::base::error::Error::InvalidArgument(format!("{}", e)))?;
        let prompt_tokens = prompt_encoding.get_ids().len();
        println!("Prefill: {} tokens in {:.2} ms (TTFT)", prompt_tokens, prefill_ms);
        if prefill_ms > 0.0 {
            println!("  Prefill throughput: {:.1} tokens/s", prompt_tokens as f64 / prefill_ms * 1000.0);
        }

        // Decode metrics
        if decode_iters > 0 {
            let itl = decode_ms / decode_iters as f64;
            let decode_tps = decode_iters as f64 / decode_ms * 1000.0;
            println!("Decode: {} tokens in {:.2} ms", decode_iters, decode_ms);
            println!("  ITL (inter-token latency): {:.2} ms", itl);
            println!("  Decode throughput: {:.1} tokens/s", decode_tps);
        }

        // Total
        let total_ms = prefill_ms + decode_ms;
        let total_tokens = prompt_tokens + num_generated;
        if total_ms > 0.0 {
            println!("Total: {} tokens in {:.2} ms ({:.1} tokens/s)",
                total_tokens, total_ms, total_tokens as f64 / total_ms * 1000.0);
        }

        // Sanity check
        assert!(num_generated >= 1, "Should generate at least 1 token");

        Ok(())
    }
}