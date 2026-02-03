use std::path::Path;
use std::sync::Arc;

use crate::base::DeviceType;
use crate::base::error::Result;
use std::time::Instant;
use crate::cuda::CudaConfig;
use crate::op::{Op, OpContext};
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

        let mut model = Self {
            config: config.into(),
            device_type,
            layers,
            workspace,
            cuda_config
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
        let cuda_config = Some(&self.cuda_config);

        // Step 1: Token Embedding [batch_size] -> [batch_size, dim]
        let mut hidden = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
        let mut qkv;
        let q;
        let k;
        let v;

        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[input_tokens], &mut [&mut hidden], cuda_config)
        )?;

        // Step 2: Transformer Layers
        for i in 0..self.config.layer_num {
            // 2.1 Pre-Attention RMSNorm
            let mut norm_out = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(
                &mut OpContext::new(&[&hidden], &mut [&mut norm_out], cuda_config)
            )?;

            // 2.2 Fused QKV Projection (single matmul produces Q, K, V concatenated)
            // Output shape: [batch, dim + kv_dim + kv_dim]
            qkv = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;
            self.layers.wqkv_layers[i].forward(
                &mut OpContext::new(&[&norm_out], &mut [&mut qkv], cuda_config)
            )?;

            // Split fused output into Q, K, V
            q = qkv.slice(&[0, 0], &[batch_size, self.config.dim])?;
            k = qkv.slice(&[0, self.config.dim], &[batch_size, self.config.dim + self.config.kv_dim])?;
            v = qkv.slice(&[0, self.config.dim + self.config.kv_dim], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;

            // 2.3 RoPE on Q and K
            // 需要可变的Q和K切片来进行旋转变换
            let mut qkv_mut = self.workspace.qkv_output.slice(&[0, 0], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;
            let mut q_mut = qkv_mut.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut k_mut = qkv_mut.slice(&[0, self.config.dim], &[batch_size, self.config.dim + self.config.kv_dim])?;

            self.layers.rope_layers[i].forward(
                &mut OpContext::new(
                    &[positions, &self.workspace.sin_cache, &self.workspace.cos_cache],
                    &mut [&mut q_mut, &mut k_mut],
                    cuda_config
                )
            )?;

            // 2.4 Write K,V to paged KV cache via new_slots
            // 从qkv_output中重新提取K和V（已经过RoPE）
            let k_updated = qkv_mut.slice(&[0, self.config.dim], &[batch_size, self.config.dim + self.config.kv_dim])?;
            let v_updated = qkv_mut.slice(&[0, self.config.dim + self.config.kv_dim], &[batch_size, self.config.dim + 2 * self.config.kv_dim])?;

            self.layers.scatter_layer.forward(&mut OpContext::new(&[&k_updated, &v_updated, &new_slots], &mut [&mut self.workspace.key, &mut self.workspace.value], cuda_config))?;

            // 2.5 Paged Attention: Q @ K^T -> softmax -> @ V
            // 使用 CSR-style 的 kv_indices 和 kv_indptr 进行 attention 计算
            // TODO: flash_attn_gqa_paged(&q, kv_cache, kv_indices, kv_indptr, &mut attn_out)
            let mut attn_out = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;

            // 2.6 Output Projection
            let mut wo_out = self.workspace.query.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.wo_layers[i].forward(
                &mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config)
            )?;

            // 2.7 Residual Add: hidden = hidden + wo_out
            let mut hidden_mut = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&wo_out], &mut [&mut hidden_mut], cuda_config)
            )?;

            // 2.8 FFN Block
            let hidden_view = self.workspace.attn_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            let mut ffn_norm = self.workspace.rms_output.slice(&[0, 0], &[batch_size, self.config.dim])?;
            self.layers.rmsnorm_ffn_layers[i].forward(
                &mut OpContext::new(&[&hidden_view], &mut [&mut ffn_norm], cuda_config)
            )?;

            // TODO: FFN 需要额外的 buffer (intermediate_size)
            // Gate: W1 @ ffn_norm -> [batch_size, intermediate_size]
            // Up:   W3 @ ffn_norm -> [batch_size, intermediate_size]
            // SiLU: silu(gate) * up
            // Down: W2 @ silu_out -> [batch_size, dim]
            // Residual: hidden = hidden + down_out
        }

        // Step 3: Final RMSNorm + LM Head
        // TODO: final norm -> lm_head -> logits [batch_size, vocab_size]

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

        // 获取 device tensor 引用
        let input_tokens = self.workspace.static_input_ids().slice(&[0], &[total_tokens])?;
        let positions = self.workspace.static_positions().slice(&[0], &[total_tokens])?;
        let new_slots = self.workspace.static_slot_mapping().slice(&[0], &[total_tokens])?;

        // TODO: 需要在 Workspace 中添加 kv_indices 和 kv_indptr 的静态 buffer
        // 暂时直接使用 CPU tensor（需要后续优化）
        let kv_indices = kv_indices_cpu;
        let kv_indptr = kv_indptr_cpu;

        // ======================= Phase 1: Decode =======================
        if decode_tokens > 0 {
            let decode_input = input_tokens.slice(&[0], &[decode_tokens])?;
            let decode_pos = positions.slice(&[0], &[decode_tokens])?;
            let decode_new_slots = new_slots.slice(&[0], &[decode_tokens])?;

            self.forward_decoding(&decode_input, &decode_pos, kv_indices, kv_indptr, &decode_new_slots, kv_cache)?;
        }

        // ======================= Phase 2: Prefill =======================
        if prefill_tokens > 0 {
            let prefill_input = input_tokens.slice(&[decode_tokens], &[total_tokens])?;
            let prefill_pos = positions.slice(&[decode_tokens], &[total_tokens])?;
            let prefill_new_slots = new_slots.slice(&[decode_tokens], &[total_tokens])?;

            self.forward_prefill(&prefill_input, &prefill_pos, &prefill_new_slots)?;
        }

        // 返回 logits [batch_size, vocab_size]
        let output = self.workspace.static_output().slice(&[0, 0], &[batch_size, self.config.vocab_size])?;
        Ok(output)
    }

    fn device_type(&self) -> DeviceType {
        self.device_type
    }
}