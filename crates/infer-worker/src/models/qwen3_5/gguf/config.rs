use super::*;
use crate::infrastructure::io::gguf::{GgufArray, GgufMetadata, GgufValue};

fn invalid(key: &str) -> OpError {
    OpError::Shape(format!("GGUF metadata {key}: missing or invalid value"))
}

fn integer(m: &GgufMetadata, key: &str) -> OpResult<usize> {
    let v = match m.get(key) {
        Some(GgufValue::U32(v)) => u64::from(*v),
        Some(GgufValue::U64(v)) => *v,
        _ => return Err(invalid(key)),
    };
    usize::try_from(v).map_err(|_| invalid(key))
}

fn positive(m: &GgufMetadata, suffix: &str) -> OpResult<usize> {
    let key = format!("qwen35.{suffix}");
    let n = integer(m, &key)?;
    if n == 0 || n > i32::MAX as usize {
        return Err(invalid(&key));
    }
    Ok(n)
}

fn float(m: &GgufMetadata, suffix: &str) -> OpResult<f64> {
    let key = format!("qwen35.{suffix}");
    let v = match m.get(&key) {
        Some(GgufValue::F32(v)) => f64::from(*v),
        Some(GgufValue::F64(v)) => *v,
        _ => return Err(invalid(&key)),
    };
    if !v.is_finite() || v <= 0.0 {
        return Err(invalid(&key));
    }
    Ok(v)
}

pub(super) fn parse(reader: &GgufReader, options: LoadOptions) -> OpResult<ModelConfig> {
    let m = reader.metadata();
    if m.get("general.architecture").and_then(GgufValue::as_str) != Some("qwen35") {
        return Err(OpError::unsupported(
            "GGUF model loader",
            "expected architecture qwen35",
        ));
    }
    if m.get("split.count").is_some_and(|v| v.as_u16() != Some(1)) {
        return Err(OpError::unsupported("GGUF model loader", "split files"));
    }
    if m.get("qwen35.rope.scaling.type")
        .is_some_and(|v| v.as_str() != Some("none"))
    {
        return Err(OpError::unsupported("GGUF model loader", "RoPE scaling"));
    }
    for key in ["qwen35.expert_count", "qwen35.expert_used_count"] {
        if m.get(key).is_some() && integer(m, key)? != 0 {
            return Err(OpError::unsupported("GGUF model loader", "MoE"));
        }
    }
    let blocks = positive(m, "block_count")?;
    let mtp = if m.get("qwen35.nextn_predict_layers").is_some() {
        integer(m, "qwen35.nextn_predict_layers")?
    } else {
        0
    };
    let layers = blocks
        .checked_sub(mtp)
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("qwen35.nextn_predict_layers"))?;
    // Every block must have at least one tensor; bound metadata-driven allocation.
    if blocks > reader.tensors().len() {
        return Err(invalid("qwen35.block_count"));
    }
    let dim = positive(m, "embedding_length")?;
    let ffn = positive(m, "feed_forward_length")?;
    let heads = positive(m, "attention.head_count")?;
    let kv_heads = positive(m, "attention.head_count_kv")?;
    let head_dim = positive(m, "attention.key_length")?;
    if positive(m, "attention.value_length")? != head_dim || !heads.is_multiple_of(kv_heads) {
        return Err(invalid("qwen35.attention head geometry"));
    }
    let rotary_dim = positive(m, "rope.dimension_count")?;
    if rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(invalid("qwen35.rope.dimension_count"));
    }
    let sections = match m.get("qwen35.rope.dimension_sections") {
        Some(GgufValue::Array(GgufArray::I32(v)))
            if v.len() == 4 && v[3] == 0 && v.iter().all(|x| *x >= 0) =>
        {
            [v[0] as usize, v[1] as usize, v[2] as usize]
        }
        Some(GgufValue::Array(GgufArray::U32(v))) if v.len() == 4 && v[3] == 0 => {
            [v[0] as usize, v[1] as usize, v[2] as usize]
        }
        _ => return Err(invalid("qwen35.rope.dimension_sections")),
    };
    if sections.iter().sum::<usize>().checked_mul(2) != Some(rotary_dim) {
        return Err(invalid("qwen35.rope.dimension_sections"));
    }
    let context = positive(m, "context_length")?;
    if options.context_length == 0 || options.context_length > context {
        return Err(OpError::Shape(format!(
            "GGUF requested context {} must be in 1..={context}",
            options.context_length
        )));
    }
    options
        .context_length
        .checked_mul(rotary_dim)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("RoPE cache size"))?;
    let key_heads = positive(m, "ssm.group_count")?;
    let value_heads = positive(m, "ssm.time_step_rank")?;
    let inner = positive(m, "ssm.inner_size")?;
    if !inner.is_multiple_of(value_heads) {
        return Err(invalid("qwen35.ssm.inner_size"));
    }
    let linear = LinearDims {
        num_key_heads: key_heads,
        num_value_heads: value_heads,
        key_head_dim: positive(m, "ssm.state_size")?,
        value_head_dim: inner / value_heads,
        conv_kernel_dim: positive(m, "ssm.conv_kernel")?,
    };
    linear.validate()?;
    if linear.key_head_dim > 1024 {
        return Err(OpError::unsupported(
            "GGUF model loader",
            "GDN key dimension > 1024",
        ));
    }
    let layer_is_full = if let Some(v) = m.get("qwen35.attention.recurrent_layers") {
        match v {
            GgufValue::Array(GgufArray::Bool(v))
                if v.len() == blocks && v[layers..].iter().all(|&x| !x) =>
            {
                v[..layers].iter().map(|&v| !v).collect()
            }
            _ => return Err(invalid("qwen35.attention.recurrent_layers")),
        }
    } else {
        let interval = positive(m, "full_attention_interval")?;
        (0..layers).map(|i| (i + 1) % interval == 0).collect()
    };
    let emb = reader
        .tensor_info("token_embd.weight")
        .ok_or_else(|| OpError::Shape("GGUF missing token_embd.weight".into()))?;
    let vocab = match emb.dimensions() {
        [k, n] if *k == dim as u64 && *n <= i32::MAX as u64 => *n as usize,
        _ => {
            return Err(OpError::Shape(
                "GGUF invalid token_embd.weight shape".into(),
            ));
        }
    };
    if let Some(v) = m.get("tokenizer.ggml.tokens")
        && !matches!(v, GgufValue::Array(GgufArray::String(v)) if v.len() == vocab)
    {
        return Err(invalid("tokenizer.ggml.tokens"));
    }
    let q = heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Q width"))?;
    let kv = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("KV width"))?;
    let qkv = kv
        .checked_mul(2)
        .and_then(|n| n.checked_add(q))
        .ok_or_else(|| invalid("QKV width"))?;
    q.checked_mul(2)
        .and_then(|n| n.checked_add(2 * kv))
        .ok_or_else(|| invalid("gated QKV width"))?;
    let dims = ModelDims {
        dim,
        intermediate_size: ffn,
        head_num: heads,
        kv_head_num: kv_heads,
        head_dim,
        q_dim: q,
        kv_dim: kv,
        qkv_dim: qkv,
        vocab_size: vocab,
        num_layers: layers,
        ..ModelDims::default()
    };
    dims.validate()?;
    let eps = float(m, "attention.layer_norm_rms_epsilon")? as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("RMS epsilon"));
    }
    Ok(ModelConfig {
        dims,
        linear,
        layer_is_full,
        context_length: options.context_length,
        trained_context_length: context,
        rotary_dim,
        rope_theta: float(m, "rope.freq_base")?,
        mrope_sections: sections,
        rms_norm_eps: eps,
        mtp_layers: mtp,
    })
}
