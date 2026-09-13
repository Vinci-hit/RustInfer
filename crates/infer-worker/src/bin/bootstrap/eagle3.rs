//! Launch-time I/O and pairing policy for the optional Qwen3 draft head.
use half::bf16;
use infer_protocol::config::SpeculativeConfig;
use infer_worker::{
    application::speculative::serving::ConditionedServing,
    components::eagle3::Eagle3DraftHead,
    domain::model::DecoderModel,
    infrastructure::{cuda::Cuda, io::safetensors::SafetensorsReader},
    models::{
        loader::{LoadConfig, WeightLoader},
        qwen3,
    },
};

pub(super) fn load_execution(
    spec: &SpeculativeConfig,
    target_path: &str,
    target: &qwen3::Qwen3Model<bf16, Cuda>,
    load_cfg: &LoadConfig,
    max_seq_len: usize,
    max_step_tokens: usize,
    cuda: &Cuda,
) -> Result<ConditionedServing<Eagle3DraftHead<bf16, Cuda>>, String> {
    let SpeculativeConfig::Eagle3 {
        draft_model,
        num_draft_tokens,
        allow_target_mismatch,
    } = spec
    else {
        return Err("expected EAGLE3 launch config".into());
    };
    if load_cfg.mlp_quant.is_some() || load_cfg.fp8_block.is_some() {
        return Err("EAGLE3 currently requires unquantized BF16 target weights".into());
    }
    let draft_reader = SafetensorsReader::open(draft_model)?;
    let draft_loader = WeightLoader::new(&draft_reader);
    let raw = std::fs::read(std::path::Path::new(draft_model).join("config.json"))
        .map_err(|e| e.to_string())?;
    let checkpoint =
        qwen3::eagle3::Eagle3Checkpoint::parse(&raw, &draft_loader).map_err(|e| e.to_string())?;
    let target_dims = target.dims();
    checkpoint
        .validate(target_dims, max_seq_len, *num_draft_tokens)
        .map_err(|e| e.to_string())?;
    tracing::info!(format=?checkpoint.format(), draft=%draft_model, "EAGLE3 checkpoint format detected");
    let actual = std::path::Path::new(&target_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    // A draft has its own trained RoPE; equality to target RoPE is not a pairing rule.
    if let Some(trained_target) = checkpoint.trained_target()
        && !actual.eq_ignore_ascii_case(trained_target.rsplit('/').next().unwrap_or(""))
    {
        if !allow_target_mismatch {
            return Err(format!(
                "EAGLE3 was trained for {trained_target}; target is {target_path}. Set speculative.allow_target_mismatch=true for an experimental pairing"
            ));
        }
        tracing::warn!(target=%target_path,trained_target,"experimental EAGLE3 target pairing; acceptance is unverified");
    }
    if checkpoint.trained_target().is_none() {
        tracing::info!(target=%target_path, "draft config omits target identity; checkpoint geometry validated");
    }
    let head = checkpoint
        .build::<bf16, Cuda>(
            &draft_loader,
            target_dims,
            &target.embed.table,
            max_seq_len,
            cuda,
        )
        .map_err(|e| format!("EAGLE3 draft load: {e}"))?;
    let execution = ConditionedServing::with_features(
        head,
        checkpoint
            .feature_spec(target_dims)
            .map_err(|e| e.to_string())?,
        max_seq_len,
        max_step_tokens,
        *num_draft_tokens,
        cuda,
    )
    .map_err(|e| format!("EAGLE3 serving: {e}"))?;
    Ok(execution)
}
