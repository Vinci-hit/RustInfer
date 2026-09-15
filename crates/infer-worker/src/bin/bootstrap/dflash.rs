//! Launch-time loading of a DFlash block proposer for a dense Qwen3 target.
use half::bf16;
use infer_protocol::config::SpeculativeConfig;
use infer_worker::{
    application::speculative::{BlockProposer, serving::SpeculativeServing},
    components::dflash::DFlashDraftHead,
    domain::model::DecoderModel,
    infrastructure::{cuda::Cuda, io::safetensors::SafetensorsReader},
    models::{
        loader::{LoadConfig, WeightLoader},
        qwen3,
    },
};

pub(super) fn load_execution(
    spec: &SpeculativeConfig,
    target: &qwen3::Qwen3Model<bf16, Cuda>,
    load_cfg: &LoadConfig,
    max_seq_len: usize,
    max_step_tokens: usize,
    cuda: &Cuda,
) -> Result<SpeculativeServing<BlockProposer<bf16, Cuda, DFlashDraftHead<bf16, Cuda>>>, String> {
    let SpeculativeConfig::Dflash {
        draft_model,
        num_draft_tokens,
    } = spec
    else {
        return Err("expected DFlash launch config".into());
    };
    if load_cfg.mlp_quant.is_some() || load_cfg.fp8_block.is_some() {
        return Err("DFlash currently requires unquantized BF16 target weights".into());
    }
    let reader = SafetensorsReader::open(draft_model)?;
    let loader = WeightLoader::new(&reader);
    let raw = std::fs::read(std::path::Path::new(draft_model).join("config.json"))
        .map_err(|e| e.to_string())?;
    let cfg = qwen3::dflash::DFlashConfig::parse(&raw).map_err(|e| e.to_string())?;
    cfg.validate(target.dims(), max_seq_len, *num_draft_tokens)
        .map_err(|e| e.to_string())?;
    let readout = target
        .lm_head
        .proj
        .weight
        .as_dense()
        .ok_or("DFlash requires a dense target readout")?;
    let head = qwen3::dflash::build(
        &loader,
        &cfg,
        target.dims(),
        &target.embed.table,
        readout,
        max_seq_len,
        cuda,
    )
    .map_err(|e| format!("DFlash load: {e}"))?;
    let proposer = BlockProposer::new(head, max_seq_len, max_step_tokens.min(max_seq_len), cuda)
        .map_err(|e| e.to_string())?;
    tracing::info!(draft=%draft_model, block_size=cfg.block_size, draft_tokens=num_draft_tokens,
        feature_layers=?cfg.dflash_config.target_layer_ids,
        "DFlashDraftModel loaded; shared target embedding/readout, independent draft RoPE");
    SpeculativeServing::from_proposer(
        proposer,
        cfg.feature_spec(target.dims()).map_err(|e| e.to_string())?,
        max_seq_len,
        max_step_tokens,
        *num_draft_tokens,
        cuda,
    )
    .map_err(|e| e.to_string())
}
