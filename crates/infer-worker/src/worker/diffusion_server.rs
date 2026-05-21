//! Diffusion Worker data plane.
//!
//! This server handles `WorkerCommand::DiffusionBatch` and returns
//! `DiffusionBatchOutput`. It intentionally does not use LLM `ModelRunner`, KV
//! cache, or decode state.

use infer_protocol::scheduler_to_worker::{DiffusionBatchCmd, WorkerCommand};
use infer_protocol::worker_to_scheduler::{
    DiffusionBatchOutput, DiffusionImage, DiffusionOutputItem, DiffusionOutputMetrics,
    DiffusionOutputStatus,
};

use crate::base::error::{Error, Result};
use crate::model::diffusion::pipeline::{DiffusionPipeline, DiffusionRequest};
use crate::tensor::Tensor;

pub struct DiffusionWorkerServer<P: DiffusionPipeline> {
    pipeline: P,
    zmq_in: zmq::Socket,
    zmq_out: zmq::Socket,
    max_batch_size: usize,
    draining: bool,
}

impl<P: DiffusionPipeline> DiffusionWorkerServer<P> {
    pub fn new(
        pipeline: P,
        zmq_in: zmq::Socket,
        zmq_out: zmq::Socket,
        max_batch_size: usize,
    ) -> Self {
        Self {
            pipeline,
            zmq_in,
            zmq_out,
            max_batch_size,
            draining: false,
        }
    }

    pub fn run(mut self) {
        tracing::info!("DiffusionWorkerServer running: pipeline={}", self.pipeline.name());
        while !self.draining {
            let data = match self.zmq_in.recv_bytes(0) {
                Ok(data) => data,
                Err(e) => {
                    tracing::error!("Diffusion worker ZMQ recv error: {}", e);
                    continue;
                }
            };

            let Some(output) = self.handle_data_plane_message(&data) else {
                continue;
            };
            match rmp_serde::to_vec(&output) {
                Ok(bytes) => {
                    if let Err(e) = self.zmq_out.send(&bytes, 0) {
                        tracing::error!("Diffusion worker ZMQ send error: {}", e);
                    }
                }
                Err(e) => tracing::error!("Failed to serialize DiffusionBatchOutput: {}", e),
            }
        }
    }

    fn handle_data_plane_message(&mut self, data: &[u8]) -> Option<DiffusionBatchOutput> {
        let cmd: WorkerCommand = match rmp_serde::from_slice(data) {
            Ok(cmd) => cmd,
            Err(e) => {
                tracing::error!("Failed to decode WorkerCommand for diffusion worker: {}", e);
                return None;
            }
        };

        match cmd {
            WorkerCommand::DiffusionBatch(batch) => Some(self.handle_diffusion_batch(batch)),
            WorkerCommand::Drain(drain) => {
                tracing::info!("Diffusion DrainWorker mode={:?}", drain.mode);
                self.draining = true;
                None
            }
            WorkerCommand::UnloadModel(unload) => {
                tracing::info!("Diffusion UnloadModel model_instance_id={}", unload.model_instance_id);
                self.draining = true;
                None
            }
            WorkerCommand::Cancel(cancel) => {
                tracing::info!("Diffusion CancelRequest ignored sequence_id={}", cancel.sequence_id);
                None
            }
            WorkerCommand::GrantBlocks(grant) => {
                tracing::debug!("Diffusion GrantBlocks ignored sequence_id={}", grant.sequence_id);
                None
            }
            WorkerCommand::Prefill(_) => {
                tracing::error!("Diffusion worker received LLM Prefill command");
                None
            }
        }
    }

    fn handle_diffusion_batch(&mut self, batch: DiffusionBatchCmd) -> DiffusionBatchOutput {
        if let Err(e) = batch.validate(self.max_batch_size) {
            let message = e.to_string();
            return DiffusionBatchOutput {
                results: batch.requests.into_iter().map(|req| DiffusionOutputItem {
                    request_id: req.request_id,
                    status: DiffusionOutputStatus::Error,
                    image: None,
                    error: Some(message.clone()),
                    metrics: DiffusionOutputMetrics::default(),
                }).collect(),
            };
        }

        let mut results = Vec::with_capacity(batch.requests.len());
        for req in batch.requests {
            let request_id = req.request_id.clone();
            let pipeline_req = DiffusionRequest {
                prompt: req.prompt,
                prompt_input_ids: req.prompt_input_ids,
                negative_prompt: req.negative_prompt,
                negative_prompt_input_ids: req.negative_prompt_input_ids,
                height: req.height as usize,
                width: req.width as usize,
                num_inference_steps: req.num_inference_steps,
                sigmas: req.sigmas,
                guidance_scale: req.guidance_scale,
                seed: req.seed,
            };

            match self.pipeline.generate(&pipeline_req) {
                Ok(output) => {
                    let metrics = DiffusionOutputMetrics {
                        encode_prompt_ms: output.metrics.encode_prompt_ms,
                        denoise_ms: output.metrics.denoise_ms,
                        decode_ms: output.metrics.decode_ms,
                        total_ms: output.metrics.total_ms,
                    };
                    match tensor_to_rgb8_image(&output.output) {
                        Ok(image) => results.push(DiffusionOutputItem {
                            request_id,
                            status: DiffusionOutputStatus::Success,
                            image: Some(image),
                            error: None,
                            metrics,
                        }),
                        Err(e) => results.push(DiffusionOutputItem {
                            request_id,
                            status: DiffusionOutputStatus::Error,
                            image: None,
                            error: Some(e.to_string()),
                            metrics,
                        }),
                    }
                }
                Err(e) => results.push(DiffusionOutputItem {
                    request_id,
                    status: DiffusionOutputStatus::Error,
                    image: None,
                    error: Some(e.to_string()),
                    metrics: DiffusionOutputMetrics::default(),
                }),
            }
        }

        DiffusionBatchOutput { results }
    }
}

fn tensor_to_rgb8_image(tensor: &Tensor) -> Result<DiffusionImage> {
    let cpu = tensor.to_cpu()?;
    let shape = cpu.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument(format!(
            "diffusion image tensor must be [B,C,H,W], got {:?}",
            shape
        )).into());
    }
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if b == 0 || c != 3 {
        return Err(Error::InvalidArgument(format!(
            "diffusion image tensor expected B>=1,C=3,H,W, got {:?}",
            shape
        )).into());
    }

    let src = cpu.as_f32()?.as_slice()?;
    let mut data = Vec::with_capacity(h * w * 3);
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let r = src[idx];
            let g = src[plane + idx];
            let b = src[2 * plane + idx];
            data.push(float_to_u8(r));
            data.push(float_to_u8(g));
            data.push(float_to_u8(b));
        }
    }

    Ok(DiffusionImage {
        width: w as u32,
        height: h as u32,
        channels: 3,
        format: "rgb8".to_string(),
        data,
    })
}

fn float_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
