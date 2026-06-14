//! Run Z-Image-Turbo end-to-end and save a 1024×1024 PNG.
//!
//! Usage: `cargo run --release --features cuda --example zimage_demo`

use std::path::Path;

use infer_worker::domain::tensor::Tensor;
use infer_worker::domain::types::Dtype;
use infer_worker::infrastructure::cuda::Cuda;
use infer_worker::models::diffusion::pipeline::{GenerateParams, ZImagePipeline};
const MODEL_DIR: &str = "/apdcephfs_qy2/share_303432435/vinciiliu/models/z-image-turbo";
const PROMPT: &str = "A majestic snow leopard standing on a cliff edge at sunset, with golden \
     light illuminating its fur, dramatic mountain landscape in the background, \
     photorealistic, 8k detail";
const OUT_PATH: &str = "/tmp/zimage_rust.png";

fn main() -> Result<(), String> {
    if !Path::new(MODEL_DIR).is_dir() {
        return Err(format!("model dir not found: {}", MODEL_DIR));
    }

    eprintln!("[demo] CUDA init...");
    let cuda = Cuda::new(0).map_err(|e| format!("Cuda::new: {:?}", e))?;
    eprintln!("[demo] loading pipeline as F32 (this can take ~30s)...");
    let t0 = std::time::Instant::now();
    let mut pipeline: ZImagePipeline<half::bf16> =
        ZImagePipeline::from_pretrained(MODEL_DIR, &cuda)
            .map_err(|e| format!("from_pretrained: {:?}", e))?;
    eprintln!("[demo] loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let params = GenerateParams {
        height: 1024,
        width: 1024,
        num_inference_steps: 9,
        guidance_scale: 0.0,
        seed: Some(42),
        sigmas: None,
    };

    eprintln!(
        "[demo] generating ({}x{}, {} steps, seed={:?})...",
        params.height, params.width, params.num_inference_steps, params.seed
    );
    let t0 = std::time::Instant::now();
    let img = pipeline
        .generate(PROMPT, &params, &cuda)
        .map_err(|e| format!("generate: {:?}", e))?;
    eprintln!("[demo] generated in {:.1}s", t0.elapsed().as_secs_f32());

    let s = img.shape().as_slice().to_vec();
    if s.len() != 4 || s[0] != 1 || s[1] != 3 {
        return Err(format!("unexpected image shape: {:?}", s));
    }
    let h = s[2];
    let w = s[3];
    let host: Vec<f32> = {
        let bf16_host: Vec<half::bf16> = img.to_host_vec().map_err(|e| format!("D2H: {:?}", e))?;
        bf16_host.iter().map(|v| v.to_f32()).collect()
    };

    let mut nfinite = 0usize;
    let mut nnan = 0usize;
    let mut sum = 0.0_f64;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in &host {
        if x.is_finite() {
            nfinite += 1;
            sum += x as f64;
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
        } else {
            nnan += 1;
        }
    }
    eprintln!(
        "[demo] image stats: finite={} nan={} mean={:.4} min={:.4} max={:.4}",
        nfinite,
        nnan,
        sum / nfinite.max(1) as f64,
        mn,
        mx
    );

    let plane = h * w;
    let mut bytes = Vec::with_capacity(h * w * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = host[c * plane + y * w + x];
                let u = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
                bytes.push(u);
            }
        }
    }

    let file = std::fs::File::create(OUT_PATH).map_err(|e| format!("create png: {}", e))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {}", e))?;
    writer
        .write_image_data(&bytes)
        .map_err(|e| format!("png data: {}", e))?;
    eprintln!("[demo] saved {} ({}x{})", OUT_PATH, w, h);
    Ok(())
}
