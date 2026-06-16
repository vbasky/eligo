//! Stable Diffusion text-to-image [`Backend`] (M2), on ONNX Runtime.
//!
//! Implements the standard latent-diffusion txt2img loop against a Hugging Face
//! diffusers ONNX export (text encoder + UNet + VAE decoder), with a hand-rolled
//! DDIM scheduler and classifier-free guidance. Same runtime (`ort`) and CLIP
//! tokenizer as the [`crate::ClipScorer`], so the `sd` feature adds no new
//! dependencies over `clip`.
//!
//! Gated behind the `sd` cargo feature; the default build needs no weights.
//!
//! ## Expected models
//!
//! A diffusers ONNX export laid out as `text_encoder/model.onnx`,
//! `unet/model.onnx`, `vae_decoder/model.onnx`, plus the CLIP `tokenizer.json`.
//! I/O follows the optimum/diffusers convention (`input_ids` → `last_hidden_state`;
//! `sample` + `timestep` + `encoder_hidden_states` → `out_sample`;
//! `latent_sample` → `sample`).

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;

use crate::{Backend, Error, Image, Result};

const SAMPLE_SIZE: usize = 512;
const LATENT_SIZE: usize = SAMPLE_SIZE / 8;
const LATENT_CHANNELS: usize = 4;
const CTX_LEN: usize = 77;
const BOS_TOKEN: i32 = 49406;
const EOT_TOKEN: i32 = 49407;
/// VAE latent scaling factor used by SD v1.x.
const VAE_SCALE: f32 = 0.181_5;

/// A Stable Diffusion txt2img backend.
pub struct SdBackend {
    text_encoder: Mutex<Session>,
    unet: Mutex<Session>,
    vae_decoder: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    scheduler: Ddim,
    guidance_scale: f32,
}

impl SdBackend {
    /// Load a backend from a diffusers ONNX export directory and CLIP tokenizer.
    ///
    /// `steps` is the number of denoising steps (20–30 is typical); fewer is
    /// faster but lower quality. `guidance_scale` ~7.5 is the SD default.
    pub fn from_dir(
        model_dir: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
        steps: usize,
        guidance_scale: f32,
    ) -> Result<Self> {
        let dir = model_dir.as_ref();
        let debug = std::env::var_os("ELIGO_SD_DEBUG").is_some();
        let load = |rel: &str| -> Result<Session> {
            let path = dir.join(rel);
            let session = Session::builder()
                .and_then(|mut b| b.commit_from_file(&path))
                .map_err(|e| Error::Backend(format!("loading {}: {e}", path.display())))?;
            if debug {
                eprintln!("[sd] {rel}");
                for i in session.inputs() {
                    eprintln!("    in  {i:?}");
                }
                for o in session.outputs() {
                    eprintln!("    out {o:?}");
                }
            }
            Ok(session)
        };
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer.as_ref())
            .map_err(|e| Error::Backend(format!("loading tokenizer: {e}")))?;
        Ok(Self {
            text_encoder: Mutex::new(load("text_encoder/model.onnx")?),
            unet: Mutex::new(load("unet/model.onnx")?),
            vae_decoder: Mutex::new(load("vae_decoder/model.onnx")?),
            tokenizer,
            scheduler: Ddim::new(steps),
            guidance_scale,
        })
    }

    /// Tokenize `prompt` to a fixed `CTX_LEN` row of CLIP token ids.
    fn tokenize(&self, prompt: &str) -> Result<Array2<i32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Backend(format!("tokenizing: {e}")))?;
        let ids = encoding.get_ids();
        let mut row = Array2::<i32>::from_elem((1, CTX_LEN), EOT_TOKEN);
        if ids.is_empty() {
            row[[0, 0]] = BOS_TOKEN;
        }
        for (i, &id) in ids.iter().take(CTX_LEN).enumerate() {
            row[[0, i]] = id as i32;
        }
        Ok(row)
    }

    /// Run the CLIP text encoder, returning `[1, CTX_LEN, hidden]` embeddings.
    fn encode_text(&self, prompt: &str) -> Result<Array3<f32>> {
        let ids = self.tokenize(prompt)?;
        let ids = Tensor::from_array(ids).map_err(|e| Error::Backend(e.to_string()))?;
        let mut session = lock(&self.text_encoder)?;
        let outputs = session
            .run(ort::inputs!["input_ids" => ids])
            .map_err(|e| Error::Backend(format!("text encoder: {e}")))?;
        let (shape, data) = first_tensor(&outputs)?;
        let hidden = *shape.last().ok_or_else(|| Error::Backend("empty text shape".into()))?;
        Array3::from_shape_vec((1, CTX_LEN, hidden), data)
            .map_err(|e| Error::Backend(format!("reshape text embeddings: {e}")))
    }

    /// Predict noise from the UNet for one (latent, timestep, conditioning).
    fn unet_eps(&self, latents: &[f32], timestep: i64, cond: &Array3<f32>) -> Result<Vec<f32>> {
        let sample = Array4::from_shape_vec(
            (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
            latents.to_vec(),
        )
        .map_err(|e| Error::Backend(format!("latent reshape: {e}")))?;
        let sample = Tensor::from_array(sample).map_err(|e| Error::Backend(e.to_string()))?;
        let ts = Tensor::from_array(Array1::<i64>::from_elem(1, timestep))
            .map_err(|e| Error::Backend(e.to_string()))?;
        let hidden = Tensor::from_array(cond.clone()).map_err(|e| Error::Backend(e.to_string()))?;

        let mut session = lock(&self.unet)?;
        let outputs = session
            .run(ort::inputs![
                "sample" => sample,
                "timestep" => ts,
                "encoder_hidden_states" => hidden,
            ])
            .map_err(|e| Error::Backend(format!("unet: {e}")))?;
        Ok(first_tensor(&outputs)?.1)
    }

    /// Decode latents to an RGB image via the VAE decoder.
    fn decode(&self, latents: &[f32]) -> Result<Image> {
        let scaled: Vec<f32> = latents.iter().map(|x| x / VAE_SCALE).collect();
        let sample = Array4::from_shape_vec((1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), scaled)
            .map_err(|e| Error::Backend(format!("latent reshape: {e}")))?;
        let sample = Tensor::from_array(sample).map_err(|e| Error::Backend(e.to_string()))?;

        let mut session = lock(&self.vae_decoder)?;
        let outputs = session
            .run(ort::inputs!["latent_sample" => sample])
            .map_err(|e| Error::Backend(format!("vae decoder: {e}")))?;
        let (shape, data) = first_tensor(&outputs)?;
        let (h, w) = (shape[2], shape[3]);

        // Output is [1,3,H,W] in roughly [-1,1]; map to RGB8.
        let mut rgb = vec![0u8; h * w * 3];
        let plane = h * w;
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = data[c * plane + y * w + x];
                    let u = ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                    rgb[(y * w + x) * 3 + c] = u;
                }
            }
        }
        Image::new(w as u32, h as u32, rgb)
    }
}

impl Backend for SdBackend {
    fn generate(&self, prompt: &str, seed: u64) -> Result<Image> {
        let cond = self.encode_text(prompt)?;
        let uncond = self.encode_text("")?;

        let n = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE;
        let mut latents = standard_normal(seed, n);

        for &timestep in &self.scheduler.timesteps {
            let eps_uncond = self.unet_eps(&latents, timestep, &uncond)?;
            let eps_cond = self.unet_eps(&latents, timestep, &cond)?;
            let eps: Vec<f32> = eps_uncond
                .iter()
                .zip(&eps_cond)
                .map(|(u, c)| u + self.guidance_scale * (c - u))
                .collect();
            latents = self.scheduler.step(&eps, timestep, &latents);
        }

        self.decode(&latents)
    }
}

/// Deterministic DDIM scheduler (eta = 0) for SD v1.x's scaled-linear betas.
struct Ddim {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<i64>,
    step_size: i64,
}

impl Ddim {
    fn new(steps: usize) -> Self {
        const TRAIN_STEPS: usize = 1000;
        const BETA_START: f32 = 0.000_85;
        const BETA_END: f32 = 0.012;

        // "scaled_linear": betas evenly spaced in sqrt-space, then squared.
        let mut alphas_cumprod = Vec::with_capacity(TRAIN_STEPS);
        let mut cumprod = 1.0f32;
        let (s0, s1) = (BETA_START.sqrt(), BETA_END.sqrt());
        for i in 0..TRAIN_STEPS {
            let frac = i as f32 / (TRAIN_STEPS - 1) as f32;
            let beta = (s0 + frac * (s1 - s0)).powi(2);
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        let step_size = (TRAIN_STEPS / steps) as i64;
        let timesteps: Vec<i64> = (0..steps).map(|i| (i as i64) * step_size).rev().collect();

        Self { alphas_cumprod, timesteps, step_size }
    }

    /// One DDIM update: predict x0 from the noise estimate, then step to the
    /// previous timestep.
    fn step(&self, eps: &[f32], timestep: i64, sample: &[f32]) -> Vec<f32> {
        let alpha_t = self.alphas_cumprod[timestep as usize];
        let prev = timestep - self.step_size;
        let alpha_prev = if prev >= 0 { self.alphas_cumprod[prev as usize] } else { 1.0 };
        let (sqrt_at, sqrt_bt) = (alpha_t.sqrt(), (1.0 - alpha_t).sqrt());
        let (sqrt_ap, sqrt_bp) = (alpha_prev.sqrt(), (1.0 - alpha_prev).sqrt());

        sample
            .iter()
            .zip(eps)
            .map(|(&x, &e)| {
                let pred_x0 = (x - sqrt_bt * e) / sqrt_at;
                sqrt_ap * pred_x0 + sqrt_bp * e
            })
            .collect()
    }
}

/// `n` standard-normal samples from a seed (xorshift + Box–Muller) — keeps
/// generation reproducible without pulling in an RNG crate.
fn standard_normal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next_u64 = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let unit = |x: u64| (x >> 11) as f32 / (1u64 << 53) as f32;

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = unit(next_u64()).max(f32::MIN_POSITIVE);
        let u2 = unit(next_u64());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

fn lock(session: &Mutex<Session>) -> Result<std::sync::MutexGuard<'_, Session>> {
    session.lock().map_err(|_| Error::Backend("SD session lock poisoned".into()))
}

/// Extract the first output of a run as `(shape, data)`.
fn first_tensor(outputs: &ort::session::SessionOutputs) -> Result<(Vec<usize>, Vec<f32>)> {
    let value =
        outputs.iter().next().ok_or_else(|| Error::Backend("model produced no outputs".into()))?.1;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Backend(format!("extract output: {e}")))?;
    Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_samples_are_reproducible_and_sane() {
        let a = standard_normal(7, 4096);
        let b = standard_normal(7, 4096);
        assert_eq!(a, b);
        assert_ne!(standard_normal(8, 16), a[..16]);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean} not near 0");
    }

    #[test]
    fn ddim_timesteps_descend_within_range() {
        let s = Ddim::new(20);
        assert_eq!(s.timesteps.len(), 20);
        assert!(s.timesteps.windows(2).all(|w| w[0] > w[1]));
        assert!(*s.timesteps.first().unwrap() < 1000);
        assert_eq!(*s.timesteps.last().unwrap(), 0);
        assert_eq!(s.alphas_cumprod.len(), 1000);
    }

    #[test]
    fn ddim_step_preserves_dimensions() {
        let s = Ddim::new(10);
        let sample = vec![0.5f32; 8];
        let eps = vec![0.1f32; 8];
        let out = s.step(&eps, s.timesteps[0], &sample);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
