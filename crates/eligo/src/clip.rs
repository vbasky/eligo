//! CLIP embeddings on ONNX Runtime — the reusable foundation for both the
//! prompt-alignment reward (M1) and image↔image similarity (M4).
//!
//! [`ClipEmbedder`] turns an image or a text prompt into an L2-normalized
//! embedding vector. Two consumers build on it:
//!
//! - [`ClipScorer`] (a [`Scorer`]): cosine(image, text) — "does this match the
//!   words?" — used by best-of-N selection.
//! - similarity / recommendations: cosine(image, image) via
//!   [`ClipEmbedder::embed_image`] + [`crate::cosine_similarity`], or the
//!   convenience [`ClipEmbedder::image_similarity`].
//!
//! Gated behind the `clip` cargo feature so the default build needs no model
//! weights and no native runtime.
//!
//! ## Expected model
//!
//! A standard Hugging Face CLIP ONNX export (e.g. `clip-vit-base-patch32`) with
//! a single graph taking `pixel_values` (`1×3×224×224` f32), `input_ids` and
//! `attention_mask` (`1×ctx` i64), and producing `image_embeds` and
//! `text_embeds` outputs. The `image_embeds` output depends only on the pixels
//! and `text_embeds` only on the tokens, so [`ClipEmbedder::embed_image`] feeds
//! a dummy prompt and [`ClipEmbedder::embed_text`] feeds a blank image.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::math::{cosine_similarity, l2_normalize};
use crate::{Error, Image, Result, Scorer};

/// CLIP image preprocessing constants (ViT-B family).
const IMAGE_SIZE: u32 = 224;
const CLIP_MEAN: [f32; 3] = [0.481_454_7, 0.457_827_5, 0.408_210_7];
const CLIP_STD: [f32; 3] = [0.268_629_5, 0.261_302_6, 0.275_777_1];
/// CLIP text context length.
const CTX_LEN: usize = 77;
/// CLIP end-of-text / padding token id.
const EOT_TOKEN: i64 = 49407;

/// Turns images and prompts into L2-normalized CLIP embeddings.
///
/// The ONNX session is wrapped in a `Mutex`, so the `Debug` representation is
/// intentionally opaque (the session type itself is not `Debug`).
pub struct ClipEmbedder {
    // `Session::run` needs unique access; a Mutex keeps `&self` methods usable
    // while leaving the embedder `Send + Sync` for use across workers.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl std::fmt::Debug for ClipEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipEmbedder").finish_non_exhaustive()
    }
}

impl ClipEmbedder {
    /// Load an embedder from an ONNX `model.onnx` and its `tokenizer.json`.
    pub fn from_files(model: impl AsRef<Path>, tokenizer: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(model.as_ref()))
            .map_err(|e| Error::Scorer(format!("loading CLIP model: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer.as_ref())
            .map_err(|e| Error::Scorer(format!("loading CLIP tokenizer: {e}")))?;
        Ok(Self { session: Mutex::new(session), tokenizer })
    }

    /// Embed an image into an L2-normalized vector (the `image_embeds` output).
    #[must_use = "the embedding is the entire result of this function"]
    pub fn embed_image(&self, image: &Image) -> Result<Vec<f32>> {
        let (ids, mask) = self.tokenize("")?;
        let mut outs = self.run(preprocess(image)?, ids, mask, &["image_embeds"])?;
        let mut v = outs.remove(0);
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Embed a text prompt into an L2-normalized vector (the `text_embeds`
    /// output).
    #[must_use = "the embedding is the entire result of this function"]
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let (ids, mask) = self.tokenize(text)?;
        let blank = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
        let mut outs = self.run(blank, ids, mask, &["text_embeds"])?;
        let mut v = outs.remove(0);
        l2_normalize(&mut v);
        Ok(v)
    }

    /// Embed both an image and a prompt in a single model run, returning
    /// `(image_embed, text_embed)`, each L2-normalized. More efficient than
    /// calling [`Self::embed_image`] and [`Self::embed_text`] separately.
    #[must_use = "the embeddings are the entire result of this function"]
    pub fn embed_both(&self, text: &str, image: &Image) -> Result<(Vec<f32>, Vec<f32>)> {
        let (ids, mask) = self.tokenize(text)?;
        let mut outs = self.run(preprocess(image)?, ids, mask, &["image_embeds", "text_embeds"])?;
        let mut txt = outs.pop().expect("text_embeds not in model outputs");
        let mut img = outs.pop().expect("image_embeds not in model outputs");
        l2_normalize(&mut img);
        l2_normalize(&mut txt);
        Ok((img, txt))
    }

    /// Cosine similarity between two images, in `[-1, 1]` (1 = identical).
    #[must_use = "the similarity value is the entire result of this function"]
    pub fn image_similarity(&self, a: &Image, b: &Image) -> Result<f32> {
        Ok(cosine_similarity(&self.embed_image(a)?, &self.embed_image(b)?))
    }

    /// Tokenize `prompt` into fixed-length `(input_ids, attention_mask)` rows,
    /// padded/truncated to [`CTX_LEN`].
    fn tokenize(&self, prompt: &str) -> Result<(Array2<i64>, Array2<i64>)> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::Scorer(format!("tokenizing prompt: {e}")))?;
        let ids = encoding.get_ids();

        let mut input_ids = Array2::<i64>::from_elem((1, CTX_LEN), EOT_TOKEN);
        let mut attention = Array2::<i64>::zeros((1, CTX_LEN));
        for (i, &id) in ids.iter().take(CTX_LEN).enumerate() {
            input_ids[[0, i]] = id as i64;
            attention[[0, i]] = 1;
        }
        Ok((input_ids, attention))
    }

    /// Run the model once and extract the requested named outputs as owned
    /// vectors (extraction happens while the session lock is held).
    fn run(
        &self,
        pixel_values: Array4<f32>,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
        wants: &[&str],
    ) -> Result<Vec<Vec<f32>>> {
        let pv = Tensor::from_array(pixel_values)
            .map_err(|e| Error::Scorer(format!("pixel tensor: {e}")))?;
        let ids = Tensor::from_array(input_ids)
            .map_err(|e| Error::Scorer(format!("input_ids tensor: {e}")))?;
        let mask = Tensor::from_array(attention_mask)
            .map_err(|e| Error::Scorer(format!("attention tensor: {e}")))?;

        let mut session =
            self.session.lock().map_err(|_| Error::Scorer("CLIP session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![
                "pixel_values" => pv,
                "input_ids" => ids,
                "attention_mask" => mask,
            ])
            .map_err(|e| Error::Scorer(format!("CLIP inference: {e}")))?;
        wants.iter().map(|name| extract_vec(&outputs, name)).collect()
    }
}

/// A [`Scorer`] backed by a [`ClipEmbedder`]: reward = cosine(image, text).
///
/// The inner [`ClipEmbedder`] is opaque (see its `Debug` documentation).
pub struct ClipScorer {
    embedder: ClipEmbedder,
}

impl std::fmt::Debug for ClipScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipScorer").finish_non_exhaustive()
    }
}

impl ClipScorer {
    /// Load a CLIP scorer from an ONNX `model.onnx` and its `tokenizer.json`.
    pub fn from_files(model: impl AsRef<Path>, tokenizer: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { embedder: ClipEmbedder::from_files(model, tokenizer)? })
    }

    /// Borrow the underlying embedder (e.g. for similarity queries).
    pub fn embedder(&self) -> &ClipEmbedder {
        &self.embedder
    }
}

impl Scorer for ClipScorer {
    fn score(&self, prompt: &str, image: &Image) -> Result<f32> {
        let (image_embed, text_embed) = self.embedder.embed_both(prompt, image)?;
        Ok(cosine_similarity(&image_embed, &text_embed))
    }
}

/// Pull a named float output out of the session result as an owned vector.
fn extract_vec(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Vec<f32>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| Error::Scorer(format!("CLIP model has no `{name}` output")))?;
    let (_shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Scorer(format!("extracting `{name}`: {e}")))?;
    Ok(data.to_vec())
}

/// Resize (shortest side to 224, center-crop) and CLIP-normalize an [`Image`]
/// into a `1×3×224×224` NCHW tensor.
fn preprocess(image: &Image) -> Result<Array4<f32>> {
    use image::{DynamicImage, RgbImage, imageops::FilterType};

    let rgb = RgbImage::from_raw(image.width, image.height, image.rgb.clone())
        .ok_or_else(|| Error::Scorer("image buffer does not match its dimensions".into()))?;

    let short = image.width.min(image.height).max(1) as f32;
    let scale = IMAGE_SIZE as f32 / short;
    let nw = ((image.width as f32 * scale).round() as u32).max(IMAGE_SIZE);
    let nh = ((image.height as f32 * scale).round() as u32).max(IMAGE_SIZE);
    let resized =
        DynamicImage::ImageRgb8(rgb).resize_exact(nw, nh, FilterType::CatmullRom).to_rgb8();

    let left = (nw - IMAGE_SIZE) / 2;
    let top = (nh - IMAGE_SIZE) / 2;
    let mut arr = Array4::<f32>::zeros((1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize));
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let p = resized.get_pixel(left + x, top + y);
            for c in 0..3 {
                arr[[0, c, y as usize, x as usize]] =
                    (p[c] as f32 / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, value: u8) -> Image {
        Image::new(width, height, vec![value; (width * height * 3) as usize]).unwrap()
    }

    #[test]
    fn preprocess_yields_normalized_nchw_tensor() {
        // A non-square image exercises the resize + center-crop path.
        let tensor = preprocess(&solid(64, 40, 128)).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
        // 128/255 normalized by CLIP mean/std lands well inside [-2, 2].
        for &v in tensor.iter() {
            assert!(v.is_finite() && v.abs() < 5.0);
        }
    }

    #[test]
    fn preprocess_normalization_matches_formula() {
        let tensor = preprocess(&solid(224, 224, 255)).unwrap();
        let expected = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((tensor[[0, 0, 0, 0]] - expected).abs() < 1e-3);
    }
}
