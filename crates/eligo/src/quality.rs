//! No-reference image quality (M3) — a prompt-independent "is this clean and
//! sharp?" signal, and a blend with any other [`Scorer`].
//!
//! The CLIP reward ([`crate::ClipScorer`]) answers "does this match the words?"
//! but a blurry, washed-out picture can still match the words. This adds the
//! orthogonal question — "does it look good?" — so best-of-N can prefer a
//! candidate that is *both* on-prompt and crisp.
//!
//! This is a lightweight, **parameter-free** metric in the no-reference family
//! (it uses Laplacian sharpness + local contrast — the building blocks of
//! BRISQUE/NIQE). It needs no model and no weights, so it lives in the core,
//! always available. A fully reference-calibrated BRISQUE/NIQE (with a trained
//! model) is tracked separately as its own bounded project; eligo only needs
//! a sound relative ordering among same-size candidates, which this provides.

use crate::{Image, Result, Scorer};

/// Soft-saturation constant for sharpness (Laplacian variance on luma in
/// `[0,1]`). Tuned so typical images land mid-range rather than at 0 or 1.
const SHARPNESS_K: f32 = 0.01;
/// Reference RMS contrast at which the contrast term saturates to 1.
const CONTRAST_REF: f32 = 0.25;
/// Weight of sharpness vs. contrast in the final score.
const SHARPNESS_WEIGHT: f32 = 0.7;

/// No-reference quality of an image, in `[0, 1]` (higher = sharper / cleaner).
///
/// A flat or heavily blurred image scores near 0; a crisp, well-contrasted one
/// scores high.
pub fn quality_score(image: &Image) -> f32 {
    let (w, h, luma) = to_luma(image);
    let sharpness = laplacian_variance(&luma, w, h);
    let contrast = std_dev(&luma);

    let sharpness_term = 1.0 - (-sharpness / SHARPNESS_K).exp();
    let contrast_term = (contrast / CONTRAST_REF).min(1.0);
    (SHARPNESS_WEIGHT * sharpness_term + (1.0 - SHARPNESS_WEIGHT) * contrast_term).clamp(0.0, 1.0)
}

/// Convert to luma in `[0, 1]`, returning `(width, height, pixels)`.
fn to_luma(image: &Image) -> (usize, usize, Vec<f32>) {
    let n = (image.width * image.height) as usize;
    let mut luma = Vec::with_capacity(n);
    for px in image.rgb.chunks_exact(3) {
        let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
        luma.push((0.299 * r + 0.587 * g + 0.114 * b) / 255.0);
    }
    (image.width as usize, image.height as usize, luma)
}

/// Variance of the 3×3 Laplacian response over the image interior — the classic
/// no-reference focus/blur measure (high = sharp, low = blurry).
fn laplacian_variance(luma: &[f32], w: usize, h: usize) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let at = |x: usize, y: usize| luma[y * w + x];
    let mut responses = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            responses.push(lap);
        }
    }
    variance(&responses)
}

/// Standard deviation of a slice (RMS contrast when applied to luma).
fn std_dev(v: &[f32]) -> f32 {
    variance(v).sqrt()
}

fn variance(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32
}

/// A [`Scorer`] that rates image quality alone, ignoring the prompt.
#[derive(Debug, Clone, Default)]
pub struct QualityScorer;

impl Scorer for QualityScorer {
    fn score(&self, _prompt: &str, image: &Image) -> Result<f32> {
        Ok(quality_score(image))
    }
}

/// Wraps a base [`Scorer`] (e.g. CLIP alignment) and blends in the no-reference
/// quality score: `score = (1 - weight) * base + weight * quality`.
///
/// `weight` is clamped to `[0, 1]`; 0 ignores quality, 1 ignores the base.
pub struct QualityWeighted {
    base: Box<dyn Scorer>,
    weight: f32,
}

impl QualityWeighted {
    /// Blend `base` with the quality score using `weight` (clamped to `[0,1]`).
    pub fn new(base: Box<dyn Scorer>, weight: f32) -> Self {
        Self { base, weight: weight.clamp(0.0, 1.0) }
    }
}

impl Scorer for QualityWeighted {
    fn score(&self, prompt: &str, image: &Image) -> Result<f32> {
        let base = self.base.score(prompt, image)?;
        let quality = quality_score(image);
        Ok((1.0 - self.weight) * base + self.weight * quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sharp, high-contrast checkerboard.
    fn checkerboard(size: u32, cell: u32) -> Image {
        let mut rgb = Vec::with_capacity((size * size * 3) as usize);
        for y in 0..size {
            for x in 0..size {
                let v = if ((x / cell) + (y / cell)) % 2 == 0 { 255 } else { 0 };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        Image::new(size, size, rgb).unwrap()
    }

    /// 3×3 box blur — softens edges, lowering sharpness.
    fn box_blur(image: &Image) -> Image {
        let (w, h) = (image.width as usize, image.height as usize);
        let (_, _, luma) = to_luma(image);
        let mut out = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut count = 0.0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            sum += luma[ny as usize * w + nx as usize];
                            count += 1.0;
                        }
                    }
                }
                let v = ((sum / count) * 255.0).round() as u8;
                let idx = (y * w + x) * 3;
                out[idx] = v;
                out[idx + 1] = v;
                out[idx + 2] = v;
            }
        }
        Image::new(image.width, image.height, out).unwrap()
    }

    fn solid(size: u32, v: u8) -> Image {
        Image::new(size, size, vec![v; (size * size * 3) as usize]).unwrap()
    }

    #[test]
    fn scores_are_in_range() {
        for img in [checkerboard(32, 4), box_blur(&checkerboard(32, 4)), solid(32, 128)] {
            let q = quality_score(&img);
            assert!((0.0..=1.0).contains(&q), "out of range: {q}");
        }
    }

    #[test]
    fn flat_image_scores_near_zero() {
        assert!(quality_score(&solid(32, 128)) < 0.05);
    }

    #[test]
    fn sharp_beats_blurred() {
        let sharp = checkerboard(48, 4);
        let blurred = box_blur(&sharp);
        let (qs, qb) = (quality_score(&sharp), quality_score(&blurred));
        assert!(qs > qb, "sharp {qs:.3} should beat blurred {qb:.3}");
    }

    #[test]
    fn weighting_blends_base_and_quality() {
        // Base scorer returns a constant 1.0; quality of a flat image ~0.
        struct One;
        impl Scorer for One {
            fn score(&self, _p: &str, _i: &Image) -> Result<f32> {
                Ok(1.0)
            }
        }
        let img = solid(32, 128);
        let blended = QualityWeighted::new(Box::new(One), 0.5);
        // 0.5*1.0 + 0.5*(~0) ≈ 0.5, strictly below the unweighted base of 1.0.
        let s = blended.score("x", &img).unwrap();
        assert!(s < 1.0 && s > 0.4 && s < 0.6, "blend was {s}");
    }
}
