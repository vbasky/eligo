//! Deterministic mock [`Backend`] and [`Scorer`] for tests and demos.
//!
//! These let the selection loop run end-to-end with no model weights: the
//! backend paints a tiny image seeded from `(prompt, seed)`, and the scorer
//! derives a stable reward from the image bytes. Same inputs → same outputs, so
//! runs are reproducible. They are *not* meant to produce meaningful imagery —
//! the real `candle` backend and CLIP scorer replace them.

use crate::{Backend, Image, Result, Scorer};

/// A 64-bit FNV-1a hash — small, dependency-free, good enough for deterministic
/// pseudo-random bytes in the mocks.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A backend that paints a deterministic `width`x`height` image from the prompt
/// and seed. Defaults to 8x8.
#[derive(Debug, Clone)]
pub struct MockBackend {
    /// Width of generated images.
    pub width: u32,
    /// Height of generated images.
    pub height: u32,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self { width: 8, height: 8 }
    }
}

impl Backend for MockBackend {
    fn generate(&self, prompt: &str, seed: u64) -> Result<Image> {
        let n = self.width as usize * self.height as usize * 3;
        let mut rgb = Vec::with_capacity(n);
        for i in 0..n {
            let mut key = Vec::with_capacity(prompt.len() + 16);
            key.extend_from_slice(prompt.as_bytes());
            key.extend_from_slice(&seed.to_le_bytes());
            key.extend_from_slice(&(i as u64).to_le_bytes());
            rgb.push((fnv1a(&key) & 0xff) as u8);
        }
        Image::new(self.width, self.height, rgb)
    }
}

/// A scorer that maps `(prompt, image)` to a stable reward in `[0, 1)`.
#[derive(Debug, Clone, Default)]
pub struct MockScorer;

impl Scorer for MockScorer {
    fn score(&self, prompt: &str, image: &Image) -> Result<f32> {
        let mut key = Vec::with_capacity(prompt.len() + image.rgb.len());
        key.extend_from_slice(prompt.as_bytes());
        key.extend_from_slice(&image.rgb);
        let h = fnv1a(&key);
        // Map the high bits into [0, 1).
        Ok((h >> 40) as f32 / (1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_deterministic_and_well_formed() {
        let b = MockBackend::default();
        let a = b.generate("a lighthouse", 1).unwrap();
        let c = b.generate("a lighthouse", 1).unwrap();
        assert_eq!(a, c);
        assert_eq!(a.rgb.len(), (a.width * a.height * 3) as usize);
    }

    #[test]
    fn different_seeds_diverge() {
        let b = MockBackend::default();
        assert_ne!(b.generate("x", 1).unwrap(), b.generate("x", 2).unwrap());
    }

    #[test]
    fn scorer_is_stable_and_in_range() {
        let img = MockBackend::default().generate("y", 3).unwrap();
        let s1 = MockScorer.score("y", &img).unwrap();
        let s2 = MockScorer.score("y", &img).unwrap();
        assert_eq!(s1, s2);
        assert!((0.0..1.0).contains(&s1));
    }
}
