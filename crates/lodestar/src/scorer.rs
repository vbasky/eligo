//! Reward contract: scoring a candidate against its prompt.
//!
//! The [`Scorer`] is the heart of lodestar selection — it turns "which image is
//! best?" into a number. The first real implementation is CLIP
//! prompt-alignment (cosine similarity between the image embedding and the text
//! embedding); a no-reference quality term (BRISQUE/NIQE) can be blended in
//! later. Both live behind this trait so the loop never hard-codes a metric.

use crate::{Image, Result};

/// Scores how well a generated image satisfies the prompt.
///
/// Higher is better. Scores need not be bounded to any range — the loop only
/// compares them — but implementations should be *consistent* across calls so
/// candidates are ranked fairly.
pub trait Scorer {
    /// Return a reward for `image` given the `prompt` it was generated from.
    fn score(&self, prompt: &str, image: &Image) -> Result<f32>;
}
