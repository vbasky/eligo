//! Candidates and the result of selection.

use crate::Image;

/// One generated image together with the seed that produced it and its reward.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The generated image.
    pub image: Image,
    /// Seed used to generate it (for reproducing the exact result).
    pub seed: u64,
    /// Reward from the [`crate::Scorer`]; higher is better.
    pub score: f32,
}

/// The outcome of [`crate::best_of_n`]: every candidate considered, plus the
/// index of the chosen one.
#[derive(Debug, Clone)]
pub struct Selection {
    /// All candidates that were generated and scored, in generation order.
    pub all: Vec<Candidate>,
    /// Index into [`Selection::all`] of the highest-scoring candidate.
    pub best_index: usize,
}

impl Selection {
    /// The chosen candidate — the one with the highest reward.
    pub fn best(&self) -> &Candidate {
        &self.all[self.best_index]
    }
}
