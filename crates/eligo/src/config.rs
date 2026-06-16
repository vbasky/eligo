//! Generation/selection configuration — the loop's knobs, and its boundary.
//!
//! The two fields that keep this project *bounded* are [`GenerateConfig::candidates`]
//! (how many to generate) and [`RerollPolicy`] (whether to retry the worst one,
//! once). There is deliberately no unbounded refinement loop.

use crate::{Error, Result};

/// How to spend a second round of generation, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerollPolicy {
    /// Generate `candidates` images, pick the best, stop. No second round.
    #[default]
    None,
    /// After the first round, regenerate the single lowest-scoring candidate
    /// with a fresh seed and keep whichever of the two scores higher. Bounded
    /// to exactly one extra generation — never a loop.
    RerollWorstOnce,
}

/// Inputs to [`crate::best_of_n`].
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Text prompt to generate from.
    pub prompt: String,
    /// Number of candidates to generate in the first round (must be >= 1).
    pub candidates: u32,
    /// Base RNG seed; candidate `i` uses `seed + i` so runs are reproducible.
    pub seed: u64,
    /// Whether to spend one extra generation on the worst candidate.
    pub reroll: RerollPolicy,
}

impl GenerateConfig {
    /// Start a config for `prompt` with sensible defaults (4 candidates,
    /// seed 0, no re-roll).
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { prompt: prompt.into(), candidates: 4, seed: 0, reroll: RerollPolicy::None }
    }

    /// Set the number of candidates to generate.
    pub fn with_candidates(mut self, n: u32) -> Self {
        self.candidates = n;
        self
    }

    /// Set the base seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the re-roll policy.
    pub fn with_reroll(mut self, reroll: RerollPolicy) -> Self {
        self.reroll = reroll;
        self
    }

    /// Validate the config before any generation happens.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() {
            return Err(Error::Config("prompt must not be empty".into()));
        }
        if self.candidates == 0 {
            return Err(Error::Config("candidates must be >= 1".into()));
        }
        Ok(())
    }
}
