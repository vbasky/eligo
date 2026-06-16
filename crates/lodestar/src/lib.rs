//! # lodestar — *the star you steer by*
//!
//! Best-of-N image generation that **selects** the best candidate by a
//! measurable reward, rather than returning a single one-shot result.
//!
//! A lodestar is the fixed star navigators steer toward. Here the lodestar is
//! the [`Scorer`] — the reward signal the loop steers candidates toward. Where
//! `revelo` *reveals* and `viser` *sees*, lodestar *chooses a heading*: given a
//! prompt, it generates `n` candidate images through a [`Backend`], scores each
//! against the prompt with the [`Scorer`], and returns the highest-scoring one.
//! An optional bounded re-roll replaces the single worst candidate once. That
//! generate → score → select loop is the smallest honest agentic pattern: a
//! numeric reward drives a decision.
//!
//! ## Scope (deliberately bounded)
//!
//! lodestar is a *selection* library, not a model zoo or an editor. It owns the
//! loop and the contracts ([`Backend`], [`Scorer`]); concrete model inference
//! (a Stable Diffusion backend, and CLIP scoring via ONNX Runtime) are pluggable
//! implementations behind those traits. The default build ships a deterministic
//! mock backend/scorer so the loop is testable without model weights.
//!
//! ```
//! use lodestar::{best_of_n, GenerateConfig};
//! use lodestar::mock::{MockBackend, MockScorer};
//!
//! let backend = MockBackend::default();
//! let scorer = MockScorer;
//! let cfg = GenerateConfig::new("a red bicycle").with_candidates(4);
//! let selection = best_of_n(&backend, &scorer, &cfg).unwrap();
//! assert_eq!(selection.all.len(), 4);
//! // The chosen candidate is the highest-scoring one.
//! let top = selection.all.iter().map(|c| c.score).fold(f32::MIN, f32::max);
//! assert!((selection.best().score - top).abs() < f32::EPSILON);
//! ```

mod backend;
mod candidate;
mod config;
mod error;
mod math;
mod pipeline;
mod quality;
mod scorer;

pub mod mock;

#[cfg(feature = "clip")]
mod clip;

#[cfg(feature = "sd")]
mod sd;

pub use backend::{Backend, Image};
pub use candidate::{Candidate, Selection};
pub use config::{GenerateConfig, RerollPolicy};
pub use error::{Error, Result};
pub use math::{cosine_similarity, l2_normalize};
pub use pipeline::best_of_n;
pub use quality::{QualityScorer, QualityWeighted, quality_score};
pub use scorer::Scorer;

#[cfg(feature = "clip")]
pub use clip::ClipScorer;

#[cfg(feature = "sd")]
pub use sd::SdBackend;
