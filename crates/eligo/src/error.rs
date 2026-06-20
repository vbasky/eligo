//! Error type for eligo.

/// Convenience alias for results in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can arise while generating or selecting candidates.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The configuration was rejected before generation began
    /// (e.g. zero candidates requested, empty prompt).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A [`crate::Backend`] failed to produce an image.
    #[error("backend failed to generate image: {0}")]
    Backend(String),

    /// A [`crate::Scorer`] failed to score a candidate.
    #[error("scorer failed: {0}")]
    Scorer(String),

    /// Generation completed but produced no candidates to choose from.
    /// Currently unreachable in normal flow (config validation rejects zero
    /// candidates); reserved for future re-roll logic that may empty the list.
    #[error("no candidates were produced")]
    #[doc(hidden)]
    Empty,
}
