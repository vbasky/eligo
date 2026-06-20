//! The selection loop: generate → score → choose (→ optionally re-roll once).

use crate::{Backend, Candidate, Error, GenerateConfig, RerollPolicy, Result, Scorer};

/// Generate `cfg.candidates` images, score each against the prompt, and return
/// the highest-scoring one along with the full set considered.
///
/// Candidate `i` is generated with seed `cfg.seed + i`, so the whole run is
/// reproducible. If [`RerollPolicy::RerollWorstOnce`] is set, the lowest-scoring
/// candidate is regenerated once with a fresh seed and the better of the two is
/// kept in its place — exactly one extra generation, never a loop.
#[must_use = "the selection result should be consumed"]
pub fn best_of_n(
    backend: &dyn Backend,
    scorer: &dyn Scorer,
    cfg: &GenerateConfig,
) -> Result<crate::Selection> {
    cfg.validate()?;

    let mut candidates = Vec::with_capacity(cfg.candidates as usize);
    for i in 0..cfg.candidates as u64 {
        let seed = cfg.seed.wrapping_add(i);
        candidates.push(generate_one(backend, scorer, &cfg.prompt, seed)?);
    }

    if matches!(cfg.reroll, RerollPolicy::RerollWorstOnce) {
        reroll_worst_once(backend, scorer, cfg, &mut candidates)?;
    }

    let best_index = argmax_by_score(&candidates).ok_or(Error::Empty)?;
    Ok(crate::Selection { all: candidates, best_index })
}

/// Generate and score a single candidate.
#[inline]
fn generate_one(
    backend: &dyn Backend,
    scorer: &dyn Scorer,
    prompt: &str,
    seed: u64,
) -> Result<Candidate> {
    let image = backend.generate(prompt, seed)?;
    let score = scorer.score(prompt, &image)?;
    Ok(Candidate { image, seed, score })
}

/// Replace the worst candidate with a fresh draw if the new one scores higher.
/// The replacement seed is offset past every first-round seed to avoid repeats.
fn reroll_worst_once(
    backend: &dyn Backend,
    scorer: &dyn Scorer,
    cfg: &GenerateConfig,
    candidates: &mut [Candidate],
) -> Result<()> {
    let Some(worst) = argmin_by_score(candidates) else {
        return Ok(());
    };
    let fresh_seed = cfg.seed.wrapping_add(cfg.candidates as u64);
    let replacement = generate_one(backend, scorer, &cfg.prompt, fresh_seed)?;
    if replacement.score > candidates[worst].score {
        candidates[worst] = replacement;
    }
    Ok(())
}

/// Index of the highest-scoring candidate (NaN scores sort last).
#[inline]
fn argmax_by_score(candidates: &[Candidate]) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        .map(|(i, _)| i)
}

/// Index of the lowest-scoring candidate (NaN scores sort first).
#[inline]
fn argmin_by_score(candidates: &[Candidate]) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockBackend, MockScorer};

    #[test]
    fn generates_requested_count_and_picks_highest() {
        let cfg = GenerateConfig::new("a calm harbour at dawn").with_candidates(5);
        let sel = best_of_n(&MockBackend::default(), &MockScorer, &cfg).unwrap();
        assert_eq!(sel.all.len(), 5);
        let top = sel.all.iter().map(|c| c.score).fold(f32::MIN, f32::max);
        assert!((sel.best().score - top).abs() < f32::EPSILON);
    }

    #[test]
    fn is_reproducible_for_same_seed() {
        let cfg = GenerateConfig::new("a calm harbour").with_candidates(3).with_seed(42);
        let a = best_of_n(&MockBackend::default(), &MockScorer, &cfg).unwrap();
        let b = best_of_n(&MockBackend::default(), &MockScorer, &cfg).unwrap();
        assert_eq!(a.best().seed, b.best().seed);
        assert_eq!(a.best().image, b.best().image);
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let cfg = GenerateConfig::new("   ");
        let err = best_of_n(&MockBackend::default(), &MockScorer, &cfg).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn zero_candidates_is_rejected() {
        let cfg = GenerateConfig::new("anything").with_candidates(0);
        let err = best_of_n(&MockBackend::default(), &MockScorer, &cfg).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn reroll_never_lowers_the_winner() {
        let base = GenerateConfig::new("a winding mountain road").with_candidates(4).with_seed(7);
        let no_reroll = best_of_n(&MockBackend::default(), &MockScorer, &base).unwrap();
        let with_reroll = best_of_n(
            &MockBackend::default(),
            &MockScorer,
            &base.clone().with_reroll(RerollPolicy::RerollWorstOnce),
        )
        .unwrap();
        // Re-rolling the worst can only keep or raise the best score.
        assert!(with_reroll.best().score >= no_reroll.best().score);
    }
}
