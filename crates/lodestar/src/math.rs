//! Small vector math shared by scorers — pure Rust, no model required.
//!
//! CLIP-style rewards reduce to "how aligned are these two embeddings?", which
//! is a cosine similarity. Keeping it here (and tested without any model) means
//! the reward arithmetic is verified independently of the heavy inference path.

/// L2-normalize a vector in place. A zero vector is left unchanged (its norm is
/// zero, so there is nothing to scale).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length vectors, in `[-1, 1]`.
///
/// Returns `0.0` if the lengths differ or either vector has zero magnitude —
/// callers treat that as "no signal" rather than an error, since it only
/// happens for degenerate embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_are_maximally_similar() {
        let a = [1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_score_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_score_minus_one() {
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn length_mismatch_and_zero_vector_are_zero() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn normalize_gives_unit_length() {
        let mut v = [3.0, 4.0];
        l2_normalize(&mut v);
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_leaves_zero_vector_untouched() {
        let mut v = [0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, [0.0, 0.0]);
    }
}
