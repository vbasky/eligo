//! End-to-end validation of the CLIP reward against real weights.
//!
//! Ignored by default (it needs a downloaded model + native ONNX Runtime). Run
//! it explicitly once weights are in place:
//!
//! ```bash
//! LODESTAR_CLIP_MODEL=.models/clip/model.onnx \
//! LODESTAR_CLIP_TOKENIZER=.models/clip/tokenizer.json \
//!   cargo test -p lodestar --features clip --test clip_real -- --ignored --nocapture
//! ```
//!
//! The check is the one that matters for best-of-N: given two candidate images,
//! the [`ClipScorer`] must reward the one that actually matches the prompt more
//! highly. If that ordering holds on real weights, the reward signal is sound.
#![cfg(feature = "clip")]

use lodestar::{ClipScorer, Image, Scorer};

/// A solid `224x224` image of one RGB colour.
fn solid(rgb: [u8; 3]) -> Image {
    let mut buf = Vec::with_capacity(224 * 224 * 3);
    for _ in 0..224 * 224 {
        buf.extend_from_slice(&rgb);
    }
    Image::new(224, 224, buf).unwrap()
}

fn scorer_from_env() -> Option<ClipScorer> {
    let model = std::env::var("LODESTAR_CLIP_MODEL").ok()?;
    let tokenizer = std::env::var("LODESTAR_CLIP_TOKENIZER").ok()?;
    Some(ClipScorer::from_files(model, tokenizer).expect("load CLIP scorer"))
}

#[test]
#[ignore = "needs real CLIP weights via LODESTAR_CLIP_MODEL / LODESTAR_CLIP_TOKENIZER"]
fn reward_prefers_the_matching_image() {
    let Some(scorer) = scorer_from_env() else {
        eprintln!("skipping: LODESTAR_CLIP_MODEL / LODESTAR_CLIP_TOKENIZER not set");
        return;
    };

    let red = solid([220, 30, 30]);
    let blue = solid([30, 30, 220]);

    // Best-of-N framing: one prompt, two candidate images — the matching image
    // must win. Checked in both directions so a correct ordering can't be chance.
    let score = |prompt: &str, img: &Image| scorer.score(prompt, img).unwrap();

    let (rr, rb) = (score("a solid red image", &red), score("a solid red image", &blue));
    let (bb, br) = (score("a solid blue image", &blue), score("a solid blue image", &red));
    eprintln!("red prompt : red={rr:.4} blue={rb:.4}");
    eprintln!("blue prompt: blue={bb:.4} red={br:.4}");

    for s in [rr, rb, bb, br] {
        assert!((-1.0..=1.0).contains(&s), "cosine out of range: {s}");
    }
    assert!(rr > rb, "red image should win for a red prompt ({rr:.4} vs {rb:.4})");
    assert!(bb > br, "blue image should win for a blue prompt ({bb:.4} vs {br:.4})");
}
