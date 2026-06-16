//! End-to-end validation of the Stable Diffusion backend against real weights.
//!
//! Ignored by default (needs ~2GB of ONNX weights + the native runtime). Run it
//! with a diffusers ONNX export directory and the CLIP tokenizer:
//!
//! ```bash
//! LODESTAR_SD_DIR=.models/sd \
//! LODESTAR_SD_TOKENIZER=.models/clip/tokenizer.json \
//!   cargo test -p lodestar --features sd --test sd_real -- --ignored --nocapture
//! ```
//!
//! A few denoising steps is enough to prove the pipeline produces a real,
//! seed-dependent image (not noise, not a constant) — quality scales with steps.
#![cfg(feature = "sd")]

use lodestar::{Backend, SdBackend};

fn backend_from_env(steps: usize) -> Option<SdBackend> {
    let dir = std::env::var("LODESTAR_SD_DIR").ok()?;
    let tokenizer = std::env::var("LODESTAR_SD_TOKENIZER").ok()?;
    Some(SdBackend::from_dir(dir, tokenizer, steps, 7.5).expect("load SD backend"))
}

/// Fraction of distinct byte values — a constant or near-constant image scores
/// near 0; a real generation has broad tonal variation.
fn distinct_fraction(rgb: &[u8]) -> f32 {
    let mut seen = [false; 256];
    for &b in rgb {
        seen[b as usize] = true;
    }
    seen.iter().filter(|&&s| s).count() as f32 / 256.0
}

#[test]
#[ignore = "needs real SD ONNX weights via LODESTAR_SD_DIR / LODESTAR_SD_TOKENIZER"]
fn generates_a_real_seed_dependent_image() {
    let Some(backend) = backend_from_env(6) else {
        eprintln!("skipping: LODESTAR_SD_DIR / LODESTAR_SD_TOKENIZER not set");
        return;
    };

    let prompt = "a photograph of a red apple on a wooden table";
    let a = backend.generate(prompt, 1).unwrap();
    let again = backend.generate(prompt, 1).unwrap();
    let b = backend.generate(prompt, 2).unwrap();

    // Right shape, real RGB buffer.
    assert_eq!((a.width, a.height), (512, 512));
    assert_eq!(a.rgb.len(), 512 * 512 * 3);

    // Same seed reproduces exactly; different seed diverges.
    assert_eq!(a.rgb, again.rgb, "same seed should be deterministic");
    assert_ne!(a.rgb, b.rgb, "different seeds should differ");

    // Not a constant / degenerate image.
    let frac = distinct_fraction(&a.rgb);
    eprintln!("distinct byte fraction: {frac:.3}");
    assert!(frac > 0.2, "image looks degenerate (distinct fraction {frac:.3})");
}
