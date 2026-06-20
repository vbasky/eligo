# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

The release workflow extracts the notes for a version from the matching
`## [x.y.z]` section below, so keep these headings intact.

## [Unreleased]

## [0.1.2] - 2026-06-20

### Fixed

- `Cargo.toml`: added `readme` field to the library crate so the README is visible
  on crates.io.
- `release.sh`: replaced `{{project-name}}` template placeholders with `eligo`.
- `SdBackend::from_dir`: reject `steps == 0` with an error instead of panicking
  on division-by-zero in the DDIM scheduler.
- Removed unused workspace dependencies (`serde`, `serde_json`, `tokio`).
- `quality.rs`: avoid potential `u32` overflow in `width * height` cast.

### Added

- `#[must_use]` attributes on `best_of_n`, `cosine_similarity`, `quality_score`,
  `Backend::generate`, `ClipEmbedder::embed_image`, `ClipEmbedder::embed_text`,
  `ClipEmbedder::embed_both`, and `ClipEmbedder::image_similarity`.

## [0.1.1] - 2026-06-20

### Changed

- Docs: replaced the ASCII selection-loop diagram with an SVG (PNG fallback for
  renderers that don't display SVG); added the lens banner and rewrote the
  README.

### Fixed

- Corrected the declared MSRV to `1.85.1` (the real minimum); the CI MSRV check
  now derives its toolchain from `Cargo.toml` instead of a hardcoded version.

## [0.1.0] - 2026-06-17

### Added

- Initial scaffold: best-of-N image-generation **selection** library.
- `Backend` and `Scorer` traits — the pluggable generation + reward contracts.
- `best_of_n` selection loop with an optional bounded "re-roll the worst once".
- Deterministic `mock` backend/scorer so the loop runs end-to-end without model
  weights.
- `math` module: `cosine_similarity` + `l2_normalize` (the reward arithmetic),
  unit-tested without any model.
- `ClipScorer` — a CLIP prompt-alignment `Scorer` on ONNX Runtime (`ort`), behind
  an optional `clip` cargo feature so the default build stays weight-free.
  Includes CLIP image preprocessing (resize + center-crop + normalize) and
  fixed-length tokenization.
- `SdBackend` — Stable Diffusion txt2img on ONNX Runtime (text encoder + UNet +
  VAE decoder, hand-rolled DDIM scheduler, classifier-free guidance, seeded
  reproducible RNG), behind an optional `sd` cargo feature (no new deps over
  `clip`).
- No-reference quality — `quality_score` + `QualityScorer` (Laplacian sharpness +
  RMS contrast, parameter-free, in the core) and `QualityWeighted` to blend it
  with any scorer.
- `ClipEmbedder` (image/text → L2-normalized embedding, `embed_both`,
  `image_similarity`) factored out of `ClipScorer`; `Image::open`/`Image::save_png`.
- `eligo` CLI with subcommands: `generate <prompt> …` (best-of-N: per-candidate
  scores, winner saved) and `similar <query> <dir> …` (rank a folder by CLIP
  image↔image similarity). Flags: `--clip-model`/`--clip-tokenizer` (`clip`),
  `--sd-model-dir`/`--sd-tokenizer`/`--steps`/`--guidance` (`sd`),
  `--quality-weight`, `--save-all`, `--out`.
- Real-weight end-to-end tests (`tests/clip_real.rs`, `tests/sd_real.rs`),
  ignored by default and driven by `ELIGO_*` env vars.
- Docs: `ROADMAP.md` (bounded milestones).
