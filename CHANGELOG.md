# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

The release workflow extracts the notes for a version from the matching
`## [x.y.z]` section below, so keep these headings intact.

## [Unreleased]

### Added

- `math` module: `cosine_similarity` + `l2_normalize` (the reward arithmetic),
  unit-tested without any model.
- M1: `ClipScorer` — a CLIP prompt-alignment `Scorer` on ONNX Runtime (`ort`),
  behind an optional `clip` cargo feature so the default build stays weight-free.
  Includes CLIP image preprocessing (resize + center-crop + normalize) and
  fixed-length tokenization.
- CLI `--clip-model` / `--clip-tokenizer` flags (CLI `clip` feature) to select
  candidates with the real CLIP reward instead of the mock.
- `tests/clip_real.rs`: end-to-end validation against real `clip-vit-base-patch32`
  ONNX weights — the matching image out-scores the other in both directions.
  Ignored by default; driven by `ELIGO_CLIP_MODEL` / `ELIGO_CLIP_TOKENIZER`.
- M2: `SdBackend` — Stable Diffusion txt2img on ONNX Runtime (text encoder +
  UNet + VAE decoder, hand-rolled DDIM scheduler, classifier-free guidance,
  seeded reproducible RNG), behind an optional `sd` cargo feature (no new deps
  over `clip`).
- CLI `--sd-model-dir` / `--sd-tokenizer` / `--steps` / `--guidance` (CLI `sd`
  feature) generate real images; `Image::save_png` + `--out foo.png` saves them.
  `ELIGO_SD_DEBUG=1` dumps model I/O.
- `tests/sd_real.rs`: end-to-end validation against a vanilla fp32 SD-1.5 ONNX
  export. Ignored by default; driven by `ELIGO_SD_DIR` / `ELIGO_SD_TOKENIZER`.
- M3: no-reference quality — `quality_score` + `QualityScorer` (Laplacian
  sharpness + RMS contrast, parameter-free, in the core), and `QualityWeighted`
  to blend it with any scorer. CLI `--quality-weight` and `--save-all`.

## [0.1.0] - 2026-06-17

### Added (0.1.0)

- Initial scaffold: best-of-N image-generation **selection** library.
- `Backend` and `Scorer` traits — the pluggable generation + reward contracts.
- `best_of_n` selection loop with an optional bounded "re-roll the worst once".
- Deterministic `mock` backend/scorer so the loop runs end-to-end without model
  weights.
- `eligo` CLI: generate N candidates, print per-candidate scores, write the
  winner as PPM.
- Docs: `ROADMAP.md` (bounded milestones — CLIP scorer, candle SD backend).
