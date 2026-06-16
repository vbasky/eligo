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
  fixed-length tokenization. End-to-end scoring against real weights pending.

## [0.1.0] - 2026-06-17

### Added

- Initial scaffold: best-of-N image-generation **selection** library.
- `Backend` and `Scorer` traits — the pluggable generation + reward contracts.
- `best_of_n` selection loop with an optional bounded "re-roll the worst once".
- Deterministic `mock` backend/scorer so the loop runs end-to-end without model
  weights.
- `lodestar` CLI: generate N candidates, print per-candidate scores, write the
  winner as PPM.
- Docs: `ROADMAP.md` (bounded milestones — CLIP scorer, candle SD backend).
