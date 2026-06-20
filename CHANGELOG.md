# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

The release workflow extracts the notes for a version from the matching
`## [x.y.z]` section below, so keep these headings intact.

## [Unreleased]

## [0.1.4] - 2026-06-20

### Changed

- `SdBackend::unet_eps`: replaced `cond.clone()` + `Tensor::from_array` with
  `TensorRef::from_array_view` — the text conditioning tensor is now borrowed
  instead of cloned on every denoising step. The surrounding `latents.to_vec()`
  and `Array1::from_elem` still allocate once per step, but the former ~236 KB
  clone per call is eliminated.
- `Image::save_png`: replaced `RgbImage::from_raw(self.rgb.clone())` with
  `image::save_buffer(…, &self.rgb, …)` — a borrow-based API that encodes directly
  from the stored pixel buffer with no intermediate copy.
- `Error::Empty`: marked `#[doc(hidden)]` and documented as reserved for future
  internal use; currently unreachable due to upstream config validation.

## [0.1.3] - 2026-06-20

### Fixed

- `ClipEmbedder::embed_both`: replaced fragile ordered-index extraction with
  `.pop()` so the method is robust to ONNX graph output reordering.
- `SdBackend::generate`: pre-flatten text conditioning tensors before the
  denoising loop — the data is still cloned once per step (inherent to the ONNX
  Runtime input API) but the `Array3` shape metadata is computed only once.

### Added

- `Debug` implementations for `ClipEmbedder`, `ClipScorer`, `SdBackend`, and
  `QualityWeighted`. ONNX sessions are opaque; `SdBackend` reports steps and
  guidance scale; `QualityWeighted` reports its blend weight.
- `#[inline]` hints on hot-path functions: `l2_normalize`, `cosine_similarity`,
  `Image::new`, `generate_one`, `argmax_by_score`, `argmin_by_score`, `std_dev`,
  `variance`.

### Changed

- `is_image_file` is now always available (was gated behind `#[cfg(feature = "clip")]`).
- `release.sh`: the pre-flight branch check now resolves the remote default
  branch instead of hardcoding `main`.

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
