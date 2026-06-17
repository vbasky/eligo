# eligo

*eligo* — Latin for **"I choose."** Best-of-N image generation that **selects**
the best candidate by a measurable reward, instead of returning a single
one-shot image.

Given a prompt, eligo generates `n` candidates through a pluggable `Backend`,
scores each against the prompt with a `Scorer` (the reward), and returns the
highest-scoring one. An optional bounded re-roll regenerates the single worst
candidate once. That generate → score → select loop is the smallest honest
agentic pattern: a numeric reward drives a decision.

**Scope is deliberately bounded.** eligo owns the loop and the contracts
(`Backend`, `Scorer`) — not a model zoo, not an editor, no unbounded refinement.
The default build ships a deterministic mock backend/scorer so the loop runs
end-to-end with no model weights. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for
the milestones (CLIP scorer, Stable Diffusion backend).

[![CI](https://github.com/vbasky/eligo/actions/workflows/ci.yml/badge.svg)](https://github.com/vbasky/eligo/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Layout

```bash
crates/
  eligo/       # library crate
  eligo-cli/   # binary crate (clap), installs as `eligo`
```

## Quick start

```bash
just setup        # one-time: enable the auto-format pre-commit hook
just build        # build the workspace
just test         # run all tests

# Generate 5 candidates with the mock backend, pick the best, write the winner:
just run -- generate "a lighthouse at dusk" -n 5 --reroll-worst --out winner.ppm
```

The CLI has two subcommands: `generate` (best-of-N) and `similar`
(find look-alike images — see below).

If you don't have [`just`](https://github.com/casey/just):
`cargo install just`.

## The full thing: real images, real selection

With both features on, eligo generates actual Stable Diffusion images and
keeps the one CLIP judges best:

```bash
cargo run -p eligo-cli --features "sd clip" -- generate \
  "a photograph of a red apple on a wooden table" -n 4 --steps 20 \
  --sd-model-dir <sd-onnx-dir> --sd-tokenizer <tokenizer.json> \
  --clip-model <clip.onnx> --clip-tokenizer <tokenizer.json> \
  --out winner.png
```

- **`--features sd`** — the *artist* (`SdBackend`): turns the prompt into images.
- **`--features clip`** — the *judge* (`ClipScorer`): scores each against the prompt.
- Together: best-of-N over real images. Either can stay on the mock for a fast,
  weight-free loop.

Models are standard ONNX exports (a diffusers `text_encoder`/`unet`/`vae_decoder`
directory for SD; a CLIP `model.onnx`). See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## The CLIP reward (optional `clip` feature)

The default build uses the deterministic mock scorer. The real reward —
`ClipScorer` — scores a candidate by the cosine similarity between the CLIP
image and text embeddings, run through ONNX Runtime (`ort`). It lives behind a
cargo feature so the base build stays weight-free:

```bash
cargo test -p eligo --features clip   # builds ort + tokenizers; runs the
                                          # preprocessing/math tests (no weights)
```

```rust
use eligo::{best_of_n, GenerateConfig, ClipScorer};

let scorer = ClipScorer::from_files("model.onnx", "tokenizer.json")?;
let selection = best_of_n(&backend, &scorer, &GenerateConfig::new("a red bicycle"))?;
```

Or from the CLI:

```bash
cargo run -p eligo-cli --features clip -- generate "a red bicycle" -n 4 \
  --clip-model model.onnx --clip-tokenizer tokenizer.json
```

It expects a standard Hugging Face CLIP ONNX export (e.g. `clip-vit-base-patch32`)
whose graph takes `pixel_values`, `input_ids`, `attention_mask` and outputs
`image_embeds` and `text_embeds`. The reward is validated end-to-end against real
weights in `crates/eligo/tests/clip_real.rs` (ignored by default; point the
`ELIGO_CLIP_MODEL` / `ELIGO_CLIP_TOKENIZER` env vars at the files and run
with `--ignored`).

## Find similar images (`similar`, needs `clip`)

The same CLIP embeddings that judge prompt-match also power *image↔image*
similarity — the basis for "more like this" and content-based recommendations.
`ClipEmbedder::embed_image` turns an image into a vector; nearby vectors are
look-alikes. The `similar` subcommand ranks a folder against a query image:

```bash
cargo run -p eligo-cli --features clip -- similar \
  query.png ./photos -k 5 \
  --clip-model model.onnx --clip-tokenizer tokenizer.json
```

```text
most similar to query.png:
  1.0000  ./photos/query.png      # itself
  0.9238  ./photos/other_a.png
  0.9101  ./photos/other_b.png
```

Library-side, the building block is reusable:

```rust
use eligo::ClipEmbedder;

let embedder = ClipEmbedder::from_files("model.onnx", "tokenizer.json")?;
let vec = embedder.embed_image(&image)?;          // image → embedding
let sim = embedder.image_similarity(&a, &b)?;     // cosine, in [-1, 1]
```

An external catalogue can embed each asset once, store the vectors, and serve
nearest-neighbour lookups — eligo provides the embedding + similarity; storage,
indexing, and per-user recommendations belong to the catalogue layer.

## Development

`just check-all` runs the exact gate CI enforces — formatting, clippy
(`-D warnings`), tests, and docs — before you push.

| Task | Command |
| --- | --- |
| Format | `just fmt` |
| Lint | `just lint` |
| Test | `just test` |
| Docs | `just docs` |
| Dependency audit | `just deny` (needs `cargo install cargo-deny`) |

## Releasing

1. Update `CHANGELOG.md` under a new `## [x.y.z]` heading and commit.
2. `just release x.y.z` — bumps versions, tags, and pushes.
3. CI (`.github/workflows/release.yml`) builds binaries for macOS (arm64 +
   x86_64), Linux, and Windows, and publishes a GitHub Release with checksums
   and the changelog notes.
4. To also publish to crates.io: `PUBLISH=1 just release x.y.z` (needs
   `cargo login`).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
