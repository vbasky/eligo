# lodestar

*The star you steer by.* Best-of-N image generation that **selects** the best
candidate by a measurable reward, instead of returning a single one-shot image.

Given a prompt, lodestar generates `n` candidates through a pluggable
`Backend`, scores each against the prompt with a `Scorer` (the reward — the
lodestar), and returns the highest-scoring one. An optional bounded re-roll
regenerates the single worst candidate once. That generate → score → select
loop is the smallest honest agentic pattern: a numeric reward drives a
decision.

**Scope is deliberately bounded.** lodestar owns the loop and the contracts
(`Backend`, `Scorer`) — not a model zoo, not an editor, no unbounded refinement.
The default build ships a deterministic mock backend/scorer so the loop runs
end-to-end with no model weights. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for
the milestones (candle Stable Diffusion backend, CLIP scorer).

[![CI](https://github.com/vbasky/lodestar/actions/workflows/ci.yml/badge.svg)](https://github.com/vbasky/lodestar/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Layout

```
crates/
  lodestar/       # library crate
  lodestar-cli/   # binary crate (clap), installs as `lodestar`
```

## Quick start

```bash
just setup        # one-time: enable the auto-format pre-commit hook
just build        # build the workspace
just test         # run all tests

# Generate 5 candidates with the mock backend, pick the best, write the winner:
just run -- "a lighthouse at dusk" -n 5 --reroll-worst --out winner.ppm
```

If you don't have [`just`](https://github.com/casey/just):
`cargo install just`.

## The CLIP reward (optional `clip` feature)

The default build uses the deterministic mock scorer. The real reward —
`ClipScorer` — scores a candidate by the cosine similarity between the CLIP
image and text embeddings, run through ONNX Runtime (`ort`). It lives behind a
cargo feature so the base build stays weight-free:

```bash
cargo test -p lodestar --features clip   # builds ort + tokenizers; runs the
                                          # preprocessing/math tests (no weights)
```

```rust
use lodestar::{best_of_n, GenerateConfig, ClipScorer};

let scorer = ClipScorer::from_files("model.onnx", "tokenizer.json")?;
let selection = best_of_n(&backend, &scorer, &GenerateConfig::new("a red bicycle"))?;
```

It expects a standard Hugging Face CLIP ONNX export (e.g. `clip-vit-base-patch32`)
whose graph takes `pixel_values`, `input_ids`, `attention_mask` and outputs
`image_embeds` and `text_embeds`. End-to-end scoring against real weights is the
next validation step (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).

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
