# eligo

![eligo — generate many images, keep the best; best-of-N selection in pure Rust](https://raw.githubusercontent.com/vbasky/eligo/main/docs/banner.png)

**Name:** *eligo* is Latin for **"I choose / I pick out"** — the root of *elect*
and *elite*. It names the tool's one job: out of many candidate images, **elect
the best one.**

[![crates.io](https://img.shields.io/crates/v/eligo?logo=rust&color=orange)](https://crates.io/crates/eligo)
[![docs.rs](https://img.shields.io/docsrs/eligo?logo=docsdotrs)](https://docs.rs/eligo)
[![CI](https://img.shields.io/github/actions/workflow/status/vbasky/eligo/ci.yml?branch=main&logo=github&label=CI)](https://github.com/vbasky/eligo/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.85.1-blue)](https://www.rust-lang.org)

---

## What it does

Image generators are random — every attempt comes out different, some good, some
junk. The usual fix is to make several and let a human pick. **eligo automates
the picking.**

Give it a prompt and it:

1. **Generates** `n` candidate images (the *artist* — a pluggable `Backend`).
2. **Scores** each one against the prompt (the *judge* — a pluggable `Scorer`
   that returns a number; higher is better).
3. **Selects** the highest-scoring candidate and returns it.

![eligo selection loop: a prompt feeds the Backend (the artist), which produces n candidate images; the Scorer (the judge) gives each a reward; argmax picks the winner](https://raw.githubusercontent.com/vbasky/eligo/main/docs/selection-loop.png)

That generate → score → select loop is the smallest honest **agentic** pattern:
a numeric reward drives a decision. An optional bounded re-roll regenerates the
single worst candidate once — and that's the only loop; there is no open-ended
"keep refining."

**Scope is deliberately bounded.** eligo owns the loop and the two contracts
(`Backend`, `Scorer`). It is not a model zoo, not an editor, and not a
recommendation service — but its parts are the foundation you build those on
(see [Extending eligo](#extending-eligo)).

## Install / layout

```bash
crates/
  eligo/       # library: the loop, traits, scorers, embedder
  eligo-cli/   # binary (clap), installs as `eligo`
```

The default build needs **no AI models and no native runtime** — it ships a
deterministic mock backend and scorer so the whole loop runs and tests green out
of the box. Real models are opt-in cargo features:

| Feature | Adds | Runtime |
| --- | --- | --- |
| *(default)* | mock backend + mock scorer | none |
| `clip` | `ClipScorer` (the real judge) + `ClipEmbedder` | ONNX Runtime (`ort`) |
| `sd` | `SdBackend` — Stable Diffusion txt2img (the real artist) | ONNX Runtime (`ort`) |

## Quick start

```bash
just build
just test

# Mock loop — no models, instant. Generate 5, keep the best, write the winner:
just run -- generate "a lighthouse at dusk" -n 5 --reroll-worst --out winner.ppm
```

The CLI has two subcommands: **`generate`** (best-of-N) and **`similar`**
(find look-alike images). If you don't have [`just`](https://github.com/casey/just),
`cargo install just` or use the `cargo run -p eligo-cli -- …` forms below.

## The full thing: real images, real selection

With both features on, eligo generates actual Stable Diffusion images and keeps
the one CLIP judges best:

```bash
cargo run -p eligo-cli --features "sd clip" -- generate \
  "a photograph of a red apple on a wooden table" -n 4 --steps 20 \
  --sd-model-dir <sd-onnx-dir> --sd-tokenizer <tokenizer.json> \
  --clip-model <clip.onnx> --clip-tokenizer <tokenizer.json> \
  --out winner.png --save-all
```

- **`--features sd`** — the artist (`SdBackend`): turns the prompt into images.
- **`--features clip`** — the judge (`ClipScorer`): scores each against the prompt.
- **`--quality-weight 0.3`** — also reward sharp, clean images (see below).
- **`--save-all`** — write every candidate, not just the winner, so you can see
  what the judge chose between.

Models are standard ONNX exports: a diffusers `text_encoder` / `unet` /
`vae_decoder` directory for SD, and a CLIP `model.onnx` + `tokenizer.json`. Both
are validated end-to-end in `crates/eligo/tests/{sd_real,clip_real}.rs` (ignored
by default; pointed at weights via env vars).

## Scoring beyond the prompt: no-reference quality

CLIP answers *"does this match the words?"* — but a blurry image can still match
the words. The `quality` signal (always in the core, no model needed) answers
*"does it look good?"* using sharpness + contrast. Blend the two:

```rust
use eligo::{ClipScorer, QualityWeighted};

let clip = ClipScorer::from_files("clip.onnx", "tokenizer.json")?;
// 70% prompt-match, 30% image quality:
let scorer = QualityWeighted::new(Box::new(clip), 0.3);
```

On the CLI that's `--quality-weight 0.3`. Raising it makes eligo prefer crisp,
detailed candidates even at a slight cost to literal prompt-match.

## Find similar images (`similar`)

The same CLIP embeddings that judge prompt-match also measure *image↔image*
similarity — the basis for "more like this", dedup, and content-based
recommendations. `ClipEmbedder::embed_image` turns an image into a vector;
nearby vectors are look-alikes. The `similar` subcommand ranks a folder against
a query image:

```bash
cargo run -p eligo-cli --features clip -- similar \
  query.png ./photos -k 5 \
  --clip-model clip.onnx --clip-tokenizer tokenizer.json
```

```text
most similar to query.png:
  1.0000  ./photos/query.png      # itself
  0.9238  ./photos/other_a.png
  0.9101  ./photos/other_b.png
```

---

## Extending eligo

eligo is built around two small traits. Everything else — real models, quality
blending, similarity — is an implementation of one of them, or a reuse of the
embedder. Here is the whole surface you extend against:

```rust
/// The artist: prompt + seed → image.
pub trait Backend {
    fn generate(&self, prompt: &str, seed: u64) -> Result<Image>;
}

/// The judge: prompt + image → reward (higher is better).
pub trait Scorer {
    fn score(&self, prompt: &str, image: &Image) -> Result<f32>;
}

pub fn best_of_n(backend: &dyn Backend, scorer: &dyn Scorer, cfg: &GenerateConfig)
    -> Result<Selection>;
```

| You want to… | Do this |
| --- | --- |
| Use a different generator (SDXL, Flux, a diffusion API, even a non-AI renderer) | implement **`Backend`** |
| Change what "best" means (aesthetics, face presence, brand-safety, NSFW filter, OCR legibility, palette) | implement **`Scorer`** |
| Combine several rewards | wrap with **`QualityWeighted`**, or write a composing `Scorer` |
| Build "more like this", dedup, or search | use **`ClipEmbedder::embed_image`** + `cosine_similarity` |
| Power a recommendation engine / media catalogue | embed assets once, store the vectors, do nearest-neighbour lookups *outside* eligo |
| Add a new no-reference metric | sit it next to `quality_score` and blend it in |

### 1. A custom backend (your own artist)

Return an RGB8 [`Image`]; the loop handles seeding (candidate *i* gets
`seed + i`) and selection for you.

```rust
use eligo::{Backend, Image, Result};

struct MyApiBackend { client: MyClient }

impl Backend for MyApiBackend {
    fn generate(&self, prompt: &str, seed: u64) -> Result<Image> {
        let pixels = self.client.txt2img(prompt, seed)?; // your model / service
        Image::new(width, height, pixels)                // RGB8, row-major
    }
}
```

### 2. A custom scorer (your own definition of "best")

Anything you can turn into a number is a reward. The prompt is provided in case
you want it; ignore it for prompt-independent rewards.

```rust
use eligo::{Scorer, Image, Result};

/// Prefer images that are mostly *not* dark.
struct PreferBright;

impl Scorer for PreferBright {
    fn score(&self, _prompt: &str, image: &Image) -> Result<f32> {
        let mean = image.rgb.iter().map(|&b| b as f32).sum::<f32>()
            / image.rgb.len() as f32;
        Ok(mean / 255.0) // 0..1, brighter = higher
    }
}
```

Drop either into the same loop:

```rust
use eligo::{best_of_n, GenerateConfig};

let selection = best_of_n(&MyApiBackend { .. }, &PreferBright, &GenerateConfig::new("a sunset"))?;
println!("winner seed = {}", selection.best().seed);
```

### 3. Similarity & recommendations (reuse the embedder)

`ClipEmbedder` is factored out so you can use the embeddings directly — no need
to go through the scorer:

```rust
use eligo::{ClipEmbedder, cosine_similarity};

let embedder = ClipEmbedder::from_files("clip.onnx", "tokenizer.json")?;
let a = embedder.embed_image(&img_a)?;   // image → L2-normalized vector
let b = embedder.embed_image(&img_b)?;
let how_alike = cosine_similarity(&a, &b);          // in [-1, 1]
// or: embedder.image_similarity(&img_a, &img_b)?
```

To build a recommender on top, the clean split is: **eligo provides the
embedding and the similarity math**; the consuming catalogue embeds each asset
once, **stores** the vectors, keeps a nearest-neighbour **index** (brute-force
cosine for thousands; an HNSW index for tens of thousands+), and adds any
**per-user** signals. Storage, indexing, and personalization stay out of eligo
so it remains a focused selection library.

### Reusable parts

| Item | Use |
| --- | --- |
| `Image` | RGB8 buffer; `Image::open` / `Image::save_png` (with `clip`/`sd`) |
| `cosine_similarity`, `l2_normalize` | vector math for any embedding |
| `quality_score` / `QualityScorer` | no-reference sharpness/contrast quality |
| `mock::{MockBackend, MockScorer}` | deterministic stand-ins for tests |

## Development

`just check-all` runs the exact gate CI enforces — formatting, clippy
(`-D warnings`), tests, and docs — before you push.

| Task | Command |
| --- | --- |
| Format | `just fmt` |
| Lint | `just lint` |
| Test | `just test` |
| Test a feature | `cargo test -p eligo --features clip` |
| Docs | `just docs` |
| Dependency audit | `just deny` (needs `cargo install cargo-deny`) |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the milestone history (M0 loop → M1
judge → M2 artist → M3 quality → M4 embeddings/similarity) and the explicit
non-goals.

## Releasing

1. Update `CHANGELOG.md` under a new `## [x.y.z]` heading and commit.
2. `just release x.y.z` — bumps versions, tags, and pushes.
3. CI builds binaries for macOS (arm64 + x86_64), Linux, and Windows, and
   publishes a GitHub Release with checksums and the changelog notes.
4. To also publish to crates.io: `PUBLISH=1 just release x.y.z`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
