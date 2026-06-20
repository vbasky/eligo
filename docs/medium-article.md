# Generate Many, Keep the Best: The Smallest Honest Agentic Loop, in Rust

*How I built eligo — best-of-N image selection as two traits and one loop, with a real Stable Diffusion + CLIP stack on ONNX Runtime, in pure Rust.*

---

Image generators are random. Run the same prompt twice and you get two different pictures — one good, one junk. Everyone knows the workaround: generate a handful, eyeball them, keep the best. It works because *you* are the selection function.

**eligo automates the picking.** That's the entire pitch, and I kept it that small on purpose.

The name is Latin for "I choose / I pick out" — the root of *elect* and *elite*. Out of many candidate images, elect the best one.

## The loop

Give eligo a prompt and it does three things:

1. **Generates** `n` candidate images — the *artist*, a pluggable `Backend`.
2. **Scores** each one against the prompt — the *judge*, a pluggable `Scorer` that returns a number where higher is better.
3. **Selects** the highest-scoring candidate and returns it.

![eligo selection loop: a prompt feeds the Backend (the artist), which produces n candidate images; the Scorer (the judge) gives each a reward; argmax picks the winner](https://raw.githubusercontent.com/vbasky/eligo/main/docs/selection-loop.png)

That generate → score → select loop is the smallest *honest* **agentic** pattern: a numeric reward drives a decision. There's exactly one optional twist — a bounded re-roll — and that's the only loop in the whole system. There is no open-ended "keep refining until it's good," because that's the part of agentic systems that quietly burns money and rarely converges. eligo makes one decision and stops.

## How seeding and the bounded re-roll work

The interesting mechanics live in `best_of_n`, and they're deliberately boring in the best way.

To produce `n` candidates from one prompt, the loop doesn't ask the backend for "something random" `n` times — it seeds candidate *i* with `seed + i`. That gives you `n` genuinely different images that are also **fully reproducible**: same prompt, same base seed, same `n` → byte-identical run. For a generator, randomness is a feature; for a *pipeline*, reproducibility is non-negotiable, and the seeding scheme gives you both.

Once all `n` are scored, selection is a plain `argmax` over the rewards. Then the one optional loop: with `--reroll-worst`, eligo regenerates the single lowest-scoring candidate *once* with a fresh seed, re-scores it, and keeps whichever of the two is better. That's it — one extra generation, bounded, predictable. It nudges the floor up without opening the door to an unbounded refinement spiral. You always know the maximum amount of work a run will do: `n` generations, or `n + 1` with re-roll.

## Two traits are the whole surface

Everything in eligo is an implementation of one of two traits, or a reuse of the embedder:

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

That's the entire extension API. The loop handles seeding and selection; you bring an artist and a definition of "best." Want a different generator — SDXL, Flux, a hosted diffusion API, even a non-AI renderer? Implement `Backend`. Want to change what "best" means — aesthetics, face presence, brand-safety, NSFW filtering, OCR legibility, a target palette? Implement `Scorer`.

## Real models are opt-in — the default build runs dry

Here's a decision I'm happy with: the default build needs **no AI models and no native runtime.** It ships a deterministic mock backend and scorer, so the whole loop compiles, runs, and tests green out of the box. You can read and hack the selection logic without downloading a single weight or linking a native library. `MockBackend` and `MockScorer` aren't an afterthought — they're how the core stays understandable and how CI stays fast.

Real models are opt-in cargo features:

- **Default build** — a mock backend + mock scorer; no runtime, no weights.
- **`clip`** — adds `ClipScorer` (the real judge) and `ClipEmbedder`; runs on ONNX Runtime (`ort`).
- **`sd`** — adds `SdBackend`, Stable Diffusion txt2img (the real artist); also ONNX Runtime (`ort`).

## Inside the judge: how CLIP scoring actually works

CLIP is the right tool for "does this image match these words" because it was trained to put images and their captions near each other in a shared embedding space. `ClipScorer` leans on exactly that:

1. Embed the prompt text into a vector with CLIP's text encoder.
2. Embed the candidate image into a vector with CLIP's image encoder.
3. L2-normalise both and take their **cosine similarity** — that scalar *is* the reward.

A candidate that depicts the prompt lands near the prompt's text vector and scores high; an off-prompt image lands far away and scores low. No training, no fine-tuning, no reference image — just a dot product in a well-shaped space.

The model is a standard ONNX export: a CLIP `model.onnx` plus its `tokenizer.json`. Running it through `ort` (ONNX Runtime) rather than a Python process is the whole point — the judge is a native call inside the same binary, no interpreter, no IPC, no `pip` environment to reproduce on a server.

## Inside the artist: Stable Diffusion as three ONNX graphs

`SdBackend` runs Stable Diffusion txt2img from standard diffusers ONNX exports — a directory with three graphs:

- **`text_encoder`** turns the prompt into a conditioning embedding,
- **`unet`** is the denoiser, run for `--steps` iterations to walk a noised latent toward an image that matches that conditioning,
- **`vae_decoder`** turns the final latent into actual RGB pixels.

The candidate's `seed` initialises the latent noise, which is why `seed + i` gives distinct-but-reproducible candidates. Put the artist and the judge together and eligo generates real images and keeps the one CLIP rates best:

```bash
cargo run -p eligo-cli --features "sd clip" -- generate \
  "a photograph of a red apple on a wooden table" -n 4 --steps 20 \
  --sd-model-dir <sd-onnx-dir> --sd-tokenizer <tokenizer.json> \
  --clip-model <clip.onnx> --clip-tokenizer <tokenizer.json> \
  --out winner.png --save-all
```

`--save-all` writes every candidate, not just the winner, so you can *see* what the judge chose between — invaluable when you're deciding whether the scorer is actually picking the image you'd pick.

## "Matches the words" isn't "looks good"

CLIP answers one question: does this image match the prompt? But a blurry, washed-out image can still match the words. So eligo carries a second signal — a no-reference **quality** score built from sharpness and contrast, always in the core and needing no model — and lets you blend the two:

```rust
use eligo::{ClipScorer, QualityWeighted};

let clip = ClipScorer::from_files("clip.onnx", "tokenizer.json")?;
// 70% prompt-match, 30% image quality:
let scorer = QualityWeighted::new(Box::new(clip), 0.3);
```

On the CLI that's `--quality-weight 0.3`. Raise it and eligo prefers crisp, detailed candidates even at a slight cost to literal prompt-match. The neat part: `QualityWeighted` is *itself* just a `Scorer` that wraps another `Scorer`. Composition is the extension mechanism — you don't need a plugin system when "combine two judges" is the same trait wrapping itself.

## A worked example: encoding a business rule as a reward

The power of the `Scorer` trait is that **anything you can turn into a number is a reward.** Suppose your brand palette is built around a warm amber, and out of four on-prompt candidates you want the one closest to that colour. That's not an AI problem — it's a `Scorer`:

```rust
use eligo::{Scorer, Image, Result};

/// Reward images whose average colour is closest to a target brand colour.
struct BrandAffinity { target: [f32; 3] } // RGB, 0..1

impl Scorer for BrandAffinity {
    fn score(&self, _prompt: &str, image: &Image) -> Result<f32> {
        let n = (image.rgb.len() / 3) as f32;
        let mut sum = [0.0f32; 3];
        for px in image.rgb.chunks_exact(3) {
            sum[0] += px[0] as f32; sum[1] += px[1] as f32; sum[2] += px[2] as f32;
        }
        let mean = [sum[0] / n / 255.0, sum[1] / n / 255.0, sum[2] / n / 255.0];
        let dist = ((mean[0] - self.target[0]).powi(2)
                  + (mean[1] - self.target[1]).powi(2)
                  + (mean[2] - self.target[2]).powi(2)).sqrt();
        Ok(1.0 - dist) // closer to target = higher reward
    }
}
```

Drop it straight into the same loop — and because `QualityWeighted` composes, you could just as easily blend this with CLIP to get "on-prompt *and* on-brand":

```rust
use eligo::{best_of_n, GenerateConfig};

let amber = BrandAffinity { target: [0.85, 0.55, 0.10] };
let selection = best_of_n(&my_backend, &amber, &GenerateConfig::new("a coffee cup"))?;
println!("winner seed = {}", selection.best().seed);
```

That's the whole story: every selection policy you can imagine — brand-safety filters, face presence, palette targets, OCR legibility, "not too dark" — is a few lines returning an `f32`.

## The embeddings do double duty

The same CLIP embeddings that judge prompt-match also measure image↔image similarity — the basis for "more like this," dedup, and content-based recommendations. `ClipEmbedder` is factored out so you can use the vectors directly, without going through the scorer:

```rust
use eligo::{ClipEmbedder, cosine_similarity};

let embedder = ClipEmbedder::from_files("clip.onnx", "tokenizer.json")?;
let a = embedder.embed_image(&img_a)?;   // image → L2-normalized vector
let b = embedder.embed_image(&img_b)?;
let how_alike = cosine_similarity(&a, &b);          // in [-1, 1]
```

There's a `similar` subcommand that ranks a folder against a query image, and the clean way to build a recommender on top is a deliberate split of responsibilities: **eligo provides the embedding and the similarity math; the consuming catalogue embeds each asset once, stores the vectors, keeps a nearest-neighbour index (brute-force cosine for thousands, an HNSW index for tens of thousands and up), and adds any per-user signals.** Storage, indexing and personalization stay *outside* eligo so it remains a focused selection library rather than creeping into a database.

## Built bottom-up

eligo grew in deliberate milestones, each one a complete, testable layer: the loop first (M0), then the real judge (M1 — CLIP), then the real artist (M2 — Stable Diffusion), then no-reference quality and blending (M3), then the embeddings and similarity that the judge already produced internally, factored out for reuse (M4). The order matters: the abstraction (two traits, one loop) existed and was tested against mocks *before* any model was wired in, which is why adding a backend or scorer never required touching the core.

## Bounded on purpose

That's the whole philosophy. eligo owns the loop and the two contracts — `Backend` and `Scorer` — and nothing else. It is **not** a model zoo, not an image editor, not a recommendation service. It's the small, sharp piece those things are built *on*.

Most "AI image" libraries grow until they do everything and you can't reason about any of it. I wanted the opposite: a focused selection library where the extension points are two trait methods, the core runs without a single model weight, the work per run is bounded and reproducible, and the one loop it does run is honest about being one decision — not an open-ended refinement spiral.

## Try it

```bash
just run -- generate "a lighthouse at dusk" -n 5 --reroll-worst --out winner.ppm
```

That runs the mock loop — no models, instant. Add `--features "sd clip"` and point it at ONNX weights for the real thing.

It's published: `cargo install eligo-cli`. Source and the full extension guide are on GitHub: **[github.com/vbasky/eligo](https://github.com/vbasky/eligo)**.

If you've been hand-picking the best of five generations, let the argmax do it.
