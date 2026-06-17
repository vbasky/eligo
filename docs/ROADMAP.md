# eligo — roadmap

The whole point of eligo is to stay **small and finishable**. Each milestone
is a self-contained increment with a clear "done", and the project is feature-
complete at M3. Nothing here is open-ended.

## M0 — Selection loop + contracts ✅ (scaffolded)

- `Backend` (prompt + seed → `Image`) and `Scorer` (prompt + image → reward)
  traits.
- `best_of_n`: generate `n` candidates, score each, return the highest, with an
  optional **bounded** re-roll of the single worst candidate (exactly once).
- Deterministic `mock` backend/scorer so the loop runs and tests green with no
  model weights.
- CLI: `eligo "<prompt>" -n N [--seed S] [--reroll-worst] [--out file.ppm]`.

**Done = ** loop is reproducible, validated, lint-clean, and demonstrable end to
end (already true).

## M1 — Real reward: CLIP prompt-alignment `Scorer` ✅

- `ClipScorer`: embeds image + prompt, score = cosine similarity.
- Inference via **ONNX Runtime (`ort`)** loading an open CLIP checkpoint —
  the same runtime the rest of the ecosystem uses for ONNX vision models.
- Gated behind a `clip` cargo feature so the default build stays weight-free.
- Wired into the CLI (`--clip-model` / `--clip-tokenizer`) and validated
  end-to-end against real `clip-vit-base-patch32` ONNX weights: given two
  candidate images and a prompt, the matching image out-scores the other in
  both directions (`tests/clip_real.rs`).
- **Done.** This is the first time the "best" candidate is meaningfully best.

## M2 — Real backend: Stable Diffusion ✅

- `SdBackend`: a full latent-diffusion txt2img loop on ONNX Runtime (text
  encoder + UNet + VAE decoder), with a hand-rolled DDIM scheduler,
  classifier-free guidance, and a reproducible seeded RNG. Same `ort` runtime
  and CLIP tokenizer as the scorer — the `sd` feature adds no new dependencies.
- Wired into the CLI (`--sd-model-dir` / `--sd-tokenizer` / `--steps` /
  `--guidance`); `--out foo.png` saves the winner via the `image` crate.
- Validated end-to-end against a vanilla fp32 SD-1.5 ONNX export: `eligo
  "a photograph of a red apple on a wooden table"` produces a recognizable
  image (`tests/sd_real.rs` checks shape, reproducibility, seed-sensitivity,
  non-degeneracy). I/O matched the diffusers convention (UNet `timestep` is
  `Int64`, hidden 768).
- **Done.** Combined with M1, `eligo --features "sd clip" "<prompt>" -n 4
  --sd-… --clip-…` generates four real images and keeps the one CLIP rates best.

## M3 — Blend in a quality term (optional, still bounded) ✅

- `QualityScorer` + `quality_score`: a no-reference image-quality signal
  (Laplacian sharpness + RMS contrast, the no-reference family that BRISQUE/NIQE
  build on). Parameter-free — no model, no weights — so it lives in the always-on
  core.
- `QualityWeighted`: wraps any `Scorer` and blends `(1-w)*base + w*quality`.
  Exposed on the CLI as `--quality-weight`.
- Unit-tested: a blurred image scores lower than its sharp original, a flat
  image scores ~0, blending stays between base and quality.
- A *reference-calibrated* BRISQUE/NIQE (with a trained model) is deliberately
  **not** in scope here — that's the standalone "Option B" project (see the
  ecosystem gap notes). eligo only needs a sound relative ordering among
  same-size candidates.
- **Done.** The chosen candidate can optimize "matches the prompt **and** looks
  clean," not alignment alone.

## M4 — Reusable embeddings + image↔image similarity ✅

A natural extension once CLIP was in place: the same embeddings that score
prompt-match also measure *image-to-image* similarity — the foundation for
"more like this" and content-based recommendations.

- `ClipEmbedder` factored out of `ClipScorer`: `embed_image` / `embed_text`
  (L2-normalized vectors), `embed_both` (one run for scoring), and
  `image_similarity` (cosine of two images). `ClipScorer` now builds on it.
- `Image::open` (load any format) so the CLI can read existing files.
- CLI restructured into subcommands: `generate` (best-of-N) and `similar`
  (rank a folder against a query image). Validated on real images — the query
  self-matches at exactly 1.0000.
- **Boundary:** eligo provides *embeddings + similarity*. Corpus storage, the
  nearest-neighbour index (e.g. HNSW at scale), and per-user recommendations
  belong to whatever catalogue consumes eligo — not here.

## Explicit non-goals (the boundary)

- ❌ Inpainting, masking, img2img, or multi-step editing.
- ❌ An LLM planner / tool-using agent. (eligo is the *reward loop*; an LLM
  agent that *calls* eligo belongs in the orchestration layer above it, not
  here.)
- ❌ Training, fine-tuning, or LoRA management.
- ❌ A model zoo — one backend implementation at a time.
- ❌ Unbounded "keep refining until good" loops. Re-roll is one extra draw, full
  stop.
