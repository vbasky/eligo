# lodestar — roadmap

The whole point of lodestar is to stay **small and finishable**. Each milestone
is a self-contained increment with a clear "done", and the project is feature-
complete at M3. Nothing here is open-ended.

## M0 — Selection loop + contracts ✅ (scaffolded)

- `Backend` (prompt + seed → `Image`) and `Scorer` (prompt + image → reward)
  traits.
- `best_of_n`: generate `n` candidates, score each, return the highest, with an
  optional **bounded** re-roll of the single worst candidate (exactly once).
- Deterministic `mock` backend/scorer so the loop runs and tests green with no
  model weights.
- CLI: `lodestar "<prompt>" -n N [--seed S] [--reroll-worst] [--out file.ppm]`.

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

## M2 — Real backend: Stable Diffusion

- A Stable Diffusion backend implementing `Backend` (txt2img, fixed model,
  CPU-ok for small sizes).
- Real PNG output (swap the placeholder PPM writer for the `image` crate).
- **Done = ** `lodestar "<prompt>" -n 4` produces four real images and picks the
  one CLIP says best matches the prompt.

## M3 — Blend in a quality term (optional, still bounded)

- Add a no-reference quality signal (BRISQUE/NIQE) as a second `Scorer`, and a
  `WeightedScorer` that blends alignment + quality with a fixed weight.
- This is shared ground with `viser`'s no-reference QC mission — reuse, don't
  reinvent, if viser exposes it.
- **Done = ** the chosen candidate optimizes "matches the prompt **and** looks
  clean," not alignment alone. **Project is feature-complete here.**

## Explicit non-goals (the boundary)

- ❌ Inpainting, masking, img2img, or multi-step editing.
- ❌ An LLM planner / tool-using agent. (lodestar is the *reward loop*; an LLM
  agent that *calls* lodestar belongs in the orchestration layer above it, not
  here.)
- ❌ Training, fine-tuning, or LoRA management.
- ❌ A model zoo — one backend implementation at a time.
- ❌ Unbounded "keep refining until good" loops. Re-roll is one extra draw, full
  stop.
