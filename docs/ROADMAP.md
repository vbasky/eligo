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

## M1 — Real reward: CLIP prompt-alignment `Scorer`

- A `clip` scorer: embed image + prompt, score = cosine similarity.
- Inference via `candle` (or `ort`) loading an open CLIP checkpoint.
- **Done = ** on a fixed prompt + a folder of images, ranking matches a
  reference CLIP implementation within tolerance. This is the first time the
  "best" candidate is meaningfully best.

## M2 — Real backend: Stable Diffusion via `candle`

- A `candle` SD backend implementing `Backend` (txt2img, fixed model, CPU-ok for
  small sizes).
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
