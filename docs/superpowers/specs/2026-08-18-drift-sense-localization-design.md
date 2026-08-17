# Drift-Sense Pattern Localization — Design

Date: 2026-08-18
Status: Approved in chat, pending spec review

## Goal

Given a zoomed-in reference SEM image (100x, 1000×1000 px, 1 nm/px) and a
zoomed-out search image (10x, 1000×1000 px, 10 nm/px), return where the
reference pattern sits in the search image: `(x, y)` in search-image pixels,
plus a confidence score, plus a plain-English sentence. If several candidate
peaks are equally strong, pick the one closest to the search-image centre.
Training and scoring always use `manifest.csv` ground truth.

## Data

- `dataset_drift_sense/`: 500 pairs from `main.py`.
  - `reference/ref_XXX.png` — clean-ish crop around a power-rail crossing.
  - `search/search_XXX.png` — whole die, rotated 1–2°, blurred, noisy.
  - `manifest.csv` — `ground_truth_x`, `ground_truth_y` in search px.
- Split: 400 train / 50 val / 50 test (fixed seed, by pair_id).

## Part A — Siamese CNN (primary model)

- PyTorch, MPS device.
- Reference downscaled 10x to 100×100 (matches search scale), then 2x more
  to 50×50. Search downscaled 2x to 500×500 for speed; predicted coords
  are multiplied by 2 back to search-image pixels.
- Shared-weight conv encoder on both images → feature maps →
  cross-correlation of reference features over search features → heatmap.
- Loss: BCE/focal against a Gaussian blob centred on ground truth.
- Inference: peak of heatmap = `(x, y)`; peak strength (after normalization)
  = confidence; threshold gives "no pattern found". Near-equal peaks
  (within a margin): choose the one nearest image centre.
- Metrics: mean pixel error, % within 5 px, % within 10 px on test split.

Files:

- `drift_sense/dataset.py` — pair loading, split, downscale, blob targets.
- `drift_sense/model.py` — encoder + correlation head.
- `drift_sense/train.py` — training loop, checkpoints, val metrics.
- `drift_sense/predict.py` — CLI: ref + search in, `(x, y)` + confidence +
  English sentence out.

## Part B — Fine-tuned VLM (natural-language model)

- Base: `Qwen/Qwen3.5-4B` (official, multimodal), 4-bit MLX
  (`mlx-community/Qwen3.5-4B-MLX-4bit`), LoRA fine-tune with `mlx-vlm`.
- Dataset: chat samples — user turn holds both images (downscaled to fit
  context) + instruction; assistant turn holds the answer text, e.g.
  `"Pattern found at (243, 241). Confidence: high."` Coordinates from
  manifest. Same 400/50/50 split.
- Eval: parse `(x, y)` from generated text; same pixel-error metrics, so the
  VLM is directly comparable with the CNN.

Files:

- `vlm/make_vlm_dataset.py` — manifest → mlx-vlm chat-format JSONL.
- `vlm/finetune.py` — LoRA config + training call.
- `vlm/eval.py` — generate on test split, parse coords, score.

## Comparison

- `compare.py` — table: CNN vs VLM, mean error, % ≤ 5 px, % ≤ 10 px,
  runtime per pair.

## Error handling

- Confidence below threshold → "No pattern found" (both models).
- VLM output that fails coordinate parsing → counted as a miss in eval,
  reported separately as parse-failure rate.

## Testing

- Unit tests for dataset coordinate math (blob position ↔ manifest GT,
  downscale rescaling round-trip).
- Held-out 50-pair test split is the acceptance gate for both models.

## Out of scope

- Rotation estimation output (dataset rotation is small; models absorb it).
- Non-synthetic SEM images.
- Backward compatibility of any intermediate script.
