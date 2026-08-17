# Drift-Sense: SEM Pattern Localization

Find where a zoomed-in SEM reference image (100x, 1 nm/px) sits inside a
zoomed-out search image (10x, 10 nm/px). Built for Applied Materials
Problem Statement 02, Hackathon 2026.

Two models, same task, honest comparison:

| model | mean err (px) | ≤1 px | ≤5 px | s/pair |
|---|---|---|---|---|
| **Siamese CNN + sub-pixel refine** | **0.50** | **96%** | **98%** | 0.05 |
| Siamese CNN (coarse) | 2.51 | 10% | 96% | 0.04 |
| Qwen3.5-4B LoRA (VLM) | 35.79 | 0% | 0% | 1.83 |

Held-out test split (50 pairs). 1 px = 10 nm, so the refined result is ~5 nm
mean error. A blind always-guess-the-nominal-spot baseline scores 34.9 px —
the VLM does not beat it.

## How it works

**Siamese CNN** (`drift_sense/`): one shared conv encoder embeds both
images, the reference features cross-correlate over the search features,
and the peak of the resulting heatmap is the location. Two details matter:

- The die is periodic — 9 rail crossings look identical. Correlation alone
  cannot pick the right one, so a **learnable position prior** (a bias map
  added to the heatmap) encodes "the true match is near the nominal spot".
- Training uses **softmax cross-entropy over the whole heatmap**, so
  competing peaks fight each other directly. Per-pixel BCE fails here.

Confidence is the probability mass near the chosen peak. Ambiguous matches
score low; below 0.5 the CLI reports "no pattern found". If several peaks
are near-equal, the one closest to the image centre wins.

A final **sub-pixel stage** (`drift_sense/refine.py`) polishes the CNN's
answer: it crops a 100×100 window from the full-resolution search image and
phase-correlates it against the 10x-downscaled reference
(`cv2.phaseCorrelate`, Hanning window). This takes 2.5 px error down to
0.5 px, with a fallback to the coarse answer when the correction is
implausibly large.

**VLM** (`vlm/`): LoRA fine-tune of Qwen3.5-4B (4-bit, MLX) that answers
in English: `Pattern found at (243, 241). Confidence: high.` It learns the
format perfectly (0% parse failures) but not fine localization — a good
demonstration of why a 6M-param task-specific CNN beats a 4B-param
generalist on precision work.

## Quickstart

```bash
uv sync
uv run python main.py                      # generate the 500-pair dataset
uv run python -m drift_sense.train         # train the CNN (~15 min on M-series)
uv run python -m drift_sense.predict \
  --ref dataset_drift_sense/reference/ref_500.png \
  --search dataset_drift_sense/search/search_500.png
# -> Pattern found at (273, 262) in the search image. Confidence: 77%.
```

VLM (optional, ~30 min fine-tune):

```bash
uv run python -m vlm.make_vlm_dataset
uv run python -m vlm.finetune
uv run python -m vlm.eval
```

Compare both: `uv run python compare.py`. Tests: `uv run pytest`.

## Weights

Trained weights are on Hugging Face:
[msssingh/drift-sense-localization](https://huggingface.co/msssingh/drift-sense-localization)
— `cnn/best.pt` (CNN checkpoint) and `lora/` (LoRA adapter for
`mlx-community/Qwen3.5-4B-MLX-4bit`).

## Layout

- `main.py` — synthetic FinFET SEM dataset generator (IRDS 2024 geometry)
- `drift_sense/` — dataset, model, training, prediction CLI
- `vlm/` — VLM chat-dataset builder, LoRA fine-tune, eval
- `compare.py` — side-by-side metrics
- `docs/superpowers/` — design spec and implementation plan
