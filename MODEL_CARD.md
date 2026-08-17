---
license: apache-2.0
base_model: mlx-community/Qwen3.5-4B-MLX-4bit
tags:
  - image-localization
  - template-matching
  - semiconductor
  - sem
  - siamese-network
  - lora
  - mlx
library_name: pytorch
---

# Drift-Sense Localization

Models that find a zoomed-in SEM reference pattern (100x, 1 nm/px) inside a
zoomed-out search image (10x, 10 nm/px) and return its pixel location.
Trained on 400 synthetic FinFET SEM pairs (IRDS 2024 geometry) with
realistic noise, blur, rotation (±2°), and stage drift (100–500 nm).

## Files

| file | what it is |
|---|---|
| `cnn/best.pt` | Siamese correlation CNN (PyTorch state_dict, ~6M params) |
| `lora/adapters.safetensors` | LoRA adapter (rank 8) for Qwen3.5-4B |
| `lora/adapter_config.json` | Adapter config for mlx-vlm |

## Results (held-out test split, 50 pairs)

| model | mean err (px) | ≤1 px | ≤5 px | s/pair |
|---|---|---|---|---|
| Siamese CNN + sub-pixel refine | 0.50 | 96% | 98% | 0.05 |
| Siamese CNN (coarse) | 2.51 | 10% | 96% | 0.04 |
| Qwen3.5-4B LoRA | 35.79 | 0% | 0% | 1.83 |

1 px = 10 nm. With phase-correlation refinement the CNN localizes to ~5 nm
mean error. A blind nominal-spot baseline scores 34.9 px; the VLM does not
beat it.

## Usage

CNN (needs the [training repo](https://github.com/bardrop/drift-sense) on PYTHONPATH):

```python
import torch
from drift_sense.model import SiameseLocator
from drift_sense.predict import predict

out = predict("ref.png", "search.png", ckpt="best.pt")
# {'x': 273.1, 'y': 262.4, 'confidence': 0.77, 'found': True,
#  'message': 'Pattern found at (273, 262) in the search image. Confidence: 77%.'}
```

VLM adapter (with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)):

```bash
python -m mlx_vlm.generate \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --adapter-path lora \
  --image ref.png search.png \
  --prompt "The first image is a zoomed-in reference pattern. The second image is a zoomed-out search image (1000x1000 pixels). Find the reference pattern in the search image. If several matches exist, pick the one closest to the centre. Answer exactly: Pattern found at (x, y). Confidence: high." \
  --max-tokens 40
```

## Design notes

- The die is periodic (9 identical rail crossings), so pure correlation is
  ambiguous. The CNN adds a learnable position-prior bias map and trains
  with softmax cross-entropy over the full heatmap so peaks compete.
- Confidence = probability mass near the chosen peak; ambiguous matches
  score low. Ties break toward the image centre.
- The VLM run is an honest negative result: it learns the output format
  (0% parse failures) but not pixel-level precision.

## Limitations

- Trained on synthetic SEM images only; not validated on real SEM data.
- Assumes drift within ~±50 px of the nominal position (as generated).
- Rotation handled implicitly up to ±2°; no rotation estimate is output.
