# Drift-Sense Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find where a zoomed-in SEM reference pattern sits in a zoomed-out search image, via (A) a Siamese correlation CNN and (B) a LoRA fine-tuned Qwen3.5-4B VLM that answers in English.

**Architecture:** Part A: shared-weight conv encoder on both images, cross-correlate reference features over search features, heatmap peak = location, soft-argmax refinement, centre tie-break. Part B: manifest → chat JSONL (two images + English answer with coordinates), LoRA fine-tune 4-bit Qwen3.5-4B with mlx-vlm, eval by parsing coordinates from generated text.

**Tech Stack:** Python 3.13, uv, PyTorch (MPS), OpenCV, pytest, mlx-vlm, model `mlx-community/Qwen3.5-4B-MLX-4bit`.

**Spec:** docs/superpowers/specs/2026-08-18-drift-sense-localization-design.md

## Global Constraints

- Dataset root: `dataset_drift_sense/` (500 pairs, `manifest.csv` has `ground_truth_x`, `ground_truth_y` in search-image pixels, 1000×1000).
- Split: sort by `pair_id`; first 400 train, next 50 val, last 50 test. Deterministic, no shuffle.
- Ground truth from manifest ALWAYS wins. "Closest to centre" is only a tie-break among near-equal peaks.
- Device: `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.
- Run everything with `uv run`. Add deps with `uv add`, never pip.
- No backward-compatibility effort anywhere.

---

### Task 1: Project setup + manifest loading and split

**Files:**
- Create: `drift_sense/__init__.py` (empty)
- Create: `drift_sense/dataset.py`
- Test: `tests/test_dataset.py`
- Modify: `pyproject.toml` (via `uv add`)

**Interfaces:**
- Produces: `load_manifest(path: str) -> list[dict]` — each dict: `{"pair_id": int, "ref_path": str, "search_path": str, "gt_x": float, "gt_y": float}`. Paths are as stored in the CSV (relative to repo root).
- Produces: `split_pairs(rows: list[dict]) -> tuple[list, list, list]` — (train 400, val 50, test 50), sorted by pair_id.

- [ ] **Step 1: Init git and add deps**

```bash
cd /Users/mradulsingh/hackathon
git init
printf '.venv/\n__pycache__/\ncheckpoints/\nvlm/adapters/\n*.pt\n' > .gitignore
git add -A && git commit -m "chore: baseline before ML work"
uv add torch pytest
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_dataset.py
from drift_sense.dataset import load_manifest, split_pairs

MANIFEST = "dataset_drift_sense/manifest.csv"

def test_load_manifest():
    rows = load_manifest(MANIFEST)
    assert len(rows) == 500
    r = rows[0]
    assert r["pair_id"] == 1
    assert r["ref_path"].endswith("ref_001.png")
    assert 0 <= r["gt_x"] <= 1000 and 0 <= r["gt_y"] <= 1000

def test_split_sizes_and_determinism():
    rows = load_manifest(MANIFEST)
    train, val, test = split_pairs(rows)
    assert (len(train), len(val), len(test)) == (400, 50, 50)
    assert train[0]["pair_id"] == 1
    assert val[0]["pair_id"] == 401
    assert test[-1]["pair_id"] == 500
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: drift_sense`

- [ ] **Step 4: Implement**

```python
# drift_sense/dataset.py
import csv

def load_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "pair_id": int(r["pair_id"]),
                "ref_path": r["reference_path"],
                "search_path": r["search_path"],
                "gt_x": float(r["ground_truth_x"]),
                "gt_y": float(r["ground_truth_y"]),
            })
    return rows

def split_pairs(rows):
    rows = sorted(rows, key=lambda r: r["pair_id"])
    return rows[:400], rows[400:450], rows[450:]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v` — Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: manifest loading and deterministic split"
```

---

### Task 2: Torch dataset with coordinate mapping and heatmap targets

**Files:**
- Modify: `drift_sense/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_manifest`, `split_pairs` (Task 1).
- Produces constants: `SEARCH_SIZE = 500`, `REF_SIZE = 50`, `STRIDE = 4`, `HEAT = 125` (heatmap side), `SCALE = 2.0` (search px per 500-scale px).
- Produces: `search_px_to_heat(x: float) -> float` = `x / (STRIDE * SCALE) - 0.5` and `heat_to_search_px(h: float) -> float` = `(h + 0.5) * STRIDE * SCALE`.
- Produces: `make_target(gt_x, gt_y, sigma=1.5) -> np.ndarray` shape (HEAT, HEAT) float32, Gaussian blob peaking at the mapped GT.
- Produces: `DriftPairDataset(rows: list[dict], root: str = ".")` — `__getitem__` returns `(ref: FloatTensor (1,50,50), search: FloatTensor (1,500,500), target: FloatTensor (125,125), gt: FloatTensor (2,))` with `gt` in original search px. Images loaded grayscale, resized with `cv2.INTER_AREA` (ref 1000→50, search 1000→500), scaled to [0,1].

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_dataset.py
import numpy as np
from drift_sense.dataset import (
    DriftPairDataset, make_target, search_px_to_heat, heat_to_search_px,
    HEAT, load_manifest,
)

def test_coord_round_trip():
    for x in [0.0, 242.25, 500.0, 999.0]:
        assert abs(heat_to_search_px(search_px_to_heat(x)) - x) < 1e-6

def test_target_peak_matches_gt():
    t = make_target(242.25, 240.71)
    assert t.shape == (HEAT, HEAT)
    hy, hx = np.unravel_index(t.argmax(), t.shape)
    assert abs(heat_to_search_px(hx) - 242.25) <= 8.0
    assert abs(heat_to_search_px(hy) - 240.71) <= 8.0
    assert t.max() <= 1.0 and t.min() >= 0.0

def test_dataset_item_shapes():
    rows = load_manifest("dataset_drift_sense/manifest.csv")[:2]
    ds = DriftPairDataset(rows)
    ref, search, target, gt = ds[0]
    assert tuple(ref.shape) == (1, 50, 50)
    assert tuple(search.shape) == (1, 500, 500)
    assert tuple(target.shape) == (125, 125)
    assert float(gt[0]) == rows[0]["gt_x"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v` — Expected: new tests FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# append to drift_sense/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SEARCH_SIZE = 500
REF_SIZE = 50
STRIDE = 4
HEAT = SEARCH_SIZE // STRIDE  # 125
SCALE = 1000 / SEARCH_SIZE    # 2.0

def search_px_to_heat(x):
    return x / (STRIDE * SCALE) - 0.5

def heat_to_search_px(h):
    return (h + 0.5) * STRIDE * SCALE

def make_target(gt_x, gt_y, sigma=1.5):
    cx, cy = search_px_to_heat(gt_x), search_px_to_heat(gt_y)
    ys, xs = np.mgrid[0:HEAT, 0:HEAT].astype(np.float32)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).astype(np.float32)

class DriftPairDataset(Dataset):
    def __init__(self, rows, root="."):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def _load(self, path, size):
        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

    def __getitem__(self, i):
        r = self.rows[i]
        ref = self._load(r["ref_path"], REF_SIZE)
        search = self._load(r["search_path"], SEARCH_SIZE)
        target = torch.from_numpy(make_target(r["gt_x"], r["gt_y"]))
        gt = torch.tensor([r["gt_x"], r["gt_y"]], dtype=torch.float32)
        return ref, search, target, gt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v` — Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: torch dataset, coord mapping, gaussian targets"
```

---

### Task 3: Siamese correlation model

**Files:**
- Create: `drift_sense/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: constants from `drift_sense.dataset` (REF_SIZE, SEARCH_SIZE, HEAT).
- Produces: `SiameseLocator(torch.nn.Module)` with `forward(ref: (B,1,50,50), search: (B,1,500,500)) -> logits (B,125,125)`.
- Produces: `soft_argmax_peak(logits: (125,125) tensor) -> tuple[float, float, float]` returning `(hx, hy, confidence)` — peak refined in a 5×5 window, confidence = `sigmoid(max logit)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
import torch
from drift_sense.model import SiameseLocator, soft_argmax_peak

def test_forward_shapes():
    m = SiameseLocator()
    logits = m(torch.rand(2, 1, 50, 50), torch.rand(2, 1, 500, 500))
    assert tuple(logits.shape) == (2, 125, 125)

def test_soft_argmax_finds_planted_peak():
    logits = torch.full((125, 125), -8.0)
    logits[40, 70] = 6.0
    hx, hy, conf = soft_argmax_peak(logits)
    assert abs(hx - 70) < 1.0 and abs(hy - 40) < 1.0
    assert conf > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# drift_sense/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class SiameseLocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _block(1, 16), _block(16, 16, stride=2),
            _block(16, 32), _block(32, 32, stride=2),
            _block(32, 64), _block(64, 64),
        )  # total stride 4: ref 50->13, search 500->125
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))

    def forward(self, ref, search):
        fr = F.normalize(self.encoder(ref), dim=1)      # (B,64,13,13)
        fs = F.normalize(self.encoder(search), dim=1)   # (B,64,125,125)
        outs = []
        for i in range(ref.shape[0]):
            corr = F.conv2d(fs[i : i + 1], fr[i].unsqueeze(0), padding=6)
            outs.append(corr[0, 0] / fr[i].numel() ** 0.5)
        heat = torch.stack(outs)                        # (B,125,125)
        return heat * self.scale + self.bias

def soft_argmax_peak(logits):
    h, w = logits.shape
    flat = logits.argmax()
    py, px = int(flat // w), int(flat % w)
    y0, y1 = max(0, py - 2), min(h, py + 3)
    x0, x1 = max(0, px - 2), min(w, px + 3)
    win = logits[y0:y1, x0:x1]
    weights = torch.softmax(win.flatten(), dim=0).reshape(win.shape)
    ys = torch.arange(y0, y1, dtype=torch.float32, device=logits.device)
    xs = torch.arange(x0, x1, dtype=torch.float32, device=logits.device)
    hy = float((weights.sum(dim=1) * ys).sum())
    hx = float((weights.sum(dim=0) * xs).sum())
    conf = float(torch.sigmoid(logits[py, px]))
    return hx, hy, conf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v` — Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: siamese correlation model with soft-argmax peak"
```

---

### Task 4: Training loop, metrics, and the actual training run

**Files:**
- Create: `drift_sense/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `evaluate(model, loader, device) -> dict` with keys `mean_err`, `pct5`, `pct10` (pixel error in original 1000-px search space, computed with `heat_to_search_px` + `soft_argmax_peak`).
- Produces: checkpoint file `checkpoints/best.pt` = `model.state_dict()` of best val `mean_err`.
- CLI: `uv run python -m drift_sense.train --epochs 30 --batch 8`.

- [ ] **Step 1: Write the failing test (fast smoke test, 4 pairs, 1 epoch)**

```python
# tests/test_train.py
import torch
from torch.utils.data import DataLoader
from drift_sense.dataset import DriftPairDataset, load_manifest
from drift_sense.model import SiameseLocator
from drift_sense.train import evaluate, train_one_epoch

def test_train_smoke():
    rows = load_manifest("dataset_drift_sense/manifest.csv")[:4]
    loader = DataLoader(DriftPairDataset(rows), batch_size=2)
    model = SiameseLocator()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, opt, torch.device("cpu"))
    assert loss > 0
    m = evaluate(model, loader, torch.device("cpu"))
    assert set(m) == {"mean_err", "pct5", "pct10"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# drift_sense/train.py
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from drift_sense.dataset import (
    DriftPairDataset, load_manifest, split_pairs, heat_to_search_px,
)
from drift_sense.model import SiameseLocator, soft_argmax_peak

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for ref, search, target, _ in loader:
        ref, search, target = ref.to(device), search.to(device), target.to(device)
        logits = model(ref, search)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss) * ref.shape[0]
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for ref, search, _, gt in loader:
        logits = model(ref.to(device), search.to(device))
        for i in range(ref.shape[0]):
            hx, hy, _ = soft_argmax_peak(logits[i])
            px, py = heat_to_search_px(hx), heat_to_search_px(hy)
            errs.append(((px - float(gt[i, 0])) ** 2 + (py - float(gt[i, 1])) ** 2) ** 0.5)
    errs = torch.tensor(errs)
    return {
        "mean_err": float(errs.mean()),
        "pct5": float((errs <= 5).float().mean()),
        "pct10": float((errs <= 10).float().mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_rows, val_rows, _ = split_pairs(load_manifest("dataset_drift_sense/manifest.csv"))
    train_loader = DataLoader(DriftPairDataset(train_rows), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(DriftPairDataset(val_rows), batch_size=args.batch)

    model = SiameseLocator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs("checkpoints", exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device)
        metrics = evaluate(model, val_loader, device)
        sched.step()
        print(f"epoch {epoch:02d} loss {loss:.4f} val {metrics}")
        if metrics["mean_err"] < best:
            best = metrics["mean_err"]
            torch.save(model.state_dict(), "checkpoints/best.pt")
    print(f"best val mean_err: {best:.2f} px")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test**

Run: `uv run pytest tests/test_train.py -v` — Expected: PASS (may take ~1 min on CPU)

- [ ] **Step 5: Run the real training**

Run: `uv run python -m drift_sense.train --epochs 30 --batch 8`
Expected: finishes (~10–25 min on MPS); best val `mean_err` **< 10 px**. If val mean_err > 10 px after 30 epochs, re-run with `--epochs 60 --lr 5e-4` once before investigating.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: training loop; trained checkpoint metrics in message"
```

---

### Task 5: Prediction CLI with tie-break, confidence, English output

**Files:**
- Create: `drift_sense/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `SiameseLocator`, `soft_argmax_peak`, dataset constants, `checkpoints/best.pt`.
- Produces: `predict(ref_path: str, search_path: str, ckpt: str = "checkpoints/best.pt") -> dict` with keys `x`, `y` (floats, search px), `confidence` (0–1), `found` (bool, confidence ≥ 0.5), `message` (English sentence).
- Tie-break rule: find all local maxima (3×3 neighborhood) with logit ≥ (max logit − 1.0); among them choose the peak nearest heatmap centre (62, 62); refine with `soft_argmax_peak` on a window around it.
- CLI: `uv run python -m drift_sense.predict --ref R --search S` prints `message` and the dict as JSON.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predict.py
from drift_sense.dataset import load_manifest
from drift_sense.predict import predict

def test_predict_on_test_pair():
    row = load_manifest("dataset_drift_sense/manifest.csv")[499]  # test split pair
    out = predict(row["ref_path"], row["search_path"])
    assert out["found"] is True
    err = ((out["x"] - row["gt_x"]) ** 2 + (out["y"] - row["gt_y"]) ** 2) ** 0.5
    assert err <= 15.0
    assert "found at (" in out["message"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_predict.py -v` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# drift_sense/predict.py
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from drift_sense.dataset import REF_SIZE, SEARCH_SIZE, HEAT, heat_to_search_px
from drift_sense.model import SiameseLocator, soft_argmax_peak

CONF_THRESHOLD = 0.5
PEAK_MARGIN = 1.0

def _load(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

def _center_tiebreak(logits):
    pooled = F.max_pool2d(logits[None, None], 3, stride=1, padding=1)[0, 0]
    is_peak = (logits == pooled) & (logits >= logits.max() - PEAK_MARGIN)
    ys, xs = torch.nonzero(is_peak, as_tuple=True)
    c = (HEAT - 1) / 2
    d2 = (xs.float() - c) ** 2 + (ys.float() - c) ** 2
    j = int(d2.argmin())
    py, px = int(ys[j]), int(xs[j])
    win = torch.full_like(logits, -1e9)
    y0, y1 = max(0, py - 2), min(HEAT, py + 3)
    x0, x1 = max(0, px - 2), min(HEAT, px + 3)
    win[y0:y1, x0:x1] = logits[y0:y1, x0:x1]
    return soft_argmax_peak(win)

def predict(ref_path, search_path, ckpt="checkpoints/best.pt"):
    model = SiameseLocator()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(_load(ref_path, REF_SIZE), _load(search_path, SEARCH_SIZE))[0]
    hx, hy, conf = _center_tiebreak(logits)
    x, y = heat_to_search_px(hx), heat_to_search_px(hy)
    found = conf >= CONF_THRESHOLD
    if found:
        msg = f"Pattern found at ({x:.0f}, {y:.0f}) in the search image. Confidence: {conf:.0%}."
    else:
        msg = f"No pattern found (best score {conf:.0%} below threshold)."
    return {"x": x, "y": y, "confidence": conf, "found": found, "message": msg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    args = ap.parse_args()
    out = predict(args.ref, args.search, args.ckpt)
    print(out["message"])
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_predict.py -v` — Expected: PASS

- [ ] **Step 5: Report test-split metrics for the CNN**

```bash
uv run python - <<'EOF'
import torch
from torch.utils.data import DataLoader
from drift_sense.dataset import DriftPairDataset, load_manifest, split_pairs
from drift_sense.model import SiameseLocator
from drift_sense.train import evaluate
_, _, test_rows = split_pairs(load_manifest("dataset_drift_sense/manifest.csv"))
model = SiameseLocator()
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu", weights_only=True))
print(evaluate(model, DataLoader(DriftPairDataset(test_rows), batch_size=8), torch.device("cpu")))
EOF
```

Expected: `mean_err` < 10 px. Record numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: predict CLI with centre tie-break and confidence"
```

---

### Task 6: VLM chat dataset builder

**Files:**
- Create: `vlm/__init__.py` (empty)
- Create: `vlm/make_vlm_dataset.py`
- Test: `tests/test_vlm_dataset.py`

**Interfaces:**
- Consumes: `load_manifest`, `split_pairs`.
- Produces: `PROMPT` constant (exact string below), `build_split(rows, out_dir, split_name) -> str` writing `vlm/data/<split_name>.jsonl` and resized images to `vlm/data/images/`.
- JSONL line format (mlx-vlm chat format): `{"images": ["vlm/data/images/ref_001.png", "vlm/data/images/search_001.png"], "messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": "Pattern found at (242, 241). Confidence: high."}]}`
- Image prep: reference resized to 224×224, search resized to 448×448, saved as PNG under `vlm/data/images/` keeping original file names. Coordinates in the answer stay in ORIGINAL search-image pixels (0–1000) rounded to int.
- `PROMPT = "The first image is a zoomed-in reference pattern. The second image is a zoomed-out search image (1000x1000 pixels). Find the reference pattern in the search image. If several matches exist, pick the one closest to the centre. Answer exactly: Pattern found at (x, y). Confidence: high."`
- CLI: `uv run python -m vlm.make_vlm_dataset` builds train/valid/test JSONL.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vlm_dataset.py
import json
import cv2
from drift_sense.dataset import load_manifest
from vlm.make_vlm_dataset import build_split

def test_build_split(tmp_path):
    rows = load_manifest("dataset_drift_sense/manifest.csv")[:2]
    out = build_split(rows, str(tmp_path), "train")
    lines = [json.loads(l) for l in open(out)]
    assert len(lines) == 2
    first = lines[0]
    assert len(first["images"]) == 2
    assert first["messages"][1]["content"] == "Pattern found at (242, 241). Confidence: high."
    ref = cv2.imread(first["images"][0])
    search = cv2.imread(first["images"][1])
    assert ref.shape[:2] == (224, 224)
    assert search.shape[:2] == (448, 448)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vlm_dataset.py -v` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# vlm/make_vlm_dataset.py
import json
import os
import cv2
from drift_sense.dataset import load_manifest, split_pairs

PROMPT = (
    "The first image is a zoomed-in reference pattern. The second image is a "
    "zoomed-out search image (1000x1000 pixels). Find the reference pattern in "
    "the search image. If several matches exist, pick the one closest to the "
    "centre. Answer exactly: Pattern found at (x, y). Confidence: high."
)

def _resize_save(src, dst, size):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(dst, cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

def build_split(rows, out_dir, split_name):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            ref_dst = os.path.join(img_dir, os.path.basename(r["ref_path"]))
            search_dst = os.path.join(img_dir, os.path.basename(r["search_path"]))
            _resize_save(r["ref_path"], ref_dst, 224)
            _resize_save(r["search_path"], search_dst, 448)
            answer = (
                f"Pattern found at ({round(r['gt_x'])}, {round(r['gt_y'])}). "
                "Confidence: high."
            )
            f.write(json.dumps({
                "images": [ref_dst, search_dst],
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": answer},
                ],
            }) + "\n")
    return out_path

def main():
    train, val, test = split_pairs(load_manifest("dataset_drift_sense/manifest.csv"))
    for name, rows in [("train", train), ("valid", val), ("test", test)]:
        print(build_split(rows, "vlm/data", name))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then build the full dataset**

Run: `uv run pytest tests/test_vlm_dataset.py -v` — Expected: PASS
Run: `uv run python -m vlm.make_vlm_dataset` — Expected: three JSONL paths printed, `vlm/data/images/` has 1000 PNGs.

- [ ] **Step 5: Commit** (data folder is small resized PNGs; add `vlm/data/` to `.gitignore` and commit code only)

```bash
printf 'vlm/data/\n' >> .gitignore
git add -A && git commit -m "feat: VLM chat dataset builder"
```

---

### Task 7: LoRA fine-tune Qwen3.5-4B with mlx-vlm

**Files:**
- Create: `vlm/finetune.py` (thin wrapper that shells out to mlx-vlm LoRA)
- Output: `vlm/adapters/` (LoRA weights, gitignored)

**Interfaces:**
- Consumes: `vlm/data/train.jsonl`, `vlm/data/valid.jsonl` (Task 6).
- Produces: adapter weights in `vlm/adapters/` loadable by `mlx_vlm.generate --adapter-path vlm/adapters`.
- Base model: `mlx-community/Qwen3.5-4B-MLX-4bit`.

- [ ] **Step 1: Install mlx-vlm and check the LoRA CLI**

```bash
uv add mlx-vlm
uv run python -m mlx_vlm.lora --help
```

Expected: help text listing flags for model, dataset, epochs/iters, batch size, adapter output. The flags below are the intended call; if names differ in the installed version, adapt to the help text (same semantics) and record the final command in `vlm/finetune.py`.

- [ ] **Step 2: Write the wrapper**

```python
# vlm/finetune.py
"""LoRA fine-tune of Qwen3.5-4B on the drift-sense chat dataset."""
import subprocess
import sys

CMD = [
    sys.executable, "-m", "mlx_vlm.lora",
    "--model", "mlx-community/Qwen3.5-4B-MLX-4bit",
    "--dataset", "vlm/data",
    "--epochs", "2",
    "--batch-size", "1",
    "--lora-rank", "8",
    "--learning-rate", "1e-4",
    "--adapter-path", "vlm/adapters",
]

if __name__ == "__main__":
    raise SystemExit(subprocess.call(CMD))
```

- [ ] **Step 3: Run the fine-tune**

Run: `uv run python -m vlm.finetune`
Expected: downloads ~3 GB model on first run, then trains. Loss printed and decreasing. Budget 1–3 hours on M4 Pro. Adapters written to `vlm/adapters/`.

- [ ] **Step 4: Sanity-check one generation**

```bash
uv run python -m mlx_vlm.generate \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --adapter-path vlm/adapters \
  --image vlm/data/images/ref_500.png vlm/data/images/search_500.png \
  --prompt "The first image is a zoomed-in reference pattern. The second image is a zoomed-out search image (1000x1000 pixels). Find the reference pattern in the search image. If several matches exist, pick the one closest to the centre. Answer exactly: Pattern found at (x, y). Confidence: high." \
  --max-tokens 40
```

Expected: output contains `Pattern found at (` with two integers.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: LoRA fine-tune wrapper for Qwen3.5-4B"
```

---

### Task 8: VLM evaluation

**Files:**
- Create: `vlm/eval.py`
- Test: `tests/test_vlm_eval.py` (parser only — no model load in tests)

**Interfaces:**
- Consumes: `vlm/data/test.jsonl`, adapters from Task 7, `PROMPT` from `vlm.make_vlm_dataset`.
- Produces: `parse_coords(text: str) -> tuple[float, float] | None` (regex `\((\d+)\s*,\s*(\d+)\)`).
- Produces: `evaluate_vlm(jsonl_path: str, limit: int | None = None) -> dict` with `mean_err`, `pct5`, `pct10`, `parse_fail_rate` — generation via `mlx_vlm` Python API (`load`, `generate`), coordinates compared against manifest GT for the same pair_id (join on ref filename).
- CLI: `uv run python -m vlm.eval` prints the dict.

- [ ] **Step 1: Write the failing parser tests**

```python
# tests/test_vlm_eval.py
from vlm.eval import parse_coords

def test_parse_ok():
    assert parse_coords("Pattern found at (242, 241). Confidence: high.") == (242.0, 241.0)

def test_parse_fail():
    assert parse_coords("I cannot find the pattern.") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vlm_eval.py -v` — Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**

```python
# vlm/eval.py
import json
import os
import re
import argparse
from drift_sense.dataset import load_manifest

def parse_coords(text):
    m = re.search(r"\((\d+)\s*,\s*(\d+)\)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def evaluate_vlm(jsonl_path="vlm/data/test.jsonl", limit=None,
                 model_path="mlx-community/Qwen3.5-4B-MLX-4bit",
                 adapter_path="vlm/adapters"):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    gt_by_ref = {os.path.basename(r["ref_path"]): (r["gt_x"], r["gt_y"])
                 for r in load_manifest("dataset_drift_sense/manifest.csv")}
    model, processor = load(model_path, adapter_path=adapter_path)
    config = model.config

    samples = [json.loads(l) for l in open(jsonl_path)]
    if limit:
        samples = samples[:limit]

    errs, fails = [], 0
    for s in samples:
        prompt = apply_chat_template(processor, config, s["messages"][0]["content"], num_images=2)
        out = generate(model, processor, prompt, s["images"], max_tokens=40, verbose=False)
        text = out.text if hasattr(out, "text") else str(out)
        coords = parse_coords(text)
        gt = gt_by_ref[os.path.basename(s["images"][0])]
        if coords is None:
            fails += 1
            continue
        errs.append(((coords[0] - gt[0]) ** 2 + (coords[1] - gt[1]) ** 2) ** 0.5)

    n = len(samples)
    return {
        "mean_err": sum(errs) / len(errs) if errs else float("inf"),
        "pct5": sum(e <= 5 for e in errs) / n,
        "pct10": sum(e <= 10 for e in errs) / n,
        "parse_fail_rate": fails / n,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(evaluate_vlm(limit=args.limit))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run parser tests, then a 5-sample live eval**

Run: `uv run pytest tests/test_vlm_eval.py -v` — Expected: 2 PASS
Run: `uv run python -m vlm.eval --limit 5` — Expected: dict printed, `parse_fail_rate` ≤ 0.2. (If the mlx_vlm API signatures differ from above, adjust to the installed version — semantics stay the same.)

- [ ] **Step 5: Full eval and commit**

Run: `uv run python -m vlm.eval` — record the dict.

```bash
git add -A && git commit -m "feat: VLM eval with coordinate parsing; metrics in message"
```

---

### Task 9: Comparison report

**Files:**
- Create: `compare.py`

**Interfaces:**
- Consumes: `evaluate` + checkpoint (Task 4/5), `evaluate_vlm` (Task 8).
- Produces: prints a markdown table comparing CNN vs VLM on the test split: `mean_err`, `pct5`, `pct10`, seconds/pair.

- [ ] **Step 1: Implement**

```python
# compare.py
import time
import torch
from torch.utils.data import DataLoader
from drift_sense.dataset import DriftPairDataset, load_manifest, split_pairs
from drift_sense.model import SiameseLocator
from drift_sense.train import evaluate
from vlm.eval import evaluate_vlm

def main():
    _, _, test_rows = split_pairs(load_manifest("dataset_drift_sense/manifest.csv"))

    model = SiameseLocator()
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu", weights_only=True))
    t0 = time.time()
    cnn = evaluate(model, DataLoader(DriftPairDataset(test_rows), batch_size=8),
                   torch.device("cpu"))
    cnn_spp = (time.time() - t0) / len(test_rows)

    t0 = time.time()
    vlm = evaluate_vlm()
    vlm_spp = (time.time() - t0) / len(test_rows)

    print("| model | mean_err (px) | <=5px | <=10px | s/pair |")
    print("|---|---|---|---|---|")
    print(f"| Siamese CNN | {cnn['mean_err']:.2f} | {cnn['pct5']:.0%} | {cnn['pct10']:.0%} | {cnn_spp:.2f} |")
    print(f"| Qwen3.5-4B LoRA | {vlm['mean_err']:.2f} | {vlm['pct5']:.0%} | {vlm['pct10']:.0%} | {vlm_spp:.2f} |")
    print(f"\nVLM parse failures: {vlm['parse_fail_rate']:.0%}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python compare.py` — Expected: table prints with both rows filled.

- [ ] **Step 3: Run the full test suite one last time**

Run: `uv run pytest -v` — Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: CNN vs VLM comparison report"
```
