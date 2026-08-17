import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from drift_sense.dataset import REF_SIZE, SEARCH_SIZE, HEAT, heat_to_search_px
from drift_sense.model import SiameseLocator, soft_argmax_peak
from drift_sense.refine import refine_subpixel

CONF_THRESHOLD = 0.5
PEAK_MARGIN = 1.0
_MODEL_CACHE = {}

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
    hx, hy, _ = soft_argmax_peak(win)
    # confidence from the full map, not the masked one
    probs = torch.softmax(logits.flatten(), dim=0).reshape(logits.shape)
    conf = float(probs[y0:y1, x0:x1].sum())
    return hx, hy, conf

def predict(ref_path, search_path, ckpt="checkpoints/best.pt", refine=True):
    if ckpt not in _MODEL_CACHE:
        model = SiameseLocator()
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        _MODEL_CACHE[ckpt] = model
    model = _MODEL_CACHE[ckpt]
    with torch.no_grad():
        logits = model(_load(ref_path, REF_SIZE), _load(search_path, SEARCH_SIZE))[0]
    hx, hy, conf = _center_tiebreak(logits)
    x, y = heat_to_search_px(hx), heat_to_search_px(hy)
    if refine:
        ref_full = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        search_full = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # bring the reference to search scale (10x smaller), then phase-correlate
        ref_small = cv2.resize(ref_full, (100, 100), interpolation=cv2.INTER_AREA)
        x, y = refine_subpixel(ref_small, search_full, x, y)
    found = bool(conf >= CONF_THRESHOLD)
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
