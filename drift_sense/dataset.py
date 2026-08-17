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
