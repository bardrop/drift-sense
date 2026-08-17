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

import numpy as np
from drift_sense.dataset import (
    DriftPairDataset, make_target, search_px_to_heat, heat_to_search_px, HEAT,
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
    rows = load_manifest(MANIFEST)[:2]
    ds = DriftPairDataset(rows)
    ref, search, target, gt = ds[0]
    assert tuple(ref.shape) == (1, 50, 50)
    assert tuple(search.shape) == (1, 500, 500)
    assert tuple(target.shape) == (125, 125)
    assert abs(float(gt[0]) - rows[0]["gt_x"]) < 1e-3
