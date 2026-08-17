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
