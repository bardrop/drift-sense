from drift_sense.dataset import load_manifest
from drift_sense.predict import predict

def test_predict_on_test_pair():
    row = load_manifest("dataset_drift_sense/manifest.csv")[499]  # test split pair
    out = predict(row["ref_path"], row["search_path"])
    assert out["found"] is True
    err = ((out["x"] - row["gt_x"]) ** 2 + (out["y"] - row["gt_y"]) ** 2) ** 0.5
    assert err <= 15.0
    assert "found at (" in out["message"]
