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
