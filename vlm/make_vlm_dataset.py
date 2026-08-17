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
