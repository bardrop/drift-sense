import argparse
import json
import os
import re
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
