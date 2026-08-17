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
