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
