import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from drift_sense.dataset import (
    DriftPairDataset, load_manifest, split_pairs, heat_to_search_px,
)
from drift_sense.model import SiameseLocator, soft_argmax_peak

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for ref, search, target, _ in loader:
        ref, search, target = ref.to(device), search.to(device), target.to(device)
        logits = model(ref, search)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss) * ref.shape[0]
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for ref, search, _, gt in loader:
        logits = model(ref.to(device), search.to(device))
        for i in range(ref.shape[0]):
            hx, hy, _ = soft_argmax_peak(logits[i])
            px, py = heat_to_search_px(hx), heat_to_search_px(hy)
            errs.append(((px - float(gt[i, 0])) ** 2 + (py - float(gt[i, 1])) ** 2) ** 0.5)
    errs = torch.tensor(errs)
    return {
        "mean_err": float(errs.mean()),
        "pct5": float((errs <= 5).float().mean()),
        "pct10": float((errs <= 10).float().mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_rows, val_rows, _ = split_pairs(load_manifest("dataset_drift_sense/manifest.csv"))
    train_loader = DataLoader(DriftPairDataset(train_rows), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(DriftPairDataset(val_rows), batch_size=args.batch)

    model = SiameseLocator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs("checkpoints", exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device)
        metrics = evaluate(model, val_loader, device)
        sched.step()
        print(f"epoch {epoch:02d} loss {loss:.4f} val {metrics}", flush=True)
        if metrics["mean_err"] < best:
            best = metrics["mean_err"]
            torch.save(model.state_dict(), "checkpoints/best.pt")
    print(f"best val mean_err: {best:.2f} px")

if __name__ == "__main__":
    main()
