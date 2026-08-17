import torch
from torch.utils.data import DataLoader
from drift_sense.dataset import DriftPairDataset, load_manifest
from drift_sense.model import SiameseLocator
from drift_sense.train import evaluate, train_one_epoch

def test_train_smoke():
    rows = load_manifest("dataset_drift_sense/manifest.csv")[:4]
    loader = DataLoader(DriftPairDataset(rows), batch_size=2)
    model = SiameseLocator()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, opt, torch.device("cpu"))
    assert loss > 0
    m = evaluate(model, loader, torch.device("cpu"))
    assert set(m) == {"mean_err", "pct5", "pct10"}
