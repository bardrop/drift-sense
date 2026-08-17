import torch
from drift_sense.model import SiameseLocator, soft_argmax_peak

def test_forward_shapes():
    m = SiameseLocator()
    logits = m(torch.rand(2, 1, 50, 50), torch.rand(2, 1, 500, 500))
    assert tuple(logits.shape) == (2, 125, 125)

def test_soft_argmax_finds_planted_peak():
    logits = torch.full((125, 125), -8.0)
    logits[40, 70] = 6.0
    hx, hy, conf = soft_argmax_peak(logits)
    assert abs(hx - 70) < 1.0 and abs(hy - 40) < 1.0
    assert conf > 0.9
