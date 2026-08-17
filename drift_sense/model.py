import torch
import torch.nn as nn
import torch.nn.functional as F

from drift_sense.dataset import HEAT

def _block(cin, cout, stride=1):
    # GroupNorm, not BatchNorm: BN running stats diverge badly from batch
    # stats here (val err 52px vs 4px), GN behaves the same in train and eval.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )

class SiameseLocator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _block(1, 16), _block(16, 16, stride=2),
            _block(16, 32), _block(32, 32, stride=2),
            _block(32, 64), _block(64, 64),
        )  # total stride 4: ref 50->13, search 500->125
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))
        # The die is periodic: several crossings look identical. Correlation is
        # translation-invariant, so it cannot pick the right copy. This map
        # learns the drift prior: the true match sits near the nominal spot.
        self.pos_bias = nn.Parameter(torch.zeros(HEAT, HEAT))

    def forward(self, ref, search):
        fr = F.normalize(self.encoder(ref), dim=1)      # (B,64,13,13)
        fs = F.normalize(self.encoder(search), dim=1)   # (B,64,125,125)
        outs = []
        for i in range(ref.shape[0]):
            corr = F.conv2d(fs[i : i + 1], fr[i].unsqueeze(0), padding=6)
            outs.append(corr[0, 0] / fr[i].numel() ** 0.5)
        heat = torch.stack(outs)                        # (B,125,125)
        return heat * self.scale + self.bias + self.pos_bias

def soft_argmax_peak(logits):
    h, w = logits.shape
    flat = logits.argmax()
    py, px = int(flat // w), int(flat % w)
    y0, y1 = max(0, py - 2), min(h, py + 3)
    x0, x1 = max(0, px - 2), min(w, px + 3)
    win = logits[y0:y1, x0:x1]
    weights = torch.softmax(win.flatten(), dim=0).reshape(win.shape)
    ys = torch.arange(y0, y1, dtype=torch.float32, device=logits.device)
    xs = torch.arange(x0, x1, dtype=torch.float32, device=logits.device)
    hy = float((weights.sum(dim=1) * ys).sum())
    hx = float((weights.sum(dim=0) * xs).sum())
    # Confidence = probability mass near the peak under a full-map softmax.
    # Ambiguous matches spread mass across competing peaks and score low.
    probs = torch.softmax(logits.flatten(), dim=0).reshape(h, w)
    conf = float(probs[y0:y1, x0:x1].sum())
    return hx, hy, conf
