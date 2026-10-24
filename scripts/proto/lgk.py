"""Logic-gate network where each gate input pin chooses among K FIXED candidate wires (programmable
K:1 selection in hardware), and the gate function is a learned LUT2 (4 truth-table entries).
K = 1 is the fixed-wiring case. Candidates: 'random' (any output of the previous layer) or 'local'
(window of radius r on a 2D layout). Training: softmax over candidates (temperature annealed) and
sigmoid truth tables; evaluation: hard argmax candidate + thresholded truth table.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, HERE + '')
from data import load

def candidates(g, n_prev, n, K, mode, prev_shape=None, shape=None, r=2):
    """index tensor [n, 2, K] of candidate sources for each gate pin."""
    if mode == 'random' or prev_shape is None:
        return torch.as_tensor(g.integers(0, n_prev, (n, 2, K)))
    ph, pw, pc = prev_shape; h, w, c = shape
    y, x, _ = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing='ij'); y, x = y.ravel(), x.ravel()
    cy = np.rint((y + 0.5) * ph / h - 0.5).astype(int); cx = np.rint((x + 0.5) * pw / w - 0.5).astype(int)
    yy = np.clip(cy[:, None, None] + g.integers(-r, r + 1, (n, 2, K)), 0, ph - 1)
    xx = np.clip(cx[:, None, None] + g.integers(-r, r + 1, (n, 2, K)), 0, pw - 1)
    return torch.as_tensor((yy * pw + xx) * pc + g.integers(0, pc, (n, 2, K)))

class Layer(nn.Module):
    def __init__(self, idx, K):
        super().__init__()
        n = idx.shape[0]; self.register_buffer('idx', idx); self.K = K
        self.w = nn.Parameter(torch.randn(n, 2, K) * 0.01) if K > 1 else None
        self.t = nn.Parameter(torch.randn(n, 4) * 0.5)          # truth table logits: (0,0),(0,1),(1,0),(1,1)
    def forward(self, x, tau=1.0, hard=False):
        cand = x[:, self.idx]                                    # B, n, 2, K
        if self.K == 1: ab = cand[..., 0]
        elif hard: ab = torch.gather(cand, 3, self.w.argmax(2)[None, :, :, None].expand(x.shape[0], -1, -1, 1))[..., 0]
        else: ab = (cand * torch.softmax(self.w / tau, 2)).sum(3)
        a, b = ab[..., 0], ab[..., 1]
        T = (self.t > 0).float() if hard else torch.sigmoid(self.t)
        return (T[:, 0] * (1 - a) * (1 - b) + T[:, 1] * (1 - a) * b + T[:, 2] * a * (1 - b) + T[:, 3] * a * b)

def group(y, k=10):
    n = y.shape[1] - y.shape[1] % k
    return y[:, :n].reshape(y.shape[0], -1, k).sum(1)

def run(cfg):
    torch.manual_seed(cfg.get('seed', 0)); torch.set_num_threads(cfg.get('threads', 2))
    Xtr, ytr, Xte, yte = load(cfg['data'])
    thr = cfg.get('thr', [0.25, 0.5, 0.75]); side = int(round(Xtr.shape[1] ** 0.5))
    enc = lambda X: torch.as_tensor(np.stack([(X / 255.0 > t) for t in thr], 2).reshape(len(X), -1).astype(np.float32))
    Etr, Ete = enc(Xtr), enc(Xte); ytr_t, yte_t = torch.as_tensor(ytr), torch.as_tensor(yte)
    g = np.random.default_rng(cfg.get('wseed', 1)); K = cfg.get('K', 1); mode = cfg.get('cand', 'random')
    layers = []; n_prev = Etr.shape[1]; prev_shape = (side, side, len(thr))
    if 'local' in cfg:
        for sh in cfg['local']:
            n = sh[0] * sh[1] * sh[2]; layers.append(Layer(candidates(g, n_prev, n, K, mode, prev_shape, tuple(sh), cfg.get('r', 2)), K))
            n_prev, prev_shape = n, tuple(sh)
    else:
        for _ in range(cfg['depth']):
            n = cfg['width']; layers.append(Layer(candidates(g, n_prev, n, K, 'random'), K)); n_prev = n
    net = nn.ModuleList(layers)
    tau_out = cfg.get('tau', math.sqrt(n_prev / 10))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 0.05))
    E = cfg['epochs']; bs = cfg.get('bs', 256); steps = E * ((len(ytr) + bs - 1) // bs); it = 0; t0 = time.time()
    for ep in range(E):
        perm = torch.randperm(len(ytr))
        for k0 in range(0, len(perm), bs):
            tc = max(0.05, 1.0 - it / steps); it += 1           # candidate temperature annealing
            bi = perm[k0:k0 + bs]; x = Etr[bi]
            for L in net: x = L(x, tc)
            loss = F.cross_entropy(group(x) / tau_out, ytr_t[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        def ev(hard):
            out = []
            for k0 in range(0, len(yte), 2000):
                x = Ete[k0:k0 + 2000]
                for L in net: x = L(x, 0.05, hard)
                out.append(x)
            return torch.cat(out)
        ys, yh = ev(False), ev(True)
        acc_s = (group(ys).argmax(1) == yte_t).float().mean().item(); acc_h = (group(yh).argmax(1) == yte_t).float().mean().item()
        tog = (yh[1:] != yh[:-1]).float().mean(0).sum().item()
    gates = sum(L.idx.shape[0] for L in net)
    cfg_bits = gates * 4 + (gates * 2 * math.ceil(math.log2(K)) if K > 1 else 0)
    return dict(acc=acc_h, acc_soft=acc_s, gates=gates, cfg_bits=cfg_bits, last_layer_toggles=tog, cfg=cfg, secs=round(time.time() - t0))

if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))), flush=True)
