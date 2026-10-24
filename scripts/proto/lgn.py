"""Logic-gate network with FIXED pseudo-random wiring and learned 2-input gate functions.
Training: each gate = softmax mixture of the 16 Boolean functions, written as c0 + c1 a + c2 b + c3 ab.
Inference: each gate = its 4-bit truth table (one LUT2 = one mux4 with 4 configuration bits).
Output: last layer split in 10 groups, class score = popcount of the group.
The wiring is a function of a seed only (it would be etched in the chip); only the truth tables
and the input thresholds are task-specific.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time
import numpy as np
import torch

C16 = torch.tensor([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                    [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]],
                   dtype=torch.float32)

def wiring(seed, n_in, widths):
    """fixed wiring: each gate of layer l takes 2 distinct outputs of layer l-1 (balanced fan-out)."""
    g = np.random.default_rng(seed); W = []; n = n_in
    for w in widths:
        need = 2 * w; src = np.concatenate([g.permutation(n) for _ in range((need + n - 1) // n)])[:need]
        a, b = src[0::2].copy(), src[1::2].copy()
        same = a == b
        b[same] = (b[same] + 1 + g.integers(0, n - 1, same.sum())) % n
        W.append((torch.as_tensor(a), torch.as_tensor(b))); n = w
    return W

def local_wiring(seed, in_shape, layers, r):
    """columnar local wiring. in_shape = (H, W, C) of the input bits; layers = list of (H, W, C).
    A gate at (y, x) of a layer reads 2 random outputs of the previous layer within a window of
    radius r around the matching position (any channel). Returns index pairs and wire lengths
    (in units of the previous layer's cell pitch)."""
    g = np.random.default_rng(seed); W = []; L = []; ph, pw, pc = in_shape
    for (h, w, c) in layers:
        ys, xs, cs = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing='ij')
        ys, xs = ys.ravel(), xs.ravel(); n = h * w * c
        cy = np.rint((ys + 0.5) * ph / h - 0.5).astype(int); cx = np.rint((xs + 0.5) * pw / w - 0.5).astype(int)
        pair = []
        for _ in range(2):
            dy = g.integers(-r, r + 1, n); dx = g.integers(-r, r + 1, n)
            yy = np.clip(cy + dy, 0, ph - 1); xx = np.clip(cx + dx, 0, pw - 1); cc = g.integers(0, pc, n)
            pair.append((yy * pw + xx) * pc + cc)
            L.append(np.hypot(yy - cy, xx - cx))
        a, b = pair; same = a == b
        b[same] = (b[same] + 1) % (ph * pw * pc)
        W.append((torch.as_tensor(a), torch.as_tensor(b))); ph, pw, pc = h, w, c
    return W, np.concatenate(L)

class LGN(torch.nn.Module):
    def __init__(self, n_in, widths, seed, res_init=True, wire=None):
        super().__init__()
        self.wire = wire if wire is not None else wiring(seed, n_in, widths)
        th = []
        for w in widths:
            t = torch.randn(w, 16) * 0.1
            if res_init: t[:, 3] = 5.0          # start close to the pass-through gate "a" (deep nets train better)
            th.append(torch.nn.Parameter(t))
        self.theta = torch.nn.ParameterList(th)
    def forward(self, x, hard=False):
        for (ia, ib), t in zip(self.wire, self.theta):
            a, b = x[:, ia], x[:, ib]
            if hard:
                k = C16[t.argmax(1)]
            else:
                k = torch.softmax(t, 1) @ C16
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b
        return x
    def tables(self):
        """truth tables t[(a,b)] for (0,0),(0,1),(1,0),(1,1) -> 4 configuration bits per gate."""
        out = []
        for t in self.theta:
            k = C16[t.argmax(1)]
            out.append(torch.stack([k[:, 0], k[:, 0] + k[:, 2], k[:, 0] + k[:, 1], k.sum(1)], 1).round().to(torch.int8))
        return out

def encode(X, thr):
    """thermometer code: one bit per (pixel, threshold)."""
    X = torch.as_tensor(X, dtype=torch.float32) / 255.0
    return torch.cat([(X > t).float() for t in thr], 1)

def group_scores(y, k=10):
    # interleaved groups: gate i belongs to class i mod k (each class pools the whole layer)
    return y.reshape(y.shape[0], -1, k).sum(1)

def run(cfg):
    sys.path.insert(0, HERE + '')
    from data import load
    torch.manual_seed(cfg.get('seed', 0)); torch.set_num_threads(cfg.get('threads', 4))
    Xtr, ytr, Xte, yte = load(cfg['data'])
    thr = cfg.get('thr', [0.5])
    Etr, Ete = encode(Xtr, thr), encode(Xte, thr)
    ytr_t, yte_t = torch.as_tensor(ytr), torch.as_tensor(yte)
    wl = None
    if cfg.get('local'):
        # input bits laid out as (28, 28, len(thr)): pixel-major, thresholds as channels
        side = int(round((Etr.shape[1] // len(thr)) ** 0.5))
        Etr = Etr.reshape(-1, len(thr), side * side).transpose(1, 2).reshape(len(Etr), -1)
        Ete = Ete.reshape(-1, len(thr), side * side).transpose(1, 2).reshape(len(Ete), -1)
        layers = [tuple(l) for l in cfg['local']]
        wire, wl = local_wiring(cfg.get('wseed', 1), (side, side, len(thr)), layers, cfg.get('r', 2))
        widths = [h * w * c for h, w, c in layers]
        net = LGN(Etr.shape[1], widths, 0, cfg.get('res', True), wire=wire)
    else:
        widths = [cfg['width']] * cfg['depth']
        net = LGN(Etr.shape[1], widths, cfg.get('wseed', 1), cfg.get('res', True))
    tau = cfg.get('tau', (widths[-1] / 10) ** 0.5)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 0.01))
    bs = cfg.get('bs', 128); t0 = time.time()
    for ep in range(cfg['epochs']):
        perm = torch.randperm(len(ytr_t))
        for k in range(0, len(perm), bs):
            bi = perm[k:k + bs]
            s = group_scores(net(Etr[bi])) / tau
            loss = torch.nn.functional.cross_entropy(s, ytr_t[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_soft = (group_scores(net(Ete)).argmax(1) == yte_t).float().mean().item()
        yh = net(Ete, hard=True)
        acc_hard = (group_scores(yh).argmax(1) == yte_t).float().mean().item()
        # activity: probability that a gate output toggles between consecutive test images
        acts = []; x = Ete
        for (ia, ib), t in zip(net.wire, net.theta):
            k = C16[t.argmax(1)]; a, b = x[:, ia], x[:, ib]
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b
            acts.append((x[1:] != x[:-1]).float().mean(0))
        act = torch.cat(acts)
        tabs = torch.cat([t.argmax(1) for t in net.theta])
        hist = torch.bincount(tabs, minlength=16).tolist()
    gates = sum(widths)
    return dict(acc_soft=acc_soft, acc=acc_hard, gates=gates, wire_mean=None if wl is None else float(wl.mean()), in_bits=Etr.shape[1], toggles=float(act.sum()),
                toggle_rate=float(act.mean()), op_hist=hist, cfg=cfg, secs=round(time.time() - t0))

if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))), flush=True)
