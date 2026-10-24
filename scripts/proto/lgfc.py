"""Logic networks with FULLY learned connectivity (each gate pin = softmax over all outputs of the
previous layer, computed as a matmul), gate types:
  lut2 : 2 pins, learned 4-entry truth table (bilinear relaxation)
  maj3 : 3 pins, each with a learned inversion, soft majority ab+bc+ca-2abc (one sky130 maj3 cell
         + optional inverters in hardware)
Output: group-sum (gate i -> class i mod 10), class score = popcount.
Hard evaluation: argmax connection per pin, thresholded truth table / inversion.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, HERE + '')
from data import load

def grid_pos(n):
    side = math.ceil(math.sqrt(n)); i = torch.arange(n)
    return torch.stack([(i // side + 0.5) / side, (i % side + 0.5) / side], 1)

class Layer(nn.Module):
    def __init__(self, n_prev, n, kind):
        super().__init__()
        self.kind = kind; self.p = 2 if kind == 'lut2' else 3; self.n = n; self.register_buffer('D', None)
        self.c = nn.Parameter(torch.randn(self.p * n, n_prev) * 0.01)      # connection logits
        if kind == 'lut2': self.t = nn.Parameter(torch.randn(n, 4) * 0.5)
        else: self.s = nn.Parameter(torch.randn(n, 3) * 0.5)               # inversion logits
    def set_positions(self, pos_prev, pos):
        d = torch.cdist(pos.repeat_interleave(self.p, 0), pos_prev, p=1)      # Manhattan wire length
        self.D = d
    def wire(self, tau, hard=False):
        if self.D is None: return torch.tensor(0.0)
        if hard: return self.D.gather(1, self.c.argmax(1, keepdim=True)).sum()
        return (torch.softmax(self.c / tau, 1) * self.D).sum()
    def pins(self, x, tau, hard):
        if hard:
            return x[:, self.c.argmax(1)].reshape(x.shape[0], self.n, self.p)
        return (x @ torch.softmax(self.c / tau, 1).T).reshape(x.shape[0], self.n, self.p)
    def forward(self, x, tau=1.0, hard=False):
        P = self.pins(x, tau, hard)
        if self.kind == 'lut2':
            a, b = P[..., 0], P[..., 1]
            T = (self.t > 0).float() if hard else torch.sigmoid(self.t)
            return T[:, 0] * (1 - a) * (1 - b) + T[:, 1] * (1 - a) * b + T[:, 2] * a * (1 - b) + T[:, 3] * a * b
        S = (self.s > 0).float() if hard else torch.sigmoid(self.s)        # S = 1: pass, 0: invert
        q = S * P + (1 - S) * (1 - P); a, b, c = q[..., 0], q[..., 1], q[..., 2]
        return a * b + b * c + c * a - 2 * a * b * c

class Cmp(nn.Module):
    """learned analog comparators on raw inputs: bit = [x_i - x_j > theta]; x_j may be the constant 0
    (then it is a learned threshold on one input). Pins choose their input by softmax (matmul)."""
    def __init__(self, n_in, n):
        super().__init__()
        self.n = n; self.c = nn.Parameter(torch.randn(2 * n, n_in + 1) * 0.01)
        self.th = nn.Parameter(torch.rand(n) * 0.5); self.T = 0.1
    def forward(self, x, tau=1.0, hard=False):
        xz = torch.cat([x, torch.zeros(x.shape[0], 1)], 1)
        if hard: P = xz[:, self.c.argmax(1)]
        else: P = xz @ torch.softmax(self.c / tau, 1).T
        P = P.reshape(x.shape[0], self.n, 2); d = P[..., 0] - P[..., 1] - self.th
        return (d > 0).float() if hard else torch.sigmoid(d / (self.T * max(tau, 0.05)))

def interval_forward(net, lo, hi, tau):
    for L in net:
        W = torch.softmax(L.c / tau, 1)
        a_lo, a_hi = (lo @ W.T).reshape(lo.shape[0], L.n, 2), (hi @ W.T).reshape(hi.shape[0], L.n, 2)
        T = torch.sigmoid(L.t)
        f = lambda a, b: T[:, 0] * (1 - a) * (1 - b) + T[:, 1] * (1 - a) * b + T[:, 2] * a * (1 - b) + T[:, 3] * a * b
        c = torch.stack([f(a_lo[..., 0], a_lo[..., 1]), f(a_lo[..., 0], a_hi[..., 1]), f(a_hi[..., 0], a_lo[..., 1]), f(a_hi[..., 0], a_hi[..., 1])], 0)
        lo, hi = c.min(0).values, c.max(0).values
    return lo, hi

def group(y, k=10):
    n = y.shape[1] - y.shape[1] % k
    return y[:, :n].reshape(y.shape[0], -1, k).sum(1)

def run(cfg):
    torch.manual_seed(cfg.get('seed', 0)); torch.set_num_threads(cfg.get('threads', 1))
    Xtr, ytr, Xte, yte = load(cfg['data'])
    thr = cfg.get('thr', [0.25, 0.5, 0.75])
    enc = lambda X: torch.as_tensor(np.stack([(X / 255.0 > t) for t in thr], 2).reshape(len(X), -1).astype(np.float32))
    Etr, Ete = enc(Xtr), enc(Xte); ytr_t, yte_t = torch.as_tensor(ytr), torch.as_tensor(yte)
    kinds = cfg.get('kinds', [cfg.get('kind', 'lut2')] * len(cfg['widths']))
    layers = []; n_prev = Etr.shape[1]
    if cfg.get('cmp'):                       # comparator front-end on raw intensities instead of thermometer bits
        Etr = torch.as_tensor((Xtr / 255.0).astype(np.float32)); Ete = torch.as_tensor((Xte / 255.0).astype(np.float32))
        layers.append(Cmp(Etr.shape[1], cfg['cmp'])); n_prev = cfg['cmp']
    for w, k in zip(cfg['widths'], kinds): layers.append(Layer(n_prev, w, k)); n_prev = w
    net = nn.ModuleList(layers)
    lam = cfg.get('wl', None)
    if lam is not None:                       # placement-aware connectivity: inputs at pixel positions, gates on a grid
        side = int(round((Etr.shape[1] // len(thr)) ** 0.5)); pix = torch.arange(side * side)
        pp = torch.stack([(pix // side + 0.5) / side, (pix % side + 0.5) / side], 1).repeat_interleave(len(thr), 0)
        for L in net:
            if isinstance(L, Layer): q = grid_pos(L.n); L.set_positions(pp, q); pp = q
    tau_out = cfg.get('tau', math.sqrt(n_prev / 10))
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 0.05))
    E = cfg['epochs']; bs = cfg.get('bs', 256); steps = E * ((len(ytr) + bs - 1) // bs); it = 0; t0 = time.time()
    for ep in range(E):
        perm = torch.randperm(len(ytr))
        for k0 in range(0, len(perm), bs):
            tc = max(cfg.get('tc_min', 0.03), 1.0 - it / steps); it += 1
            bi = perm[k0:k0 + bs]; x = Etr[bi]
            for L in net: x = L(x, tc)
            loss = F.cross_entropy(group(x) / tau_out, ytr_t[bi])
            if cfg.get('cert'):
                # phase-1 knowledge: only the lowest threshold is known; higher bits are 0 if it is 0, else unknown
                E3 = Etr[bi].reshape(len(bi), -1, len(thr)); k0 = E3[:, :, :1]
                lo_in = torch.cat([k0, torch.zeros_like(E3[:, :, 1:])], 2).reshape(len(bi), -1)
                hi_in = torch.cat([k0, k0.expand(-1, -1, len(thr) - 1)], 2).reshape(len(bi), -1)
                glo, ghi = interval_forward(net, lo_in, hi_in, tc)
                glo, ghi = group(glo), group(ghi)
                yt = ytr_t[bi]; own = glo[torch.arange(len(bi)), yt]
                oth = ghi.clone(); oth[torch.arange(len(bi)), yt] = -1e9
                margin = own - oth.max(1).values
                loss = loss + cfg['cert'] * F.softplus(-margin / tau_out).mean()
            if lam: loss = loss + lam * sum(L.wire(tc) for L in net if isinstance(L, Layer)) / sum(L.D.shape[0] for L in net if isinstance(L, Layer))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        def ev(hard):
            out = []
            for k0 in range(0, len(yte), 2000):
                x = Ete[k0:k0 + 2000]
                for L in net: x = L(x, cfg.get('tc_min', 0.03), hard)
                out.append(x)
            return torch.cat(out)
        ys, yh = ev(False), ev(True)
        acc_s = (group(ys).argmax(1) == yte_t).float().mean().item(); acc_h = (group(yh).argmax(1) == yte_t).float().mean().item()
    if cfg.get('save'):
        torch.save({'net': net.state_dict(), 'cfg': cfg}, cfg['save'])
    wl_hard = None
    if lam is not None:
        with torch.no_grad(): wl_hard = float(sum(L.wire(0.03, True) for L in net if isinstance(L, Layer)) / sum(L.D.shape[0] for L in net if isinstance(L, Layer)))
    return dict(acc=acc_h, acc_soft=acc_s, gates=sum(cfg['widths']), mean_wire=wl_hard, comparators=cfg.get('cmp', Etr.shape[1]), cfg=cfg, secs=round(time.time() - t0))

if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))), flush=True)
