"""Certified early decision during a ramp ADC: thermometer bits arrive threshold by threshold
(low thresholds first). Unknown bits are X; 3-valued simulation of the HARD logic network gives
per-class [lo, hi] vote counts; the decision is certified when lo(best) > hi(others)
(ties are broken by lowest index, as argmax does, so >= when the best index is lower).
Reports the fraction of images certified after each threshold phase."""
import sys, json, torch, numpy as np
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from data import load
from lgf import Layer, group
ck = torch.load(sys.argv[1], weights_only=False); cfg = ck['cfg']
Xtr, ytr, Xte, yte = load(cfg['data']); thr = cfg.get('thr', [0.25, 0.5, 0.75])
layers = []; n_prev = len(thr) * Xte.shape[1]
for w in cfg['widths']: layers.append(Layer(n_prev, w, cfg.get('kind', 'lut2'))); n_prev = w
net = torch.nn.ModuleList(layers); net.load_state_dict(ck["net"], strict=False)
X = Xte / 255.0
def tri_lut(lo_a, hi_a, lo_b, hi_b, T):
    # each signal: (can_be_0, can_be_1) encoded as lo (min value) / hi (max value) in {0,1}
    lo = torch.ones_like(lo_a); hi = torch.zeros_like(lo_a)
    for av in (0, 1):
        for bv in (0, 1):
            ok = ((lo_a <= av) & (hi_a >= av) & (lo_b <= bv) & (hi_b >= bv)).float()
            v = T[:, 2 * av + bv]
            lo = torch.where(ok > 0, torch.minimum(lo, v.expand_as(lo)), lo)
            hi = torch.where(ok > 0, torch.maximum(hi, v.expand_as(hi)), hi)
    return lo, hi
res = []
with torch.no_grad():
    exact = None
    for phase in range(len(thr) + 1):
        # bits for thresholds with index < phase are known; others: known 0 if a lower threshold bit is 0, else X
        B = np.stack([(X > t) for t in thr], 2).astype(np.float32)          # S, 784, nthr
        lo = B.copy(); hi = B.copy()
        for k in range(phase, len(thr)):
            known0 = (B[:, :, phase - 1] == 0) if phase > 0 else np.zeros_like(B[:, :, 0], bool)
            lo[:, :, k] = 0; hi[:, :, k] = np.where(known0, 0, 1)
        lo = torch.as_tensor(lo.reshape(len(X), -1)); hi = torch.as_tensor(hi.reshape(len(X), -1))
        for L in net:
            idx = L.c.argmax(1); T = (L.t > 0).float()
            la, ha = lo[:, idx[0::2]], hi[:, idx[0::2]]; lb, hb = lo[:, idx[1::2]], hi[:, idx[1::2]]
            lo, hi = tri_lut(la, ha, lb, hb, T)
        glo, ghi = group(lo), group(hi)
        if phase == len(thr): exact = glo.argmax(1)
        best = glo.argmax(1)
        others = ghi.clone(); others[torch.arange(len(X)), best] = -1
        cert = (glo[torch.arange(len(X)), best] > others.max(1).values)
        res.append((phase, float(cert.float().mean())))
print(json.dumps(dict(cert_by_phase=res, acc=float((exact.numpy() == yte).mean()))))
