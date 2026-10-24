import sys, json, torch, numpy as np
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from data import load
from lgf import Layer, group
from trilogic import tri_lut
ck = torch.load(sys.argv[1], weights_only=False); cfg = ck['cfg']
_, _, Xte, yte = load(cfg['data']); thr = cfg.get('thr', [0.25, 0.5, 0.75]); assert thr == [0.25, 0.5, 0.75]
layers = []; n_prev = 3 * Xte.shape[1]
for w in cfg['widths']: layers.append(Layer(n_prev, w, 'lut2')); n_prev = w
net = torch.nn.ModuleList(layers); net.load_state_dict(ck['net'], strict=False)
X = Xte / 255.0; B = np.stack([(X > t) for t in thr], 2).astype(np.float32)   # S, 784, 3
lo, hi = B.copy(), B.copy()
m = B[:, :, 1] == 1                       # x > 0.5: bit 0.25 implied 1, bit 0.75 unknown
lo[:, :, 2] = np.where(m, 0, 0); hi[:, :, 2] = np.where(m, 1, 0)
lo[:, :, 0] = np.where(m, 1, 0); hi[:, :, 0] = np.where(m, 1, 1)   # x <= 0.5: bit 0.25 unknown
res = {}
with torch.no_grad():
    for phase, (L0, H0) in {'after_1_comparison_per_pixel': (lo, hi), 'after_2': (B, B)}.items():
        l = torch.as_tensor(L0.reshape(len(X), -1)); h = torch.as_tensor(H0.reshape(len(X), -1))
        for Ly in net:
            idx = Ly.c.argmax(1); T = (Ly.t > 0).float()
            l, h = tri_lut(l[:, idx[0::2]], h[:, idx[0::2]], l[:, idx[1::2]], h[:, idx[1::2]], T)
        gl, gh = group(l), group(h); best = gl.argmax(1)
        oth = gh.clone(); oth[torch.arange(len(X)), best] = -1
        # ties resolved by lowest index, like argmax: certified if lo(best) > hi(others with lower index) and >= hi(higher index)
        idxs = torch.arange(10)[None].expand(len(X), -1)
        lower = idxs < best[:, None]
        ok = torch.where(lower, gl[torch.arange(len(X)), best][:, None] > gh, gl[torch.arange(len(X)), best][:, None] >= gh)
        ok[torch.arange(len(X)), best] = True
        res[phase] = float(ok.all(1).float().mean())
print(json.dumps(dict(model=sys.argv[1], cert=res)))
