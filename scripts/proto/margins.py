"""Top-1 / top-2 vote-margin distribution of a trained single-layer lut2 model (lgf.py format)."""
import sys, json, torch, numpy as np
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__)); sys.path.insert(0, HERE)
from data import load
from lgf import Layer, group
ck = torch.load(sys.argv[1], weights_only=False); cfg = ck['cfg']
_, _, Xte, yte = load(cfg['data']); thr = cfg.get('thr', [0.25, 0.5, 0.75])
E = torch.as_tensor(np.stack([(Xte / 255.0 > t) for t in thr], 2).reshape(len(Xte), -1).astype(np.float32))
layers = []; n = E.shape[1]
for w in cfg['widths']: layers.append(Layer(n, w, 'lut2')); n = w
net = torch.nn.ModuleList(layers); net.load_state_dict(ck['net'], strict=False)
with torch.no_grad():
    x = E
    for L in net: x = L(x, 0.03, True)
    S = group(x).numpy()
top = np.sort(S, 1); m = top[:, -1] - top[:, -2]
print(json.dumps(dict(model=_os.path.basename(sys.argv[1]), acc=float((S.argmax(1) == yte).mean()),
                      voters_per_class=cfg['widths'][-1] // 10,
                      p_margin_le={d: float((m <= d).mean()) for d in [0, 1, 2, 3, 5, 8, 12]})))
