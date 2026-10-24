"""Numpy logic-gate network (same model as lgn.py, manual backprop, float32).
Fixed wiring (random or local columnar), learned 2-input gate functions (softmax over 16 ops),
interleaved group-sum output. Reports hard (discretised) accuracy, gate count, toggles, wire length.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time
import numpy as np
sys.path.insert(0, HERE + '')
from data import load

C16 = np.array([[0,0,0,0],[0,0,0,1],[0,1,0,-1],[0,1,0,0],[0,0,1,-1],[0,0,1,0],[0,1,1,-2],[0,1,1,-1],
                [1,-1,-1,1],[1,-1,-1,2],[1,0,-1,0],[1,0,-1,1],[1,-1,0,0],[1,-1,0,1],[1,0,0,-1],[1,0,0,0]], np.float32)

def random_wiring(g, n_in, widths):
    W = []; n = n_in; L = []
    for w in widths:
        need = 2 * w; src = np.concatenate([g.permutation(n) for _ in range((need + n - 1) // n)])[:need]
        a, b = src[0::2].copy(), src[1::2].copy(); same = a == b
        b[same] = (b[same] + 1 + g.integers(0, n - 1, same.sum())) % n
        W.append((a, b)); n = w
    return W, None

def butterfly_wiring(g, n_in, width, depth, first='random'):
    """first layer: random from the inputs; then butterfly stages: gate i reads i and i ^ 2^s."""
    W, _ = random_wiring(g, n_in, [width]); lg = int(np.log2(width)); assert 1 << lg == width
    idx = np.arange(width)
    for l in range(1, depth):
        s = (l - 1) % lg
        W.append((idx.copy(), idx ^ (1 << s)))
    return W, None

def local_wiring(g, in_shape, layers, r, balance=False):
    W = []; L = []; ph, pw, pc = in_shape
    for (h, w, c) in layers:
        ys, xs, cs = np.meshgrid(np.arange(h), np.arange(w), np.arange(c), indexing='ij')
        ys, xs = ys.ravel(), xs.ravel(); n = h * w * c
        cy = np.rint((ys + 0.5) * ph / h - 0.5).astype(int); cx = np.rint((xs + 0.5) * pw / w - 0.5).astype(int)
        pair = []
        for _ in range(2):
            yy = np.clip(cy + g.integers(-r, r + 1, n), 0, ph - 1); xx = np.clip(cx + g.integers(-r, r + 1, n), 0, pw - 1)
            pair.append((yy * pw + xx) * pc + g.integers(0, pc, n)); L.append(np.hypot(yy - cy, xx - cx))
        a, b = pair; same = a == b; b[same] = (b[same] + 1) % (ph * pw * pc)
        if balance:   # re-point duplicated inputs to unused outputs of the previous layer within the window
            npv = ph * pw * pc; use = np.bincount(np.concatenate([a, b]), minlength=npv)
            uy, ux = np.divmod(np.arange(npv) // pc, pw)
            for u in g.permutation(np.flatnonzero(use == 0)):
                cand = np.flatnonzero((np.abs(cy - uy[u]) <= r) & (np.abs(cx - ux[u]) <= r))
                for gi in g.permutation(cand):
                    side = a if use[a[gi]] > 1 else (b if use[b[gi]] > 1 else None)
                    if side is None or (a[gi] if side is b else b[gi]) == u: continue
                    use[side[gi]] -= 1; side[gi] = u; use[u] += 1; break
            L[-2] = np.hypot(np.divmod(a // pc, pw)[0] - cy, np.divmod(a // pc, pw)[1] - cx)
            L[-1] = np.hypot(np.divmod(b // pc, pw)[0] - cy, np.divmod(b // pc, pw)[1] - cx)
        W.append((a, b)); ph, pw, pc = h, w, c
    return W, np.concatenate(L)

def softmax(t):
    e = np.exp(t - t.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

class Net:
    def __init__(self, wire, widths, g, res=5.0):
        self.wire = wire; self.theta = []
        for w in widths:
            t = (g.standard_normal((w, 16)) * 0.1).astype(np.float32)
            if res: t[:, 3] = float(res)
            self.theta.append(t)
    def forward(self, x, hard=False, keep=False):
        cache = []
        for (ia, ib), t in zip(self.wire, self.theta):
            a, b = x[:, ia], x[:, ib]
            if hard: k = C16[t.argmax(1)]; P = None
            else: P = softmax(t); k = P @ C16
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * (a * b)
            if keep: cache.append((a, b, P, k))
        return x, cache
    def backward(self, gy, cache, n_in):
        grads = [None] * len(self.theta); g = gy
        for li in range(len(self.theta) - 1, -1, -1):
            a, b, P, k = cache[li]; ab = a * b
            dk = np.stack([g.sum(0), (g * a).sum(0), (g * b).sum(0), (g * ab).sum(0)], 1)     # w x 4
            dP = dk @ C16.T                                                                    # w x 16
            grads[li] = P * (dP - (dP * P).sum(1, keepdims=True))
            if li == 0: break
            ga = g * (k[:, 1] + k[:, 3] * b); gb = g * (k[:, 2] + k[:, 3] * a)
            ia, ib = self.wire[li]; n_prev = self.theta[li - 1].shape[0]
            gp = np.zeros((g.shape[0], n_prev), np.float32)
            # scatter-add over columns (fan-out): use a sparse matrix product
            gp += ga @ self.Sa[li] + gb @ self.Sb[li]
            g = gp
        return grads
    def build_scatter(self, n_in):
        import scipy.sparse as sp
        self.Sa, self.Sb = [None], [None]
        for li in range(1, len(self.wire)):
            ia, ib = self.wire[li]; w = len(ia); n_prev = self.theta[li - 1].shape[0]
            self.Sa.append(sp.csr_matrix((np.ones(w, np.float32), (np.arange(w), ia)), shape=(w, n_prev)))
            self.Sb.append(sp.csr_matrix((np.ones(w, np.float32), (np.arange(w), ib)), shape=(w, n_prev)))

def group(y, k=10):
    n = y.shape[1] - y.shape[1] % k
    return y[:, :n].reshape(y.shape[0], -1, k).sum(1)

def run(cfg):
    g = np.random.default_rng(cfg.get('seed', 0))
    Xtr, ytr, Xte, yte = load(cfg['data']); nte = cfg.get('ntest', len(yte)); Xte, yte = Xte[:nte], yte[:nte]
    thr = cfg.get('thr', [0.5])
    enc = lambda X: np.stack([(X / 255.0 > t) for t in thr], 2).reshape(len(X), -1).astype(np.float32)  # (pixel, thr)
    Etr, Ete = enc(Xtr), enc(Xte); side = int(round(Xtr.shape[1] ** 0.5))
    gw = np.random.default_rng(cfg.get('wseed', 1))
    if cfg.get('local'):
        layers = [tuple(l) for l in cfg['local']]
        wire, wl = local_wiring(gw, (side, side, len(thr)), layers, cfg.get('r', 2), cfg.get('bal', False)); widths = [h * w * c for h, w, c in layers]
    elif cfg.get('butterfly'):
        widths = [cfg['width']] * cfg['depth']; wire, wl = butterfly_wiring(gw, Etr.shape[1], cfg['width'], cfg['depth'])
    else:
        widths = [cfg['width']] * cfg['depth']; wire, wl = random_wiring(gw, Etr.shape[1], widths)
    net = Net(wire, widths, g, cfg.get('res', 5.0)); net.build_scatter(Etr.shape[1])
    tau = cfg.get('tau', (widths[-1] / 10) ** 0.5); lr = cfg.get('lr', 0.02); bs = cfg.get('bs', 128)
    m = [np.zeros_like(t) for t in net.theta]; v = [np.zeros_like(t) for t in net.theta]; st = 0; t0 = time.time()
    for ep in range(cfg['epochs']):
        perm = g.permutation(len(ytr))
        for k0 in range(0, len(perm), bs):
            bi = perm[k0:k0 + bs]
            y, cache = net.forward(Etr[bi], keep=True)
            s = group(y) / tau; p = softmax(s); p[np.arange(len(bi)), ytr[bi]] -= 1; p /= len(bi)
            gy = np.zeros((len(bi), widths[-1]), np.float32)
            gy[:, :10 * (widths[-1] // 10)] = np.repeat(p[:, None, :], widths[-1] // 10, 1).reshape(len(bi), -1) / tau
            grads = net.backward(gy.astype(np.float32), cache, Etr.shape[1]); st += 1
            for i, gr in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * gr; v[i] = 0.999 * v[i] + 0.001 * gr * gr
                net.theta[i] -= lr * (m[i] / (1 - 0.9 ** st)) / (np.sqrt(v[i] / (1 - 0.999 ** st)) + 1e-8)
    ys, _ = net.forward(Ete); yh, _ = net.forward(Ete, hard=True)
    # toggles between consecutive test images, per layer
    x = Ete; tog = 0.0
    for (ia, ib), t in zip(net.wire, net.theta):
        k = C16[t.argmax(1)]; a, b = x[:, ia], x[:, ib]
        x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b; tog += float((x[1:] != x[:-1]).mean(0).sum())
    ops = np.bincount(np.concatenate([t.argmax(1) for t in net.theta]), minlength=16)
    if cfg.get('save'): np.savez(cfg['save'], scores=group(yh), y=yte, theta=np.array(net.theta, dtype=object), allow_pickle=True)
    return dict(acc_soft=float((group(ys).argmax(1) == yte).mean()), acc=float((group(yh).argmax(1) == yte).mean()),
                gates=int(sum(widths)), toggles=tog, wire_mean=None if wl is None else float(wl.mean()),
                passthru=int(ops[[3, 5]].sum()), const=int(ops[[0, 15]].sum()), cfg=cfg, secs=round(time.time() - t0))

if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))), flush=True)
