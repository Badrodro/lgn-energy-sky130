"""Convolutional logic fabric: FIXED translation-invariant wiring, truth tables SHARED per channel.
Layer types (spec list):
  {"t":"conv","c":C,"s":stride,"r":radius}  each output channel k reads 2 inputs at fixed offsets
                                            (dy,dx,channel) drawn once per channel (same at every position)
  {"t":"pool"}                              fixed 2x2 OR pooling (no configuration)
Configuration = 4 bits per conv channel (broadcast to every position), versus 4 bits per gate.
Output: last layer, gate i -> class i mod 10, class score = popcount.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, HERE + '')
from data import load
from lgn_np import C16, softmax

def build(g, in_shape, spec):
    """returns a list of layers: ('conv', ia, ib, ch, C, shape) or ('pool', idx4, shape)."""
    ph, pw, pc = in_shape; out = []
    for L in spec:
        if L['t'] == 'pool':
            h, w = ph // 2, pw // 2
            y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(pc), indexing='ij')
            idx = [((2 * y + dy) * pw + (2 * x + dx)) * pc + c for dy in (0, 1) for dx in (0, 1)]
            out.append(('pool', [i.ravel() for i in idx], (h, w, pc))); ph, pw = h, w; continue
        s, r, C = L.get('s', 1), L.get('r', 1), L['c']
        h, w = (ph + s - 1) // s, (pw + s - 1) // s
        off = [(g.integers(-r, r + 1, C), g.integers(-r, r + 1, C), g.integers(0, pc, C)) for _ in range(2)]
        y, x, k = np.meshgrid(np.arange(h), np.arange(w), np.arange(C), indexing='ij'); y, x, k = y.ravel(), x.ravel(), k.ravel()
        src = []
        for dy, dx, cc in off:
            yy = np.clip(y * s + dy[k], 0, ph - 1); xx = np.clip(x * s + dx[k], 0, pw - 1)
            src.append((yy * pw + xx) * pc + cc[k])
        out.append(('conv', src[0], src[1], k, C, (h, w, C))); ph, pw, pc = h, w, C
    return out

class Fabric:
    def __init__(self, layers, g, res):
        self.layers = layers; self.theta = []
        for L in layers:
            if L[0] == 'conv':
                t = (g.standard_normal((L[4], 16)) * 0.1).astype(np.float32)
                if res: t[:, 3] = float(res)
                self.theta.append(t)
            else: self.theta.append(None)
        self.n_prev = []
    def forward(self, x, hard=False, keep=False):
        cache = []
        for L, t in zip(self.layers, self.theta):
            if L[0] == 'pool':
                ins = [x[:, i] for i in L[1]]; q = [1 - v for v in ins]
                prod = q[0] * q[1] * q[2] * q[3]; y = 1 - prod
                if keep: cache.append((q, prod))
            else:
                a, b = x[:, L[1]], x[:, L[2]]
                if hard: kc = C16[t.argmax(1)]; P = None
                else: P = softmax(t); kc = P @ C16
                k = kc[L[3]]
                y = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * (a * b)
                if keep: cache.append((a, b, P, k))
            x = y
        return x, cache
    def prepare(self, n_in):
        self.S = []; n = n_in
        for L in self.layers:
            if L[0] == 'pool':
                w = len(L[1][0]); self.S.append([sp.csr_matrix((np.ones(w, np.float32), (np.arange(w), i)), shape=(w, n)) for i in L[1]])
            else:
                w = len(L[1]); self.S.append([sp.csr_matrix((np.ones(w, np.float32), (np.arange(w), i)), shape=(w, n)) for i in (L[1], L[2])])
                self.S[-1].append(sp.csr_matrix((np.ones(w, np.float32), (L[3], np.arange(w))), shape=(L[4], w)))
            n = w
    def backward(self, g, cache):
        grads = [None] * len(self.layers)
        for li in range(len(self.layers) - 1, -1, -1):
            L = self.layers[li]
            if L[0] == 'pool':
                q, prod = cache[li]
                gin = [g * (q[(i + 1) % 4] * q[(i + 2) % 4] * q[(i + 3) % 4]) for i in range(4)]   # prod of the others
                g = sum(gi @ S for gi, S in zip(gin, self.S[li]))
                continue
            a, b, P, k = cache[li]; ab = a * b
            dk_gate = np.stack([g.sum(0), (g * a).sum(0), (g * b).sum(0), (g * ab).sum(0)], 1)
            dk = self.S[li][2] @ dk_gate                                     # sum over positions per channel
            dP = dk @ C16.T
            grads[li] = P * (dP - (dP * P).sum(1, keepdims=True))
            if li == 0: break
            ga = g * (k[:, 1] + k[:, 3] * b); gb = g * (k[:, 2] + k[:, 3] * a)
            g = ga @ self.S[li][0] + gb @ self.S[li][1]
        return grads

def group(y, k=10): return y.reshape(y.shape[0], -1, k).sum(1)

def run(cfg):
    g = np.random.default_rng(cfg.get('seed', 0))
    Xtr, ytr, Xte, yte = load(cfg['data']); nte = cfg.get('ntest', len(yte)); Xte, yte = Xte[:nte], yte[:nte]
    thr = cfg.get('thr', [0.5]); side = int(round(Xtr.shape[1] ** 0.5))
    enc = lambda X: np.stack([(X / 255.0 > t) for t in thr], 2).reshape(len(X), -1).astype(np.float32)
    Etr, Ete = enc(Xtr), enc(Xte)
    layers = build(np.random.default_rng(cfg.get('wseed', 1)), (side, side, len(thr)), cfg['spec'])
    net = Fabric(layers, g, cfg.get('res', 2.0)); net.prepare(Etr.shape[1])
    nl = len(layers[-1][1]) if layers[-1][0] == 'conv' else len(layers[-1][1][0])
    tau = cfg.get('tau', (nl / 10) ** 0.5); lr = cfg.get('lr', 0.02); bs = cfg.get('bs', 128)
    th = [t for t in net.theta if t is not None]
    m = [np.zeros_like(t) for t in th]; v = [np.zeros_like(t) for t in th]; st = 0; t0 = time.time()
    for ep in range(cfg['epochs']):
        perm = g.permutation(len(ytr))
        for k0 in range(0, len(perm), bs):
            bi = perm[k0:k0 + bs]
            y, cache = net.forward(Etr[bi], keep=True)
            p = softmax(group(y) / tau); p[np.arange(len(bi)), ytr[bi]] -= 1; p /= len(bi)
            gy = (np.repeat(p[:, None, :], nl // 10, 1).reshape(len(bi), -1) / tau).astype(np.float32)
            grads = [gr for gr in net.backward(gy, cache) if gr is not None]; st += 1
            for i, gr in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * gr; v[i] = 0.999 * v[i] + 0.001 * gr * gr
                th[i] -= lr * (m[i] / (1 - 0.9 ** st)) / (np.sqrt(v[i] / (1 - 0.999 ** st)) + 1e-8)
    ys, _ = net.forward(Ete); yh, _ = net.forward(Ete, hard=True)
    x = Ete; tog = 0.0; gates = 0; pools = 0
    for L, t in zip(net.layers, net.theta):
        if L[0] == 'pool':
            x = 1 - np.prod([1 - x[:, i] for i in L[1]], 0); pools += len(L[1][0])
        else:
            k = C16[t.argmax(1)][L[3]]; a, b = x[:, L[1]], x[:, L[2]]
            x = k[:, 0] + k[:, 1] * a + k[:, 2] * b + k[:, 3] * a * b; gates += len(L[1])
        tog += float((x[1:] != x[:-1]).mean(0).sum())
    cfg_bits = 4 * sum(t.shape[0] for t in th)
    return dict(acc_soft=float((group(ys).argmax(1) == yte).mean()), acc=float((group(yh).argmax(1) == yte).mean()),
                gates=gates, or4=pools, cfg_bits=cfg_bits, toggles=tog, cfg=cfg, secs=round(time.time() - t0))

if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))), flush=True)
