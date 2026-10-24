"""Dominance-Tree Networks (DTN) vs dense ReLU MLPs: accuracy and exact SRAM traffic.

Every model is evaluated bit-exactly in integers. SRAM traffic is counted in 32-bit words of a
bit-plane layout (32 input lanes per word) with the SAME exact skipping rules for every model:
  - zero-word skipping: a word whose 32 activation lanes are all zero is not read;
  - MSB-first certified early termination for rows whose output is 0 (sign-magnitude, sign
    split box bound), and certified argmax on the last layer (synchronous elimination);
  - DTN only: children of a root whose output is 0 are skipped (dominance lemma), and a child is
    computed as acc_parent - D.x - db with D >= 0 read MSB-first (nb planes).
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import numpy as np, sys, json, time

def softplus(z): return np.logaddexp(0, z)
def sigmoid(z): return 0.5 * (1 + np.tanh(0.5 * z))
def requant(acc, M, sh): return np.clip((acc * M + (1 << (sh - 1))) >> sh, 0, 255)
def bitlen(v): return int(v).bit_length() if v > 0 else 0

from data import load  # datasets: see data.py

class Tree:
    def __init__(s, rng, n, R, C):
        s.R, s.C, s.n, s.a = R, C, n, np.sqrt(2 / n)
        s.P = rng.normal(0, np.sqrt(2 / n), (R, n)).astype(np.float32); s.bp = np.zeros(R, np.float32)
        s.T = rng.normal(0, 1.0, (R, C, n)).astype(np.float32); s.B = np.zeros((R, C), np.float32)
    def params(s): return [s.P, s.bp, s.T, s.B]
    def W(s): return np.concatenate([s.P, (s.P[:, None, :] - s.a * softplus(s.T)).reshape(-1, s.n)])
    def b(s): return np.concatenate([s.bp, (s.bp[:, None] - s.a * softplus(s.B)).reshape(-1)])
    def grads(s, dW, db):
        R, C, n = s.R, s.C, s.n
        dWr, dWc = dW[:R], dW[R:].reshape(R, C, n); dbr, dbc = db[:R], db[R:].reshape(R, C)
        return [dWr + dWc.sum(1), dbr + dbc.sum(1), -s.a * dWc * sigmoid(s.T), -s.a * dbc * sigmoid(s.B)]

class Dense:
    def __init__(s, rng, n, M): s.Wm = rng.normal(0, np.sqrt(2 / n), (M, n)).astype(np.float32); s.bm = np.zeros(M, np.float32)
    def params(s): return [s.Wm, s.bm]
    def W(s): return s.Wm
    def b(s): return s.bm
    def grads(s, dW, db): return [dW, db]

def train(rng, layers, Xf, y, epochs, lr=2e-3, bs=64, lam=0.0, wd=1e-4, lamc=1.0):
    # L1 weight per hidden unit: children of a tree layer get lam*lamc
    lw = []
    for l in layers[:-1]:
        if isinstance(l, Tree): lw.append((np.concatenate([np.ones(l.R), np.full(l.R * l.C, lamc)]) * lam).astype(np.float32))
        else: lw.append(np.full(l.W().shape[0], lam, np.float32))
    ps = [p for l in layers for p in l.params()]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]; t = 0
    for ep in range(epochs):
        lr_e = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))          # cosine schedule
        idx = rng.permutation(len(y))
        for k in range(0, len(idx), bs):
            bi = idx[k:k + bs]; h = [Xf[bi]]; zs = []
            Ws = [l.W() for l in layers]; bs_ = [l.b() for l in layers]
            for li in range(len(layers)):
                z = h[-1] @ Ws[li].T + bs_[li]; zs.append(z)
                h.append(np.maximum(z, 0) if li < len(layers) - 1 else z)
            p = np.exp(h[-1] - h[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(bi)), y[bi]] -= 1; g /= len(bi)
            gs = []
            for li in range(len(layers) - 1, -1, -1):
                if li < len(layers) - 1:
                    g = (g + lw[li] * (zs[li] > 0) / len(bi)) * (zs[li] > 0)
                dW = g.T @ h[li] + wd * Ws[li]; db = g.sum(0)
                gs = layers[li].grads(dW, db) + gs
                g = g @ Ws[li]
            t += 1
            for i, (pp, gg) in enumerate(zip(ps, gs)):
                m[i] = 0.9 * m[i] + 0.1 * gg; v[i] = 0.999 * v[i] + 0.001 * gg * gg
                pp -= lr_e * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)

def quantize(layers, Xq_cal):
    q = []; s_x = 1 / 255.0; xq = Xq_cal
    for li, l in enumerate(layers):
        W, b = l.W(), l.b(); s_w = np.abs(W).max() / 127.0
        if isinstance(l, Tree):
            Pq = np.clip(np.rint(l.P / s_w), -127, 127).astype(np.int64)
            Dq = np.clip(np.rint(l.a * softplus(l.T) / s_w), 0, 255).astype(np.int64)
            bpq = np.rint(l.bp / (s_w * s_x)).astype(np.int64)
            dbq = np.rint(l.a * softplus(l.B) / (s_w * s_x)).astype(np.int64)
            Wq = np.concatenate([Pq, (Pq[:, None, :] - Dq).reshape(-1, l.n)])
            bq = np.concatenate([bpq, (bpq[:, None] - dbq).reshape(-1)])
            tree = dict(R=l.R, C=l.C, D=Dq, db=dbq)
        else:
            Wq = np.clip(np.rint(W / s_w), -127, 127).astype(np.int64)
            bq = np.rint(b / (s_w * s_x)).astype(np.int64); tree = None
        lay = dict(W=Wq, b=bq, tree=tree, kind='relu' if li < len(layers) - 1 else 'argmax')
        acc = xq @ Wq.T + bq
        if li < len(layers) - 1:
            yf = np.maximum(acc, 0) * (s_w * s_x); s_y = max(np.percentile(yf, 99.99), 1e-9) / 255.0
            mult = s_w * s_x / s_y; sh = int(15 - np.ceil(np.log2(mult))); M = int(np.rint(mult * (1 << sh)))
            lay.update(M=M, sh=sh); xq = requant(acc, M, sh); s_x = s_y
        q.append(lay)
    return q

def nz_words(H):
    S, N = H.shape; G = (N + 31) // 32
    pad = np.zeros((S, G * 32), dtype=H.dtype); pad[:, :N] = H
    return (pad.reshape(S, G, 32) != 0).any(2).sum(1)              # per sample

def relu_rows_words(W, b, H, out, T0, nzw):
    """words read per (sample,row) for rows computed from full int8 weights."""
    S = H.shape[0]; Mr = W.shape[0]
    m = np.abs(W); sg = np.sign(W)
    Spos = H @ (W > 0).T                                            # S x M
    planes = np.full((S, Mr), 8)
    done = np.zeros((S, Mr), bool)
    for k in range(7, -1, -1):
        hi = sg * ((m >> k) << k)
        U = H @ hi.T + b + ((1 << k) - 1) * Spos
        newly = (U <= T0) & ~done
        planes[newly] = 1 + 7 - k; done |= newly
    planes[out > 0] = 8
    return planes * nzw[:, None]

def evaluate(q, Xq, y, chunk=500):
    tot = np.zeros(len(q)); correct = 0; viol = 0; act = np.zeros(len(q)); n = len(y)
    for s0 in range(0, n, chunk):
        H = Xq[s0:s0 + chunk]; yy = y[s0:s0 + chunk]; S = len(yy)
        for li, l in enumerate(q):
            W, b = l['W'], l['b']; Mr, N = W.shape
            nzw = nz_words(H); acc = H @ W.T + b
            if l['kind'] == 'relu':
                out = requant(acc, l['M'], l['sh']); T0 = ((1 << (l['sh'] - 1)) - 1) // l['M']
                act[li] += (out > 0).mean() * S
                t = l['tree']
                if t is None:
                    tot[li] += relu_rows_words(W, b, H, out, T0, nzw).sum()
                else:
                    R, C, D, db = t['R'], t['C'], t['D'], t['db']
                    nb = max(1, bitlen(D.max()))
                    tot[li] += relu_rows_words(W[:R], b[:R], H, out[:, :R], T0, nzw).sum()
                    for r in range(R):
                        on = out[:, r] > 0                              # samples where root r is active
                        kids = out[:, R + r * C: R + (r + 1) * C]
                        viol += int((kids[~on] > 0).sum())
                        if not on.any(): continue
                        Hs = H[on]; ap = acc[on, r]; nz = nzw[on]
                        for c in range(C):
                            Dr = D[r, c]; ub = ap - db[r, c]
                            planes = np.full(on.sum(), nb); done = np.zeros(on.sum(), bool)
                            for k in range(nb - 1, -1, -1):
                                hi = (Dr >> k) << k
                                cert = (ub - Hs @ hi <= T0) & ~done
                                planes[cert] = nb - k; done |= cert
                            planes[kids[on, c] > 0] = nb
                            tot[li] += (planes * nz).sum()
                H = out
            else:
                pred = np.argmax(acc, 1); correct += int((pred == yy).sum())
                # certified argmax: all rows read MSB-first in lockstep until one class dominates
                m = np.abs(W); sg = np.sign(W)
                Spos = H @ (W > 0).T; Sneg = H @ (W < 0).T
                planes = np.full(S, 8)
                done = np.zeros(S, bool)
                for k in range(7, -1, -1):
                    part = H @ (sg * ((m >> k) << k)).T + b
                    L = part - ((1 << k) - 1) * Sneg; U = part + ((1 << k) - 1) * Spos
                    best = np.argmax(L, 1)
                    Uo = U.copy(); Uo[np.arange(S), best] = np.iinfo(np.int64).min
                    cert = (L[np.arange(S), best] > Uo.max(1)) & ~done
                    planes[cert] = 1 + 7 - k; done |= cert
                tot[li] += (planes * nzw * Mr).sum()
    return dict(acc=correct / n, words=(tot / n).round(1).tolist(), total=float(tot.sum() / n),
                b0=float(sum(l['W'].size * 8 / 32 for l in q)), active=(act / n).round(4).tolist(), viol=viol)

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1])
    rng = np.random.default_rng(cfg.get('seed', 0))
    Xtr, ytr, Xte, yte = load(cfg['data'])
    nte = cfg.get('ntest', len(yte)); Xte, yte = Xte[:nte], yte[:nte]
    Xf = (Xtr / 255.0).astype(np.float32); n_in = Xtr.shape[1]
    arch = cfg['arch']   # list of hidden layers: [M] dense or [R, C] tree
    layers = []; n = n_in
    for h in arch:
        if cfg['kind'] == 'dense' or len(h) == 1:
            M = h[0] if len(h) == 1 else h[0] * (1 + h[1]); layers.append(Dense(rng, n, M)); n = M
        else:
            layers.append(Tree(rng, n, h[0], h[1])); n = h[0] * (1 + h[1])
    layers.append(Dense(rng, n, 10))
    t0 = time.time()
    train(rng, layers, Xf, ytr, cfg.get('epochs', 100), lr=cfg.get('lr', 2e-3), lam=cfg.get('lam', 0.0), lamc=cfg.get('lamc', 1.0))
    q = quantize(layers, Xtr[:5000])
    r = evaluate(q, Xte, yte); r['cfg'] = cfg; r['secs'] = round(time.time() - t0)
    print(json.dumps(r), flush=True)
