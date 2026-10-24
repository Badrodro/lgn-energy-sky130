"""Dominance-tree ReLU layers: exact conditional computation by construction.

Layer = forest of R roots, each with C children. Child weights w_c = w_p - D_c with D_c >= 0
elementwise and b_c <= b_p. For x >= 0: acc_c <= acc_p, requant is monotone, so an inactive
parent (output 0) proves all its children inactive: they are skipped exactly.
"""
from sklearn.datasets import load_digits as _load_digits
import numpy as np, sys, json

rng = np.random.default_rng(0)
d = dict(X=_load_digits().data, y=_load_digits().target)  # scikit-learn 8x8 digits
X, y = d['X'].astype(np.int64), d['y'].astype(np.int64)
perm = np.random.default_rng(7).permutation(len(y)); te, tr = perm[:450], perm[450:]
Xq = np.rint(X * 255.0 / 16.0).astype(np.int64)
Xf = Xq / 255.0

def softplus(z): return np.logaddexp(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

class TreeLayer:
    def __init__(s, n_in, R, C, init_gap=0.0, alpha=None):
        s.R, s.C, s.n = R, C, n_in
        s.a = alpha if alpha is not None else np.sqrt(2 / n_in)   # gap scale ~ weight scale
        s.P = rng.normal(0, np.sqrt(2 / n_in), (R, n_in)); s.bp = np.zeros(R)
        s.T = init_gap + rng.normal(0, 1.0, (R, C, n_in))
        s.B = np.full((R, C), init_gap)
    def params(s): return [s.P, s.bp, s.T, s.B]
    def W(s):
        Wc = s.P[:, None, :] - s.a * softplus(s.T)
        return np.concatenate([s.P, Wc.reshape(-1, s.n)])
    def b(s):
        bc = s.bp[:, None] - s.a * softplus(s.B)
        return np.concatenate([s.bp, bc.reshape(-1)])
    def grads(s, dW, db):
        R, C, n = s.R, s.C, s.n
        dWr, dWc = dW[:R], dW[R:].reshape(R, C, n)
        dbr, dbc = db[:R], db[R:].reshape(R, C)
        return [dWr + dWc.sum(1), dbr + dbc.sum(1), -s.a * dWc * sigmoid(s.T), -s.a * dbc * sigmoid(s.B)]

class DenseLayer:
    def __init__(s, n_in, M):
        s.Wm = rng.normal(0, np.sqrt(2 / n_in), (M, n_in)); s.bm = np.zeros(M)
    def params(s): return [s.Wm, s.bm]
    def W(s): return s.Wm
    def b(s): return s.bm
    def grads(s, dW, db): return [dW, db]

def train(layers, epochs=200, lr=2e-3, bs=64, lam=0.0, wd=1e-4):
    ps = [p for l in layers for p in l.params()]
    m = [np.zeros_like(p) for p in ps]; v = [np.zeros_like(p) for p in ps]; t = 0
    Xt, yt = Xf[tr], y[tr]
    for ep in range(epochs):
        idx = rng.permutation(len(yt))
        for k in range(0, len(idx), bs):
            bi = idx[k:k + bs]; h = [Xt[bi]]; zs = []
            for li, l in enumerate(layers):
                z = h[-1] @ l.W().T + l.b(); zs.append(z)
                h.append(np.maximum(z, 0) if li < len(layers) - 1 else z)
            p = np.exp(h[-1] - h[-1].max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(bi)), yt[bi]] -= 1; g /= len(bi)
            gs = []
            for li in range(len(layers) - 1, -1, -1):
                l = layers[li]
                if li < len(layers) - 1:
                    g = g + lam * (zs[li] > 0) / len(bi)       # L1 on activations
                    g = g * (zs[li] > 0)
                dW = g.T @ h[li] + wd * l.W(); db = g.sum(0)
                gs = l.grads(dW, db) + gs
                g = g @ l.W()
            t += 1
            for i, (pp, gg) in enumerate(zip(ps, gs)):
                m[i] = 0.9 * m[i] + 0.1 * gg; v[i] = 0.999 * v[i] + 0.001 * gg * gg
                pp -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return layers

def requant(acc, M, sh): return np.clip((acc * M + (1 << (sh - 1))) >> sh, 0, 255)

def quantize(layers, dmax_bits=8):
    """int8 roots, integer gaps D >= 0 (exact dominance), int bias, requant M*2^-sh."""
    q = []; s_x = 1 / 255.0; xq = Xq[tr]
    for li, l in enumerate(layers):
        W, b = l.W(), l.b()
        s_w = np.abs(W).max() / 127.0
        if isinstance(l, TreeLayer):
            Pq = np.clip(np.rint(l.P / s_w), -127, 127).astype(np.int64)
            Dq = np.clip(np.rint(l.a * softplus(l.T) / s_w), 0, (1 << dmax_bits) - 1).astype(np.int64)
            bpq = np.rint(l.bp / (s_w * s_x)).astype(np.int64)
            dbq = np.rint(l.a * softplus(l.B) / (s_w * s_x)).astype(np.int64)
            Wq = np.concatenate([Pq, (Pq[:, None, :] - Dq).reshape(-1, l.n)])
            bq = np.concatenate([bpq, (bpq[:, None] - dbq).reshape(-1)])
            tree = dict(R=l.R, C=l.C, Dq=Dq)
        else:
            Wq = np.clip(np.rint(W / s_w), -127, 127).astype(np.int64)
            bq = np.rint(b / (s_w * s_x)).astype(np.int64); tree = None
        lay = dict(W=Wq, b=bq, tree=tree, kind='relu' if li < len(layers) - 1 else 'argmax')
        acc = xq @ Wq.T + bq
        if li < len(layers) - 1:
            yf = np.maximum(acc, 0) * (s_w * s_x); s_y = np.percentile(yf, 99.99) / 255.0
            mult = s_w * s_x / s_y; sh = int(15 - np.ceil(np.log2(mult))); M = int(np.rint(mult * (1 << sh)))
            lay.update(M=M, sh=sh); xq = requant(acc, M, sh); s_x = s_y
        q.append(lay)
    return q

def bitlen(v): return int(v).bit_length() if v > 0 else 0

def planes_to_certify_zero(w, x, b, T0, nzw):
    """sign-magnitude, MSB-first: sign plane then magnitude planes 6..0.
    Upper bound after reading magnitudes down to plane k: U = b + sum sign*mag_hi*x + (2^k-1)*S_pos.
    Returns words read until U <= T0 (or all 8 planes). nzw = words with a nonzero x lane."""
    pos = w > 0; Spos = int(x[pos].sum()); m = np.abs(w); sg = np.sign(w)
    for k in range(7, -1, -1):                 # k = number of unread magnitude planes
        hi = (m >> k) << k
        U = b + int((sg * hi) @ x) + ((1 << k) - 1) * Spos
        if U <= T0: return (1 + 7 - k) * nzw
    return 8 * nzw

def gap_words_to_certify(Dr, x, ub_parent, T0, nzw, nb):
    """child acc = parent_acc - Dr.x - db, Dr >= 0, read MSB-first over nb planes.
    Upper bound after reading planes >= k: parent_acc - db - Dr_hi.x. Returns (words, certified)."""
    for k in range(nb - 1, -2, -1):
        if k < 0: return nb * nzw
        hi = (Dr >> k) << k
        if ub_parent - int(hi @ x) <= T0: return (nb - k) * nzw
    return nb * nzw

def evaluate_et(q, Xs):
    """bit-exact inference; words read with MSB-first certified ET + zero-word skipping
    (same rules for dense and tree layers)."""
    correct = 0; words = np.zeros(len(q)); b0 = np.zeros(len(q)); viol = 0
    for x, lab in zip(Xs[0], Xs[1]):
        h = x
        for li, l in enumerate(q):
            W, b = l['W'], l['b']; Mr, N = W.shape
            nzw = sum(1 for g in range(0, N, 32) if h[g:g + 32].any())
            acc = W @ h + b
            b0[li] += Mr * N * 8 / 32
            if l['kind'] == 'relu':
                out = requant(acc, l['M'], l['sh'])
                T0 = ((1 << (l['sh'] - 1)) - 1) // l['M']     # largest acc with output 0
                t = l['tree']
                if t is None:
                    for r in range(Mr):
                        words[li] += planes_to_certify_zero(W[r], h, b[r], T0, nzw) if out[r] == 0 else 8 * nzw
                else:
                    R, C, Dq = t['R'], t['C'], t['Dq']
                    nb = max(1, max(bitlen(v) for v in Dq.ravel()))
                    for r in range(R):
                        words[li] += planes_to_certify_zero(W[r], h, b[r], T0, nzw) if out[r] == 0 else 8 * nzw
                        kids = slice(R + r * C, R + (r + 1) * C)
                        if out[r] == 0:
                            viol += int((out[kids] > 0).sum()); continue
                        for c in range(C):
                            j = R + r * C + c
                            dbias = b[r] - b[j]
                            if out[j] == 0:
                                words[li] += gap_words_to_certify(Dq[r, c], h, int(acc[r]) - dbias, T0, nzw, nb)
                            else:
                                words[li] += nb * nzw
                h = out
            else:
                words[li] += Mr * 8 * nzw          # argmax layer: full read (same for all)
                correct += int(np.argmax(acc) == lab)
    n = len(Xs[1])
    return dict(acc=correct / n, words=(words / n).round(1).tolist(), b0=(b0 / n).round(1).tolist(),
                total=float(words.sum() / n), total_b0=float(b0.sum() / n), dominance_violations=viol)

def evaluate(q, Xs):
    """bit-exact inference + SRAM word count (32-bit words, bit-plane layout)."""
    correct = 0; words = np.zeros(len(q)); b0 = np.zeros(len(q)); act = np.zeros(len(q)); viol = 0
    for x, lab in zip(Xs[0], Xs[1]):
        h = x
        for li, l in enumerate(q):
            W, b = l['W'], l['b']; Mr, N = W.shape
            acc = W @ h + b
            b0[li] += Mr * N * 8 / 32
            if l['kind'] == 'relu':
                out = requant(acc, l['M'], l['sh'])
                t = l['tree']
                if t is None:
                    words[li] += Mr * N * 8 / 32
                else:
                    R, C, Dq = t['R'], t['C'], t['Dq']
                    words[li] += R * N * 8 / 32                      # every root, full int8
                    for r in range(R):
                        kids = out[R + r * C: R + (r + 1) * C]
                        if out[r] == 0:
                            viol += int((kids > 0).sum())          # must stay 0 (dominance)
                            continue
                        for c in range(C):                          # active root: read its children's gaps
                            words[li] += N * max(bitlen(Dq[r, c].max()), 1) / 32
                act[li] += (out > 0).mean()
                h = out
            else:
                words[li] += Mr * N * 8 / 32
                correct += int(np.argmax(acc) == lab)
    n = len(Xs[1])
    return dict(acc=correct / n, words=(words / n).round(1).tolist(), b0=(b0 / n).round(1).tolist(),
                total=float(words.sum() / n), total_b0=float(b0.sum() / n), active=(act / n).round(3).tolist(),
                dominance_violations=viol)

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    lam = cfg.get('lam', 0.0); R1, C1, R2, C2 = cfg.get('arch', [12, 9, 6, 9])
    kind = cfg.get('kind', 'tree')
    if kind == 'tree':
        layers = [TreeLayer(64, R1, C1, cfg.get('gap0', 0.0)), TreeLayer(R1 * (1 + C1), R2, C2, cfg.get('gap0', 0.0)), DenseLayer(R2 * (1 + C2), 10)]
    else:
        layers = [DenseLayer(64, R1 * (1 + C1)), DenseLayer(R1 * (1 + C1), R2 * (1 + C2)), DenseLayer(R2 * (1 + C2), 10)]
    train(layers, epochs=cfg.get('epochs', 200), lam=lam, lr=cfg.get('lr', 2e-3))
    q = quantize(layers, cfg.get('dbits', 8))
    r = evaluate(q, (Xq[te], y[te]))
    r2 = evaluate_et(q, (Xq[te], y[te]))
    r.update(cfg=cfg, et=r2)
    print(json.dumps(r))
