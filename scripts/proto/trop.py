"""Max-plus (tropical) feature layer + linear readout, exact sorted-scan evaluation cost.
Dilation d_j = max_n (W_jn + x_n), erosion e_j = min_n (x_n - V_jn). Trained with a
log-sum-exp relaxation (temperature annealed), evaluated with hard max/min in integers.
Cost model: each neuron keeps its template sorted offline (index, value) in 32-bit entries;
the scan stops as soon as no unread entry can change the result (threshold algorithm with
xmax = max of the image). Readout: 4-bit dense rows over the features (bit-plane words).
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys, time
import numpy as np
sys.path.insert(0, HERE + '')
from data import load

def lse(Z, tau):                       # tau * log sum exp(Z / tau) over the last axis, and softmax
    m = Z.max(-1, keepdims=True); E = np.exp((Z - m) / tau); s = E.sum(-1, keepdims=True)
    return (m[..., 0] + tau * np.log(s[..., 0])), E / s

def forward(P, X, tau):
    Hd, Sd = lse(P['W'][None] + X[:, None, :], tau)            # dilation
    He, Se = lse(P['V'][None] - X[:, None, :], tau)            # erosion = -max(V - x)
    F = np.concatenate([Hd, -He], 1)
    return F, Sd, Se

def train(rng, X, y, Hd, epochs, lr=3e-3, bs=64, tau0=0.3, tau1=0.01):
    n = X.shape[1]
    P = dict(W=rng.normal(-0.6, 0.3, (Hd, n)).astype(np.float32),
             V=rng.normal(-0.6, 0.3, (Hd, n)).astype(np.float32),
             A=(rng.normal(0, 0.1, (10, 2 * Hd))).astype(np.float32), c=np.zeros(10, np.float32),
             mu=np.zeros(2 * Hd, np.float32))
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}; t = 0
    steps = epochs * ((len(y) + bs - 1) // bs); it = 0
    for ep in range(epochs):
        idx = rng.permutation(len(y))
        for k in range(0, len(idx), bs):
            tau = tau0 * (tau1 / tau0) ** (it / max(1, steps - 1)); it += 1
            bi = idx[k:k + bs]; Xb = X[bi]
            F, Sd, Se = forward(P, Xb, tau)
            Fc = F - P['mu']                                    # centred features
            z = Fc @ P['A'].T + P['c']
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g = p; g[np.arange(len(bi)), y[bi]] -= 1; g /= len(bi)
            G = {'A': g.T @ Fc, 'c': g.sum(0)}
            dF = g @ P['A']; G['mu'] = -dF.sum(0)
            G['W'] = np.einsum('bh,bhn->hn', dF[:, :Hd], Sd)
            G['V'] = np.einsum('bh,bhn->hn', -dF[:, Hd:], Se)
            t += 1
            for kk in P:
                m[kk] = 0.9 * m[kk] + 0.1 * G[kk]; v2[kk] = 0.999 * v2[kk] + 0.001 * G[kk] ** 2
                P[kk] -= lr * (m[kk] / (1 - 0.9 ** t)) / (np.sqrt(v2[kk] / (1 - 0.999 ** t)) + 1e-8)
        P['mu'] = 0 * P['mu']                                   # fold the centring into c below
    return P

def evaluate(P, Xq, y, Xcal):
    """integers: x in 0..255, templates rounded to the same scale (255 = 1.0)."""
    Wi = np.rint(P['W'] * 255).astype(np.int64); Vi = np.rint(P['V'] * 255).astype(np.int64)
    Hd = Wi.shape[0]
    F = np.concatenate([(Wi[None] + Xq[:, None, :]).max(2), (Xq[:, None, :] - Vi[None]).min(2)], 1)
    # scan cost: entries sorted by template value (dilation: W desc; erosion: V desc)
    xmax = Xq.max(1)
    cost = np.zeros(len(y))
    for T, sgn in ((Wi, 1), (Vi, -1)):
        order = np.argsort(-T, 1); Ts = np.take_along_axis(T, order, 1)        # H x n
        for j in range(Hd):
            xs = Xq[:, order[j]]                                               # S x n in scan order
            if sgn > 0:   # dilation: best = max(T + x); stop when T_next + xmax <= best
                run = np.maximum.accumulate(Ts[j][None] + xs, 1)
                nxt = np.append(Ts[j][1:], -10**9)[None] + xmax[:, None]
                stop = (nxt <= run)
            else:         # erosion: worst = min(x - V); stop when 0 - V_next >= worst
                run = np.minimum.accumulate(xs - Ts[j][None], 1)
                nxt = -np.append(Ts[j][1:], -10**9)[None] + 0
                stop = (nxt >= run)
            first = np.where(stop.any(1), stop.argmax(1) + 1, stop.shape[1])
            cost += first                                                       # one 32-bit entry each
    # readout: 4-bit weights on 8-bit requantised features
    lo, hi = np.percentile(F[:2000], 0.5, 0), np.percentile(F[:2000], 99.5, 0)
    Fq = np.clip(np.rint((F - lo) / np.maximum(hi - lo, 1) * 255), 0, 255)
    Ff = np.concatenate(forward(P, Xcal / 255.0, 1e-3)[0:1], 0)
    z_float = None
    A = P['A'] * (np.maximum(hi - lo, 1) / 255.0)[None] / 255.0                 # fold the scaling
    c = P['c'] + (P['A'] * (lo / 255.0)[None]).sum(1)
    s = np.abs(A).max() / 7; Aq = np.clip(np.rint(A / s), -7, 7)
    pred = np.argmax(Fq @ Aq.T + c / s, 1)
    nzw = (Fq.reshape(len(y), -1, 32) != 0).any(2).sum(1) if Fq.shape[1] % 32 == 0 else np.full(len(y), (Fq.shape[1] + 31) // 32)
    cost_ro = 10 * 4 * nzw
    return dict(acc=float((pred == y).mean()), scan=float(cost.mean()), readout=float(cost_ro.mean()),
                total=float((cost + cost_ro).mean()))

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); rng = np.random.default_rng(cfg.get('seed', 0))
    Xtr, ytr, Xte, yte = load(cfg['data']); nte = cfg.get('ntest', len(yte)); Xte, yte = Xte[:nte], yte[:nte]
    t0 = time.time()
    P = train(rng, (Xtr / 255.0).astype(np.float32), ytr, cfg['H'], cfg.get('epochs', 30), lr=cfg.get('lr', 3e-3))
    r = evaluate(P, Xte, yte, Xtr[:2000]); r['cfg'] = cfg; r['secs'] = round(time.time() - t0)
    print(json.dumps(r), flush=True)
