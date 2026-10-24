"""Datasets used by every experiment.
digits: scikit-learn's 8x8 digits (1797 images), fixed split (seed 7): 450 test, 1347 train.
mnist, fmnist: dl/mnist.npz and dl/fmnist.npz, written by get_data.py.
Pixels are returned as integers in 0..255."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    if name == 'digits':
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data.astype(np.int64), d.target.astype(np.int64)
        perm = np.random.default_rng(7).permutation(len(y)); te, tr = perm[:450], perm[450:]
        Xq = np.rint(X * 255.0 / 16.0).astype(np.int64)
        return Xq[tr], y[tr], Xq[te], y[te]
    if name in ('mnist', 'fmnist'):
        d = np.load(os.path.join(HERE, 'dl', name + '.npz'))
        return (d['x_train'].reshape(-1, 784).astype(np.int64), d['y_train'].astype(np.int64),
                d['x_test'].reshape(-1, 784).astype(np.int64), d['y_test'].astype(np.int64))
    raise ValueError(name)
