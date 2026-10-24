"""Rebuild dl/mnist.npz and dl/fmnist.npz from the official IDX files and check that the array
content is bit-identical to the data used in the experiments.
Usage: python3 get_data.py   (writes into ./dl next to this script)"""
import gzip, hashlib, os, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'dl'); os.makedirs(D, exist_ok=True)
SRC = {'mnist': 'https://ossci-datasets.s3.amazonaws.com/mnist/',
       'fmnist': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'}
EXPECTED = {'mnist': 'c868f4cc70f1ec2961bd3b26437ea3a5d0e93cba4d95b336c6d475dd9dabb021', 'fmnist': '59c4ea05a36442d6513afff578fc9c4a49bab9b60a7432bf84452b97809256a8', }
FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
def idx(path):
    raw = gzip.open(path).read(); nd = raw[3]; dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], 'big') for i in range(nd)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(dims)
for name, url in SRC.items():
    arr = []
    for f in FILES:
        p = os.path.join(D, f'{name}-{f}')
        if not os.path.exists(p): urllib.request.urlretrieve(url + f, p)
        arr.append(idx(p))
    xtr, ytr, xte, yte = arr
    if name == 'fmnist': xtr, xte = xtr.reshape(len(xtr), -1), xte.reshape(len(xte), -1)
    out = dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    h = hashlib.sha256(b''.join(np.ascontiguousarray(out[k]).tobytes() for k in sorted(out))).hexdigest()
    np.savez(os.path.join(D, f'{name}.npz'), **out)
    print(name, 'OK, identical to the experiments' if h == EXPECTED[name] else 'MISMATCH ' + h)
