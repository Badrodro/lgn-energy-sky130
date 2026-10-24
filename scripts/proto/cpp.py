import numpy as np, sys
d=np.load(sys.argv[1])
L=[]
i=0
while f'L{i}_W' in d.files:
    L.append(dict(W=d[f'L{i}_W'],b=d[f'L{i}_b'],kind=str(d[f'L{i}_kind']),M=d.get(f'L{i}_M'),sh=int(d[f'L{i}_sh']) if f'L{i}_sh' in d.files else None)); i+=1
X=d['Xq_te']; Y=d['yte']
def rq(acc,M,sh): return np.clip((acc*M+(1<<(sh-1)))>>sh,0,255)
def exact(x):
    for l in L:
        acc=l['W']@x+l['b']
        if l['kind']=='relu': x=rq(acc,l['M'],l['sh'])
        else: return np.argmax(acc)  # ties -> lowest index (numpy)
def interval_net(x, k):
    """sign-magnitude weights; only the top k of the 7 magnitude planes are read.
    unread magnitude bits make |w| in [hi, hi + 2^(7-k)-1]."""
    lo=hi=x.astype(np.int64)
    for l in L:
        W=l['W']; s=np.sign(W); m=np.abs(W)
        sh=7-k
        mh=(m>>sh)<<sh; mx=mh+((1<<sh)-1)          # magnitude interval
        mx=np.where(m==0, np.where(mh==0, mx, mh), mx)  # (no info: zero weight still unknown)
        wlo=np.where(s>=0, mh, -mx); whi=np.where(s>=0, mx, -mh)
        wlo=np.where(W>=0, mh, -mx); whi=np.where(W>=0, mx, -mh)  # sign is read (plane 7)
        # products with x in [lo,hi], x>=0
        p1=wlo*lo[None,:]; p2=wlo*hi[None,:]; p3=whi*lo[None,:]; p4=whi*hi[None,:]
        alo=np.minimum(np.minimum(p1,p2),np.minimum(p3,p4)).sum(1)+l['b']
        ahi=np.maximum(np.maximum(p1,p2),np.maximum(p3,p4)).sum(1)+l['b']
        if l['kind']=='relu':
            lo=rq(alo,l['M'],l['sh']); hi=rq(ahi,l['M'],l['sh'])
        else:
            c=np.argmax(alo)  # candidate: best lower bound
            others=np.delete(np.arange(len(alo)),c)
            # certified if c's lower bound beats every other upper bound (strict; ties broken by index)
            ok=all(alo[c]>ahi[j] or (alo[c]==ahi[j] and c<j) for j in others)
            return c if ok else None
B0=sum(l['W'].size*8 for l in L)
ex=np.array([exact(x) for x in X]); print('int acc', (ex==Y).mean())
print('k  certified%  wrong(should be 0)  bits read vs B0 (k planes+sign) | strategy: try k then full')
for k in range(0,8):
    res=[interval_net(x,k) for x in X]
    cert=np.array([r is not None for r in res])
    wrong=sum(1 for r,e in zip(res,ex) if r is not None and r!=e)
    frac=(1+k)/8
    cost=frac + (1-cert.mean())*1.0
    print(k, f'{100*cert.mean():6.1f}', wrong, f'{frac:.3f}', f' expected cost {cost:.3f} of B0')
