import numpy as np, sys
rng=np.random.default_rng(0)
d=np.load(sys.argv[1]); S=int(sys.argv[2]) if len(sys.argv)>2 else 8
L=[];i=0
while f'L{i}_W' in d.files:
    L.append(dict(W=d[f'L{i}_W'],b=d[f'L{i}_b'],kind=str(d[f'L{i}_kind']),M=d[f'L{i}_M'] if f'L{i}_M' in d.files else None,sh=int(d[f'L{i}_sh']) if f'L{i}_sh' in d.files else 0)); i+=1
def rq(acc,M,sh): return np.clip((acc*M+(1<<(sh-1)))>>sh,0,255)
X=d['Xq_te']
def kmeans_groups(W,s,iters=30):
    """balanced-ish grouping of rows into groups of ~s by greedy k-means on weight vectors"""
    M=len(W); G=int(np.ceil(M/s))
    C=W[rng.choice(M,G,replace=False)].astype(float)
    for _ in range(iters):
        D=((W[:,None,:]-C[None,:,:])**2).sum(2)
        # balanced assignment: greedy by distance with capacity s
        order=np.argsort(D,axis=None); cap=np.full(G,s); a=-np.ones(M,int)
        for idx in order:
            r,g=divmod(idx,G)
            if a[r]<0 and cap[g]>0: a[r]=g; cap[g]-=1
        for g in range(G):
            if (a==g).any(): C[g]=W[a==g].mean(0)
    return a,G
def bits(v): v=int(v); return 0 if v<=0 else int(v).bit_length()
print(f'group size {S}')
tot_b0=tot_new=0
xs=[x for x in X]
for li,l in enumerate(L):
    W,b=l['W'],l['b']; M,N=W.shape
    a,G=kmeans_groups(W,S)
    T0 = None
    if l['kind']=='relu':
        # largest acc with output 0:  (acc*M + 2^(sh-1)) >> sh == 0  <=> acc*M < 2^(sh-1)
        T0 = ((1<<(l['sh']-1)) - 1)//l['M']   # per-row (M per row)
    words_b0 = M*N*8/32
    cost=0.0; cert_groups=0; total_groups=0; active=0
    Dbits=[]
    env={}; 
    for g in range(G):
        rows=np.where(a==g)[0]
        E=W[rows].max(0); Dg=E[None,:]-W[rows]
        env[g]=(rows,E,Dg,b[rows].max(), max(bits(Dg.max()),0))
        Dbits.append(env[g][4])
    nxt=[]
    for x in xs:
        acc=W@x+b
        for g,(rows,E,Dg,bmax,bd) in env.items():
            total_groups+=1
            cost += N*9/32               # read the envelope row (int9)
            if l['kind']=='relu':
                ub = E@x + bmax
                if np.all(ub <= T0[rows]): cert_groups+=1; continue
            cost += len(rows)*N*bd/32    # read the members' residuals (bd bits each)
        if l['kind']=='relu':
            out=rq(acc,l['M'],l['sh']); active+=(out>0).sum(); nxt.append(out)
    n=len(xs)
    print(f'L{li} {M}x{N}: B0 {words_b0:.0f} words/input | envelope+residual {cost/n:.0f} ({100*(1-cost/n/words_b0):.0f}% saved) | groups certified off {100*cert_groups/max(total_groups,1):.0f}% | residual bits/group {np.mean(Dbits):.1f} (min {min(Dbits)} max {max(Dbits)}) | active rows {active/n/M*100 if n else 0:.1f}%')
    tot_b0+=words_b0; tot_new+=cost/n
    if nxt: xs=nxt
print(f'TOTAL: B0 {tot_b0:.0f} -> {tot_new:.0f} words/input ({100*(1-tot_new/tot_b0):.0f}% saved)')
