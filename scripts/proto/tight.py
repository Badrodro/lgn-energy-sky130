import numpy as np, json, sys
sys.argv=[sys.argv[0]]
import dtn_exp as E
for arch,lam in [([[24,4],[12,4]],0.03),([[40,4],[20,4]],0.01)]:
    rng=np.random.default_rng(0)
    Xtr,ytr,Xte,yte=E.load('digits'); Xf=(Xtr/255.0).astype(np.float32)
    layers=[]; n=64
    for h in arch: layers.append(E.Tree(rng,n,h[0],h[1])); n=h[0]*(1+h[1])
    layers.append(E.Dense(rng,n,10))
    E.train(rng,layers,Xf,ytr,150,lam=lam)
    q=E.quantize(layers,Xtr)
    H=Xte
    for li,l in enumerate(q[:-1]):
        acc=H@l['W'].T+l['b']; out=E.requant(acc,l['M'],l['sh']); t=l['tree']; R,C=t['R'],t['C']
        roots=out[:,:R]>0; kids=(out[:,R:].reshape(len(H),R,C)>0)
        p_root=roots.mean(); p_any=kids.any(2).mean(); q_leaf=kids.mean()
        print(f"{arch} L{li}: q(child active)={q_leaf:.3f}  P(root active)={p_root:.3f}  P(some child active)={p_any:.3f}  tightness={p_any/max(p_root,1e-9):.2f}  C*q={C*q_leaf:.3f}  nb={max(1,int(t['D'].max()).bit_length())}")
        H=out
