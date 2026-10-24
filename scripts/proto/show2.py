import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: print(l.strip()[:300]); continue
    c=r['cfg']
    print(f"{c['data']:6s} {c['kind']:5s} {str(c['arch']):18s} lam={c.get('lam',0):<6} seed={c.get('seed',0)} acc={r['acc']:.4f} words={r['total']:7.0f} B0={r['b0']:6.0f} ({100*(1-r['total']/r['b0']):4.1f}% saved) layers={r['words']} active={r['active']} viol={r['viol']} {r['secs']}s")
