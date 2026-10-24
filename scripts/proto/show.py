import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: print(l.strip()[:300]); continue
    e=r['et']
    print(r['cfg'], f"acc {r['acc']:.4f} | tree-skip only {r['total']:.0f} | +ET+zw {e['total']:.0f} / B0 {e['total_b0']:.0f} ({100*(1-e['total']/e['total_b0']):.0f}% saved) per-layer {e['words']} viol {e['dominance_violations']}")
