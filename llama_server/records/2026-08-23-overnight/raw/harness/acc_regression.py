#!/usr/bin/env python3
"""Where does adaptive overtake n3? Regress tg on MEASURED acceptance.

Prompt mix % turned out to be a poor instrument: what the model generates from a
mixed prompt is itself mixed, so mix% only loosely controls acceptance (both the
concatenated and the interleaved prompt sets came out non-monotone). Measured
acceptance is the real independent variable, and every rep already reports it.

Pools only the -c 98304 / tensor 50/50 rig so the comparison is like-for-like.
"""
import csv, glob, os, statistics as st

SP = os.path.dirname(os.path.abspath(__file__))
FILES = ["record.csv", "maxtg.csv", "ceiling.csv", "crossover.csv", "crossover2.csv"]

def family(cfg):
    c = cfg.lower()
    if "mmvq" in c or "chain" in c: return None          # different binaries
    if "n3" in c or "ceil_shipped_n3" in c: return "n3"
    if "adaptive" in c or "shipped_adaptive" in c: return "adaptive"
    return None

pts = {"n3": [], "adaptive": []}
for f in FILES:
    p = os.path.join(SP, f)
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        fam = family(r["cfg"])
        if not fam: continue
        try: pts[fam].append((float(r["acc"]), float(r["tg"])))
        except (KeyError, ValueError): pass

def fit(xy):
    n=len(xy)
    if n < 3: return None
    mx=st.mean(x for x,_ in xy); my=st.mean(y for _,y in xy)
    sxx=sum((x-mx)**2 for x,_ in xy); sxy=sum((x-mx)*(y-my) for x,y in xy)
    if sxx == 0: return None
    b=sxy/sxx; a=my-b*mx
    ss_t=sum((y-my)**2 for _,y in xy); ss_r=sum((y-(a+b*x))**2 for x,y in xy)
    return a, b, (1-ss_r/ss_t if ss_t else 0), n

print(f"{'config':<10}{'n':>5}{'intercept':>11}{'slope':>10}{'R^2':>8}   tg at acc:")
fits={}
for k in ("n3","adaptive"):
    f=fit(pts[k])
    if not f: print(f"  {k}: too few points"); continue
    a,b,r2,n=f; fits[k]=(a,b)
    preds="  ".join(f"{x:.2f}->{a+b*x:6.1f}" for x in (0.55,0.70,0.85,0.99))
    print(f"{k:<10}{n:>5}{a:>11.2f}{b:>10.2f}{r2:>8.3f}   {preds}")

if len(fits)==2:
    (a1,b1),(a2,b2)=fits["adaptive"],fits["n3"]
    if abs(b1-b2) > 1e-9:
        x=(a2-a1)/(b1-b2)
        print(f"\nCROSSOVER: adaptive overtakes n3 at measured acceptance = {x:.3f}")
        print("  (linear fit; the true relation is convex, so treat as indicative)")
    lo=[p for p in pts['adaptive'] if p[0]<0.70]; hi=[p for p in pts['adaptive'] if p[0]>0.95]
    lo3=[p for p in pts['n3'] if p[0]<0.70];      hi3=[p for p in pts['n3'] if p[0]>0.95]
    if lo and lo3:
        print(f"\n  acc<0.70  adaptive {st.mean(y for _,y in lo):.2f} (n={len(lo)})  "
              f"n3 {st.mean(y for _,y in lo3):.2f} (n={len(lo3)})")
    if hi and hi3:
        print(f"  acc>0.95  adaptive {st.mean(y for _,y in hi):.2f} (n={len(hi)})  "
              f"n3 {st.mean(y for _,y in hi3):.2f} (n={len(hi3)})")
