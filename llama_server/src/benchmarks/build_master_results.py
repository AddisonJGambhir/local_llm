#!/usr/bin/env python3
"""Regenerate records/MASTER_RESULTS.csv from every recorded benchmark source.

Two measurement families are folded into one schema:
  llama-bench  - synthetic pp8192 / tg256, no speculation (output/benchmarks/*/all_runs.csv)
  server       - real prompts through /completion, speculation active (scratchpad runs,
                 transcribed here because the scratchpad is not durable)
`method` distinguishes them. Never compare a llama-bench tg against a server tg:
llama-bench does not run the MTP draft path at all.
"""
import csv, statistics as st, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "records", "MASTER_RESULTS.csv")

COLS = ["date","method","model","quant","backend","build","split_mode","ratio","ctx",
        "kv","spec_type","n_max","n_min_ad","content","reps",
        "pp_mean","pp_sd","tg_mean","tg_sd","peak_xtx_mib","source","notes"]

rows = []

def add(**kw):
    r = {c: kw.get(c, "") for c in COLS}
    rows.append(r)

def agg(vals):
    if not vals: return "", ""
    return (round(st.mean(vals), 2), round(st.stdev(vals), 2) if len(vals) > 1 else 0.0)

# ---------------------------------------------------------------- server runs, 2026-08-22
Q8, M = "Q8_0", "Qwen3.8-27B"
PRB  = "13ac90156-pr25051"          # PR #25051, tensor split works
TRB  = "7c80213c3-rebased"          # #25051 rebased on master b21e4de74 + #27210
MRG  = "merged-tp-adaptive"         # #25051 + #27210 ported (this session)
SRC  = "records/2026-08-22-adaptive-mtp/"

def srv(build, split, ratio, spec, content, tg, pp=None, peak="", nmax="", nmin="", notes="", ctx=98304):
    tgm, tgs = agg(tg); ppm, pps = agg(pp or [])
    add(date="2026-08-22", method="server", model=M, quant=Q8, backend="vulkan",
        build=build, split_mode=split, ratio=ratio, ctx=ctx, kv="f16",
        spec_type=spec, n_max=nmax, n_min_ad=nmin, content=content, reps=len(tg),
        pp_mean=ppm, pp_sd=pps, tg_mean=tgm, tg_sd=tgs, peak_xtx_mib=peak,
        source=SRC, notes=notes)

# A/B sweep (4 configs) + repro
srv(PRB, "tensor", "50/50", "draft-mtp", "agentic", [56.06,58.87,56.53,56.77], peak=19674, nmax=3,
    notes="production config")
srv(PRB, "tensor", "50/50", "draft-mtp", "novel",   [50.74,36.06,45.71,46.84], peak=19674, nmax=3)
srv(TRB, "layer",  "28/37", "draft-mtp", "agentic", [52.39,50.90,54.54,54.26, 53.47,51.48,53.58],
    pp=[810.59,834.18,844.00], peak=17575, nmax=3, notes="A/B + repro pooled")
srv(TRB, "layer",  "28/37", "draft-mtp", "novel",   [33.18,34.17,42.90,33.33], peak=17575, nmax=3)
srv(TRB, "layer",  "28/37", "draft-mtp-adaptive", "agentic",
    [64.58,78.80,74.34,78.62, 63.97,77.89,74.21], pp=[820.05,831.02,833.42], peak=17866,
    nmax=12, nmin=3, notes="+38.3% vs n3, permutation p=0.00058")
srv(TRB, "layer",  "28/37", "draft-mtp-adaptive", "novel",
    [79.67,77.88,79.98,53.87], peak=17866, nmax=12, nmin=3, notes="not re-tested; treat as provisional")
srv(PRB, "layer",  "28/37", "draft-mtp", "agentic", [53.28,51.27,54.15,54.33], peak=17556, nmax=3,
    notes="equals rebased build -> rebase gains nothing")
srv(PRB, "layer",  "28/37", "draft-mtp", "novel",   [33.23,34.36,43.00,33.18], peak=17556, nmax=3)

# merged build (this session): #25051 tensor split + #27210 adaptive MTP ported onto it
srv(MRG, "tensor", "50/50", "draft-mtp", "agentic", [56.08,58.57,56.32],
    pp=[651.2,627.2,587.6], peak=19652, nmax=3, notes="merge does not regress n3")
srv(MRG, "tensor", "50/50", "draft-mtp-adaptive", "agentic", [86.95,71.33,86.72],
    pp=[644.2,629.9,593.9], peak=20388, nmax=12, nmin=3,
    notes="BEST: +43.3% vs tensor n3, prefill unchanged, +736 MiB XTX")

# 256k f16 sizing on the merged build (text-only, adaptive), real work under load
srv(MRG, "tensor", "49/51", "draft-mtp-adaptive", "synthetic-4.9k", [50.86],
    pp=[809.0], peak=23521, nmax=12, nmin=3, ctx=262144, notes="SHIPPED preset; headroom 679 MiB")
srv(MRG, "tensor", "45/55", "draft-mtp-adaptive", "synthetic-4.9k", [49.14],
    pp=[809.8], peak=22820, nmax=12, nmin=3, ctx=262144, notes="fallback ratio; headroom 1380 MiB")

# ---------------------------------------------------------------- llama-bench history
src = os.path.join(ROOT, "output", "benchmarks", "2026-08-21-pr25051", "all_runs.csv")
if os.path.exists(src):
    with open(src) as f:
        for r in csv.DictReader(f):
            if r.get("error"): continue
            add(date="2026-08-21", method="llama-bench", model=M, quant=r["quant"],
                backend=r["backend"], build=r["build"], split_mode=r["split"],
                ratio=r["ratio"], ctx="", kv="f16", spec_type="none",
                content=f"synthetic d{r['depth']}", reps=r["reps"],
                pp_mean=r["pp8192_mean"], pp_sd=r["pp8192_sd"],
                tg_mean=r["tg256_mean"], tg_sd=r["tg256_sd"],
                peak_xtx_mib=r["peak_xtx_mib"],
                source="output/benchmarks/2026-08-21-pr25051/all_runs.csv",
                notes="no speculation - tg not comparable to server rows")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}: {len(rows)} rows")
