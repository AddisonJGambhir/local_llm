#!/usr/bin/env python3
"""Fold a night's raw benchmark CSVs into MASTER_RESULTS.csv.

Why this exists: build_master_results.py hardcodes rows transcribed by hand from
scratchpad runs, with a comment noting the scratchpad is not durable. That is the
same fragility that lost the prompt behind the 81.67 t/s figure -- the number
survived, the thing needed to reproduce it did not. This reads committed raw CSVs
instead, so every row can be traced back to its measurements.

Usage: python3 src/benchmarks/ingest_run.py records/2026-08-23-overnight/raw
"""
import csv, glob, os, sys, statistics as st, collections

RAW = sys.argv[1] if len(sys.argv) > 1 else "records/2026-08-23-overnight/raw"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(RAW))), "SUMMARY.csv")

# rig metadata per source file: (ctx, split, ratio, prompt, note)
RIGS = {
    "depth_ab.csv":   (262144, "tensor", "49/51", "novel source", "depth rig"),
    "record.csv":     ( 98304, "tensor", "50/50", "SHORT_PROMPT", "record rig"),
    "maxtg.csv":      ( 98304, "tensor", "50/50", "agentic_toolcalls", "record rig"),
    "ceiling.csv":    ( 98304, "tensor", "50/50", "repetitive_sentence", "DEGENERATE - upper bound"),
    "crossover.csv":  ( 98304, "tensor", "50/50", "mix_* (concatenated)", "x-axis invalid, cells valid"),
    "crossover2.csv": ( 98304, "tensor", "50/50", "ilv_* (interleaved)", "x-axis invalid, cells valid"),
    "irl_sampler.csv":(262144, "tensor", "49/51", "novel source", "sampler isolation"),
}

rows = []
for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
    base = os.path.basename(f)
    if base == "pp_record.csv":            # different schema (llama-bench)
        for r in csv.DictReader(open(f)):
            if not r.get("pp_mean"): continue
            rows.append(dict(source=base, cfg=f"{r['build']}/{r['split']}", ctx="derived",
                             prompt=f"pp8192 ub{r['ub']} b{r['b']} {r['ctx']}",
                             metric="pp", mean=r["pp_mean"], sd=r["pp_sd"], n="3",
                             acc="", peak=r["peak_xtx_mib"], note="llama-bench, no speculation"))
        continue
    ctx, split, ratio, prompt, note = RIGS.get(base, ("", "", "", "", ""))
    g = collections.defaultdict(list); a = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        key = (r["cfg"], r.get("depth", ""))
        try:
            g[key].append(float(r["tg"])); a[key].append(float(r.get("acc", 0) or 0))
        except (KeyError, ValueError): pass
    for (cfg, depth), v in sorted(g.items()):
        rows.append(dict(source=base, cfg=cfg, ctx=ctx,
                         prompt=f"{prompt}{(' @d'+depth) if depth else ''}",
                         metric="tg", mean=round(st.mean(v), 2),
                         sd=round(st.pstdev(v), 2), n=len(v),
                         acc=round(st.mean(a[(cfg, depth)]), 4) if a[(cfg, depth)] else "",
                         peak="", note=f"{split} {ratio}; {note}"))

cols = ["source","cfg","ctx","prompt","metric","mean","sd","n","acc","peak","note"]
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r.get(c, "") for c in cols})
print(f"wrote {OUT}: {len(rows)} aggregated rows from {len(glob.glob(os.path.join(RAW,'*.csv')))} raw files")
