#!/usr/bin/env python3
"""Exact/Monte-Carlo permutation test on the depth A/B csv, per depth bucket."""
import csv, sys, itertools, statistics as st, random, collections

def perm_p(a, b, iters=200000):
    obs = st.mean(a) - st.mean(b)
    pool = a + b; n = len(a)
    total = len(list(itertools.combinations(range(len(pool)), n)))
    if total <= 250000:                      # exact
        cnt = 0; tot = 0
        for idx in itertools.combinations(range(len(pool)), n):
            s = set(idx)
            x = [pool[i] for i in idx]; y = [pool[i] for i in range(len(pool)) if i not in s]
            tot += 1
            if abs(st.mean(x) - st.mean(y)) >= abs(obs) - 1e-12: cnt += 1
        return obs, cnt / tot, "exact(%d)" % tot
    cnt = 0
    for _ in range(iters):
        random.shuffle(pool)
        if abs(st.mean(pool[:n]) - st.mean(pool[n:])) >= abs(obs) - 1e-12: cnt += 1
    return obs, (cnt + 1) / (iters + 1), "mc(%d)" % iters

rows = list(csv.DictReader(open(sys.argv[1])))
def bucket(d):                                # group depths within 5%
    d = int(d)
    for b in (0, 60000, 120000):
        if abs(d - b) <= max(3000, b * 0.05): return b
    return d
data = collections.defaultdict(lambda: collections.defaultdict(list))
acc  = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    data[bucket(r["depth"])][r["cfg"]].append(float(r["tg"]))
    acc[bucket(r["depth"])][r["cfg"]].append(float(r["acc"]))

for d in sorted(data):
    print(f"\n=== depth ~{d:,} tokens ===")
    cfgs = sorted(data[d])
    for c in cfgs:
        v = data[d][c]; a = acc[d][c]
        print(f"  {c:<26} tg mean {st.mean(v):7.2f}  sd {st.pstdev(v):5.2f}  n={len(v):2d}"
              f"   range [{min(v):.2f},{max(v):.2f}]   acc {st.mean(a):.3f}")
    base = "mtp_n3_q8draft"
    if base in data[d]:
        for c in cfgs:
            if c == base: continue
            A, B = data[d][c], data[d][base]
            if len(A) < 3 or len(B) < 3: continue
            diff, p, how = perm_p(list(A), list(B))
            pct = 100 * diff / st.mean(B)
            print(f"  -> {c} vs {base}: {diff:+.2f} t/s ({pct:+.1f}%)  p={p:.5f}  {how}")
