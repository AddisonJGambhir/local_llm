#!/usr/bin/env python3
"""Record attempt under the EXACT conditions the 81.67 t/s number was set in:
-c 98304, tensor 50/50, f16 KV, ub 1024, matrix.SHORT_PROMPT (= make_prompt(2048)),
2048-token generation, temp 0 + top_k 1 + ignore_eos."""
import os, json, sys, time, urllib.request, statistics as st
sys.path.insert(0, "/home/addison-gambhir/Desktop/local_llm/llama_server/src/benchmarks")
from matrix import SHORT_PROMPT          # the identical prompt object
_pf = os.environ.get("PROMPT_FILE")
if _pf:
    SHORT_PROMPT = open(_pf).read()

BASE = "http://127.0.0.1:1234"
def post(path, obj, timeout=1800):
    r = urllib.request.Request(BASE+path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def run(label, reps, outcsv):
    if not os.path.exists(outcsv):
        open(outcsv, "w").write("cfg,rep,tg,pp,draft_n,draft_acc_n,acc,mean_acc_len\n")
    tgs = []
    for r in range(reps):
        t = post("/completion", {"prompt": SHORT_PROMPT, "n_predict": 2048,
                                 "temperature": 0, "top_k": 1, "ignore_eos": True,
                                 "cache_prompt": False})["timings"]
        dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
        acc = (da/dn) if dn else 0.0
        mal = t.get("draft_n_accepted", 0)
        tg = t["predicted_per_second"]; pp = t.get("prompt_per_second", 0)
        tgs.append(tg)
        open(outcsv, "a").write(f"{label},{r+1},{tg:.3f},{pp:.2f},{dn},{da},{acc:.4f},{mal}\n")
        print(f"[{label}] rep{r+1}: tg={tg:.2f} pp={pp:.1f} acc={acc:.4f} ({da}/{dn})", flush=True)
    print(f"[{label}] SUMMARY mean={st.mean(tgs):.2f} sd={st.pstdev(tgs):.2f} "
          f"max={max(tgs):.2f} n={len(tgs)}", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
