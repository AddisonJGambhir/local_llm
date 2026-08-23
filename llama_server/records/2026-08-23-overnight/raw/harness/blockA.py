#!/usr/bin/env python3
"""Depth A/B driver. Talks to an already-running llama-server on :1234.
Diverse (non-repetitive) content built from llama.cpp sources, sized by /tokenize."""
import json, sys, time, urllib.request, glob, os, statistics

BASE="http://127.0.0.1:1234"
def post(path, obj, timeout=1800):
    r=urllib.request.Request(BASE+path, data=json.dumps(obj).encode(),
                             headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

def corpus():
    """Diverse content: real source files, deduped, order-shuffled deterministically."""
    pats=["/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-adaptive/src/*.cpp",
          "/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-adaptive/src/models/*.cpp",
          "/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-adaptive/common/*.cpp",
          "/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-adaptive/tools/server/*.cpp",
          "/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-adaptive/ggml/src/*.c"]
    files=sorted({f for p in pats for f in glob.glob(p)})
    out=[]
    for f in files:
        try: out.append("// ==== %s ====\n%s"%(os.path.basename(f), open(f,encoding="utf-8",errors="ignore").read()))
        except Exception: pass
    return "\n\n".join(out)

def ntok(s):
    return len(post("/tokenize", {"content": s})["tokens"])

def sized(text, target):
    """binary-search a char count that tokenizes to ~target tokens (within 1%)"""
    lo, hi = 0, len(text)
    if ntok(text) < target:
        text = text * (target // max(1, ntok(text)) + 2); hi = len(text)
    best=text[:hi]
    for _ in range(18):
        mid=(lo+hi)//2
        n=ntok(text[:mid])
        if abs(n-target) <= target*0.01: return text[:mid], n
        if n < target: lo=mid
        else: hi=mid; best=text[:mid]
    return best, ntok(best)

def run(label, depths, reps, outcsv):
    full=corpus()
    rows=[]
    for d in depths:
        if d==-1:
            # REPETITIVE regime: the exact content class the 81.67 t/s record was set on
            prompt=("The system processes requests through a pipeline. "*420)[:24000]; nt=ntok(prompt)
        elif d==0:
            prompt="Continue writing technical documentation about GPU memory hierarchies.\n\n"; nt=ntok(prompt)
        else:
            prompt, nt = sized(full, d)
            prompt += "\n\n// Continue the analysis above in detail.\n"
            nt = ntok(prompt)
        print(f"[{label}] depth target {d} -> actual {nt} tokens", flush=True)
        # warm: encode the prefix once
        t0=time.time()
        post("/completion", {"prompt":prompt,"n_predict":1,"temperature":0,"top_k":1,
                             "cache_prompt":True})
        print(f"[{label}] d{d} warm encode {time.time()-t0:.1f}s", flush=True)
        for r in range(reps):
            req={"prompt":prompt,"n_predict":2048,"ignore_eos":True,"cache_prompt":True}
            if os.environ.get("IRL_SAMPLER"):
                # the production profile's sampler chain, not the bench one
                req.update({"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,
                            "presence_penalty":0.0,"repeat_penalty":1.0,"seed":1234})
            else:
                req.update({"temperature":0,"top_k":1})
            t=post("/completion", req)["timings"]
            dn=t.get("draft_n",0); da=t.get("draft_n_accepted",0)
            acc=(da/dn) if dn else 0.0
            row=dict(cfg=label, depth=nt, rep=r+1,
                     tg=round(t["predicted_per_second"],3),
                     pp=round(t.get("prompt_per_second",0),2),
                     draft_n=dn, draft_acc_n=da, acc=round(acc,4))
            rows.append(row)
            print(f"[{label}] d{nt} rep{r+1}: tg={row['tg']} acc={row['acc']} ({da}/{dn})", flush=True)
            with open(outcsv,"a") as f:
                f.write("%s,%d,%d,%.3f,%.2f,%d,%d,%.4f\n"%(label,nt,r+1,row["tg"],row["pp"],dn,da,acc))
        tgs=[x["tg"] for x in rows if x["depth"]==nt]
        print(f"[{label}] d{nt} SUMMARY mean={statistics.mean(tgs):.2f} sd={statistics.pstdev(tgs):.2f} n={len(tgs)}", flush=True)
    return rows

if __name__=="__main__":
    label=sys.argv[1]; depths=[int(x) for x in sys.argv[2].split(",")]
    reps=int(sys.argv[3]); outcsv=sys.argv[4]
    if not os.path.exists(outcsv):
        open(outcsv,"w").write("cfg,depth,rep,tg,pp,draft_n,draft_acc_n,acc\n")
    run(label, depths, reps, outcsv)
