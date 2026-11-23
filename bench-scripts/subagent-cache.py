import json,urllib.request,time,random,sys
port=sys.argv[1]; w="the river city market morning train window garden letter winter music".split()
def post(m,mt=200):
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps({"model":"local","messages":m,"max_tokens":mt,"temperature":1.0,"top_p":0.95,"top_k":20,"reasoning_effort":"low"}).encode(),{"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=1800)); return d,time.time()-t
def asst(d):
    mm=d["choices"][0]["message"]; a={"role":"assistant","content":mm.get("content") or ""}
    if mm.get("reasoning_content"): a["reasoning_content"]=mm["reasoning_content"]
    return a
random.seed(31); main=[{"role":"system","content":"Be brief."},{"role":"user","content":" ".join(random.choice(w) for _ in range(int(sys.argv[2]) if len(sys.argv)>2 else 100000))+"\nSay OK."}]
d,t=post(main); main.append(asst(d)); print(f"main 1: prompt={d['usage']['prompt_tokens']} {t:.0f}s",flush=True)
main.append({"role":"user","content":"Say OK again."}); d,t=post(main); main.append(asst(d)); print(f"main 2: cached={d['usage']['prompt_tokens_details']['cached_tokens']} {t:.0f}s",flush=True)
random.seed(32); sub=[{"role":"user","content":" ".join(random.choice(w) for _ in range(30000))+"\nSubagent: say OK."}]
d,t=post(sub); print(f"subagent: prompt={d['usage']['prompt_tokens']} {t:.0f}s",flush=True)
main.append({"role":"user","content":"Back to main: say OK."}); d,t=post(main); print(f"main 3 after subagent: prompt={d['usage']['prompt_tokens']} cached={d['usage']['prompt_tokens_details']['cached_tokens']} {t:.0f}s",flush=True)
