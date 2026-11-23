import json,urllib.request,random,time,sys
# loom-cap-probe.py SIZES... : descending prompt sizes (words) with max_tokens 8192; stops at first OK; reports model health after each
random.seed(4); w="the river city market morning train window garden letter winter music engine paper bridge forest".split()
for n in map(int, sys.argv[1:]):
    doc=" ".join(random.choice(w) for _ in range(n)); t=time.time()
    req=urllib.request.Request("http://127.0.0.1:1239/v1/chat/completions",json.dumps({"model":"local","messages":[{"role":"user","content":doc+"\n\nSay OK."}],"max_tokens":8192,"temperature":1.0,"top_p":0.95,"top_k":20}).encode(),{"Content-Type":"application/json"})
    try: d=json.load(urllib.request.urlopen(req,timeout=1800)); r=f"OK prompt {d['usage']['prompt_tokens']} {time.time()-t:.0f}s"
    except urllib.error.HTTPError: r=f"refused {time.time()-t:.0f}s"
    s=json.load(urllib.request.urlopen("http://127.0.0.1:11438/v1/status"))["data"][0]["loom"]
    print(n, r, "| model:", s.get("registration"), flush=True)
    if r.startswith("OK") or s.get("registration")!="ready": break
