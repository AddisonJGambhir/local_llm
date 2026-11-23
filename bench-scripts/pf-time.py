import json,urllib.request,time,random,sys
random.seed(int(sys.argv[2])); w="the river city market morning train window garden letter winter music".split()
doc=" ".join(random.choice(w) for _ in range(200000)); t=time.time()
r=urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions",json.dumps({"model":"local","messages":[{"role":"user","content":doc+"\nSay OK."}],"max_tokens":16,"temperature":1.0,"top_p":0.95,"top_k":20}).encode(),{"Content-Type":"application/json"})
d=json.load(urllib.request.urlopen(r,timeout=1800)); dt=time.time()-t; n=d["usage"]["prompt_tokens"]; print(f"prompt={n} {dt:.0f}s -> {n/dt:.0f} tok/s")
