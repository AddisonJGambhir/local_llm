import os, json, urllib.request, random, sys, time
# needle.py PORT NWORDS : OTTER-7712 hidden at each POS fraction (env POS, default 0.1,0.5,0.9); model name from env MODEL (default local)
port, n = sys.argv[1], int(sys.argv[2]); random.seed(7)
w = "the river city market morning train window garden letter winter music engine paper bridge forest".split()
f = lambda k: " ".join(random.choice(w) for _ in range(k))
res = []
for frac in [float(x) for x in os.environ.get("POS", "0.1,0.5,0.9").split(",")]:
    doc = f(int(n * frac)) + " The secret code for the vault is OTTER-7712. " + f(n - int(n * frac))
    msg = [{"role": "user", "content": doc + "\n\nWhat is the secret code for the vault? Reply with only the code."}]
    t = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": os.environ.get("MODEL", "local"), "messages": msg, "max_tokens": 3000,
                             "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode(),
            headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=5400)); m = d["choices"][0]["message"]
        txt = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
        res.append(f"@{frac}: prompt={d['usage']['prompt_tokens']} found={'OTTER-7712' in txt} {time.time()-t:.0f}s")
    except Exception as e:
        res.append(f"@{frac}: FAILED {str(e)[:100]}")
print(" | ".join(res))
