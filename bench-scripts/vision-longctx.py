import base64, json, sys, time, urllib.request, random
# vision-longctx.py PORT IMAGE NWORDS : long text turn, then the same conversation + an image (what Pi does)
port, img, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); random.seed(3)
w = "the river city market morning train window garden letter winter music engine paper bridge forest".split()
doc = " ".join(random.choice(w) for _ in range(n))
def post(msgs, mt):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"model": "local", "messages": msgs,
        "max_tokens": mt if mt < 1000 else 8192, "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode(), {"Content-Type": "application/json", **({"X-Loom-Session": __import__("os").environ["SESSION"]} if __import__("os").environ.get("SESSION") else {})})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); return d, time.time() - t
hist = [{"role": "user", "content": doc + "\n\nSay OK."}]
d, t = post(hist, 16); print(f"text turn: prompt={d['usage']['prompt_tokens']} {t:.0f}s", flush=True)
hist += [{"role": "assistant", "content": d["choices"][0]["message"].get("content") or "OK"},
         {"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(img, "rb").read()).decode()}}]}]
try:
    d, t = post(hist, 1500); print(f"image turn: prompt={d["usage"]["prompt_tokens"]} cached={(d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")} {t:.0f}s -> {(d['choices'][0]['message'].get('content') or '')[:160]!r}")
except Exception as e:
    print("image turn FAILED:", str(e)[:200], getattr(e, 'read', lambda: b'')()[:300])
