import base64, json, sys, time, urllib.request, random
# mimic the failing Pi session: ~35K context, a ~9K-token reply, more turns, then an image
port, img = sys.argv[1], sys.argv[2]; random.seed(5)
w = "the river city market morning train window garden letter winter music engine paper bridge forest".split()
doc = " ".join(random.choice(w) for _ in range(35000))
def post(msgs, mt, stream=False):
    body = {"model": "local", "messages": msgs, "max_tokens": mt, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stream": stream}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); r = urllib.request.urlopen(req, timeout=1800)
    if stream:
        out = "".join((json.loads(l[5:])["choices"][0]["delta"].get("content") or "") for l in r if l.startswith(b"data:") and l.strip() != b"data: [DONE]" and json.loads(l[5:]).get("choices"))
        return out, time.time() - t
    d = json.load(r); return d["choices"][0]["message"].get("content") or "", time.time() - t
hist = [{"role": "user", "content": doc + "\n\nWrite a very long, detailed story (at least 6000 words) about the river city."}]
out, t = post(hist, 9500, stream=True); print(f"long reply: {len(out.split())} words {t:.0f}s", flush=True)
hist += [{"role": "assistant", "content": out}]
for i in range(3):
    hist += [{"role": "user", "content": f"Summarize part {i+1} of that in two sentences."}]
    o, t = post(hist, 300, stream=True); hist += [{"role": "assistant", "content": o}]; print(f"turn {i+1}: {t:.0f}s", flush=True)
hist += [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(img, "rb").read()).decode()}}]}]
try:
    o, t = post(hist, 1500, stream=True); print(f"image turn {t:.0f}s -> {o[:150]!r}")
except Exception as e:
    print("image turn FAILED:", str(e)[:200])
