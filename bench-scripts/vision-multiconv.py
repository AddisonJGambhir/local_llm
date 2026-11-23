import base64, json, sys, time, urllib.request, random
# N distinct long conversations (Pi main + subagents), then an image on the first one
port, img, n, words = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
w = "the river city market morning train window garden letter winter music engine paper bridge forest".split()
def post(msgs, mt):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"model": "local", "messages": msgs,
        "max_tokens": mt, "temperature": 1.0, "top_p": 0.95, "top_k": 20}).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=1800)); return d, time.time() - t
convs = []
for c in range(n):
    random.seed(100 + c); doc = f"Conversation {c}. " + " ".join(random.choice(w) for _ in range(words))
    h = [{"role": "user", "content": doc + "\n\nSay OK."}]
    try:
        d, t = post(h, 16); print(f"conv {c}: prompt={d['usage']['prompt_tokens']} {t:.0f}s", flush=True)
    except Exception as e:
        print(f"conv {c} FAILED: {str(e)[:150]}", flush=True); break
    convs.append(h + [{"role": "assistant", "content": d["choices"][0]["message"].get("content") or "OK"}])
h = convs[0] + [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(img, "rb").read()).decode()}}]}]
try:
    d, t = post(h, 1500); print(f"image turn {t:.0f}s -> {(d['choices'][0]['message'].get('content') or '')[:120]!r}")
except Exception as e:
    print("image turn FAILED:", str(e)[:200])
