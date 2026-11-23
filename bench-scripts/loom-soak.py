import base64, json, sys, time, urllib.request, random
# Pi-like soak on :1234: one conversation grows ~30K -> ~245K over 12 turns, images at turns 3/7/11, a subagent conversation at turn 9
port = sys.argv[1]; img = sys.argv[2]; random.seed(11)
w = "the river city market morning train window garden letter winter music engine paper bridge forest".split()
chunk = lambda n: " ".join(random.choice(w) for _ in range(n))
b64 = base64.b64encode(open(img, "rb").read()).decode()
def post(msgs, mt=1500):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps({"model": "local", "messages": msgs,
        "max_tokens": mt, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "reasoning_effort": "low"}).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=2400)); return d, time.time() - t
def health():
    s = json.load(urllib.request.urlopen("http://127.0.0.1:11437/v1/status"))["data"][0]["loom"]; return s.get("registration")
hist = [{"role": "system", "content": "You are a coding agent. Be brief."}]
ok = fail = 0
for turn in range(1, 13):
    add = 30000 if turn == 1 else 19000
    if turn in (3, 7, 11):
        content = [{"type": "text", "text": chunk(add) + "\n\nAlso: describe this screenshot in one sentence."},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]
    else:
        content = chunk(add) + "\n\nReply with one short sentence."
    hist.append({"role": "user", "content": content})
    try:
        d, t = post(hist); u = d["usage"]; mm = d["choices"][0]["message"]; msg = mm.get("content") or ""
        am = {"role": "assistant", "content": msg}
        if mm.get("reasoning_content"): am["reasoning_content"] = mm["reasoning_content"]   # Pi resends reasoning
        hist.append(am); ok += 1
        print(f"turn {turn:2d}{' IMG' if turn in (3,7,11) else '    '}: prompt={u['prompt_tokens']:6d} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens')!s:>6} {t:5.0f}s  model={health()}  {msg[:70]!r}", flush=True)
    except Exception as e:
        hist.pop(); fail += 1
        print(f"turn {turn:2d}: FAILED {str(e)[:80]} {getattr(e,'read',lambda:b'')()[:120]}  model={health()}", flush=True)
    if turn == 9:
        sub = [{"role": "user", "content": chunk(30000) + "\n\nSubagent task: reply OK."}]
        try: d, t = post(sub, 200); print(f"  subagent: prompt={d['usage']['prompt_tokens']} {t:.0f}s model={health()}", flush=True)
        except Exception as e: print(f"  subagent FAILED {str(e)[:100]} model={health()}", flush=True)
print(f"DONE ok={ok} fail={fail}")
