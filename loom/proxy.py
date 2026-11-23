#!/usr/bin/env python3
"""Expose the Loom model as `local` for Pi, Hermes, and other clients.

  - POST bodies: "model" is rewritten to the served tag.
  - GET /v1/models: also lists "local".
  - GET /props: llama.cpp-style {"default_generation_settings": {"n_ctx": N}} for hermes-sync-ctx.
  - LOOM_VISION=1 advertises image support and preserves multimodal content parts.
  - everything else (incl. SSE streams) is passed through unchanged, flushed as it arrives.
Logs one line per request to stdout: path, status, TTFT, total time, bytes.

usage: proxy.py LISTEN_PORT BACKEND_PORT SERVED_TAG
"""
import http.client
import os
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, BACKEND, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
VISION = os.environ.get("LOOM_VISION") == "1"
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "upgrade", "te", "trailers"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# Approved sampling (BetterBench config/qwen-thinking-xtx.json). Loom's own default is temperature 0 = greedy,
# which makes Inkling loop in its reasoning; clients like Pi send no sampling fields, so fill them in here.
# Temperature 0 (greedy) is never passed through.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def _apply_sampling(j):
    changed = []
    for k, v in SAMPLING.items():
        if j.get(k) is None:
            j[k] = v
            changed.append(k)
    t = j.get("temperature")
    if not isinstance(t, (int, float)) or t <= 0:
        j["temperature"] = SAMPLING["temperature"]
        changed.append(f"temperature {t!r}->1.0")
    return changed


def _text(content):
    """Loom takes plain-string content only: join text parts, drop images/other parts."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                out.append(part.get("text") or "")
        return "\n".join(out)
    return str(content)


def normalize_messages(msgs):
    """Normalize message roles and flatten content for text-only packages."""
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            m["role"] = "system"
        elif role == "function":
            m["role"] = "tool"
        if "content" in m:
            if not VISION:
                m["content"] = _text(m["content"])
            if m["content"] is None and m.get("role") != "assistant":
                m["content"] = ""
        out.append(m)
    if VISION and os.environ.get("LOOM_KEEP_OLD_IMAGES") != "1":
        out = _drop_old_images(out)
    return out


def _is_image_part(p):
    return isinstance(p, dict) and (p.get("type") in ("image_url", "input_image", "image") or "image_url" in p)


def _drop_old_images(msgs):
    """Keep image parts only after the last assistant message (the current step).

    Loom never reuses its prefix cache for a conversation that contains an image (cached_tokens stays at 2048), so a
    screenshot left in the history makes every later turn re-prefill the whole conversation (243K: ~2 min per turn).
    Older images become a short text note; the model already saw them and its reply describing them stays (2026-11-23).
    """
    last_assistant = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content")
        if i < last_assistant and isinstance(c, list) and any(_is_image_part(p) for p in c):
            m = dict(m)
            parts = [p if not _is_image_part(p) else {"type": "text", "text": "[image shown earlier]"} for p in c]
            if all(isinstance(p, dict) and p.get("type") == "text" for p in parts):
                m["content"] = "\n".join(p.get("text", "") for p in parts)
            else:
                m["content"] = parts
        out.append(m)
    return out


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _backend(self, method, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=3600)
        c.request(method, self.path, body=body, headers=headers or {})
        return c, c.getresponse()

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _n_ctx(self):
        import os
        if os.environ.get("LOOM_NCTX"):
            return int(os.environ["LOOM_NCTX"])
        try:
            c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=5)
            c.request("GET", "/stats")
            s = json.loads(c.getresponse().read())
            return int((s.get("loaded_config") or {}).get("max_seq") or 0)
        except Exception:
            return 0

    def _flush_history(self):
        try:
            fb = json.dumps({"model": TAG, "messages": [{"role": "user", "content": "."}], "max_tokens": 1,
                             "temperature": 1.0, "stream": False}).encode()
            fc = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=120)
            fc.request("POST", "/v1/chat/completions", fb, {"Content-Type": "application/json", "Content-Length": str(len(fb))})
            fc.getresponse().read(); fc.close()
        except Exception as e:
            log(f"history flush failed: {e}")

    def _relay(self, method):
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        model_in = None
        stream = False
        strip_usage = False   # we asked for usage the client did not: drop that chunk again
        if body and "json" in (self.headers.get("Content-Type") or "json"):
            try:
                j = json.loads(body)
                if isinstance(j, dict) and isinstance(j.get("messages"), list):
                    j["messages"] = normalize_messages(j["messages"])
                    _apply_sampling(j)
                if isinstance(j, dict) and "model" in j:
                    model_in = j["model"]
                    j["model"] = TAG
                    stream = bool(j.get("stream"))
                    if stream and self.path.startswith("/v1/") and "completions" in self.path:
                        so = j.get("stream_options") or {}
                        if not so.get("include_usage"):
                            so["include_usage"] = True
                            j["stream_options"] = so
                            strip_usage = True
                    body = json.dumps(j).encode()
            except Exception:
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            c, r = self._backend(method, body, headers)
        except Exception as e:
            self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
            log(f"{method} {self.path} 502 backend down")
            return
        # Loom keeps the previous conversation resident until the next request replaces it. An image turn (rendered
        # with the generic template, so no prefix reuse) or a new long conversation can then need two copies and is
        # refused. Replace the held history with a 1-token throwaway request and retry once (2026-11-23).
        if r.status == 503 and body and method == "POST":
            err503 = r.read()
            if any(k in err503 for k in (b"retained state leaves insufficient completion capacity", b"exceeds the serving memory limit")):
                c.close()
                self._flush_history()
                log(f"freed held history after: {err503[:90]!r} - retrying once")
                try:
                    c, r = self._backend(method, body, headers)
                except Exception as e:
                    self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                    return
                if r.status == 503:   # still over the memory limit with one copy: shrink the reply reservation once
                    err2 = r.read()
                    try:
                        j3 = json.loads(body)
                    except Exception:
                        j3 = None
                    if b"exceeds the serving memory limit" in err2 and isinstance(j3, dict):
                        for k in ("max_tokens", "max_completion_tokens"):
                            if k in j3:
                                j3[k] = min(int(j3[k]), 2048)
                        if "max_tokens" not in j3 and "max_completion_tokens" not in j3:
                            j3["max_tokens"] = 2048
                        body = json.dumps(j3).encode(); headers["Content-Length"] = str(len(body))
                        c.close(); self._flush_history()
                        log("still over the memory limit - retrying with max_tokens 2048")
                        try:
                            c, r = self._backend(method, body, headers)
                        except Exception as e:
                            self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                            return
                    else:
                        self.send_response(503); self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err2))); self.end_headers(); self.wfile.write(err2)
                        c.close(); log(f"{method} {self.path} 503 after retry {err2[:120]!r}"); return
            else:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(err503)))
                self.end_headers()
                self.wfile.write(err503)
                c.close()
                log(f"{method} {self.path} {r.status} {err503[:160]!r}")
                return
        # Loom rejects input + max_tokens > context (clients like Pi ask for ~110K). Retry once with max_tokens clamped.
        if r.status == 400 and body:
            err = r.read()
            m = re.search(rb"(\d+) input tokens", err) if b"OUT_OF_RANGE" in err else None
            nctx = self._n_ctx()
            if m and nctx:
                room = nctx - int(m.group(1)) - 64
                try:
                    j2 = json.loads(body)
                except Exception:
                    j2 = None
                if room > 0 and isinstance(j2, dict):
                    for k in ("max_tokens", "max_completion_tokens"):
                        if k in j2:
                            j2[k] = min(int(j2[k]), room)
                    if "max_tokens" not in j2 and "max_completion_tokens" not in j2:
                        j2["max_tokens"] = room
                    body = json.dumps(j2).encode()
                    headers["Content-Length"] = str(len(body))
                    c.close()
                    log(f"clamped max_tokens to {room} ({m.group(1).decode()} input tokens, {nctx} ctx)")
                    try:
                        c, r = self._backend(method, body, headers)
                    except Exception as e:
                        self._send_json(502, {"error": {"message": f"loom backend not reachable: {e}", "type": "server_error"}})
                        return
                    err = None
            if err is not None:
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                c.close()
                log(f"{method} {self.path} {r.status} {err[:160]!r}")
                return
        if method == "GET" and self.path.rstrip("/") == "/v1/models" and r.status == 200:
            data = json.loads(r.read())
            ids = [m.get("id") for m in data.get("data", [])]
            if "local" not in ids and data.get("data"):
                alias = dict(data["data"][0]); alias["id"] = "local"
                data["data"].insert(0, alias)
            # Pi's pi-llama-cpp reads meta.n_ctx (llama-server's field) and falls back to 128K without it.
            for m in data.get("data", []):
                n = m.get("context_length") or self._n_ctx()
                if n:
                    m.setdefault("meta", {})["n_ctx"] = n
                m.setdefault("architecture", {})["input_modalities"] = ["text", "image"] if VISION else ["text"]
            c.close()
            self._send_json(200, data)
            return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        first = None; first_tok = None; total = 0; usage = None
        sse = "text/event-stream" in (r.getheader("Content-Type") or "")
        buf = b""; raw = b"" if not sse else None
        def note_event(ev):   # inspect one SSE event; return False to drop it
            nonlocal first_tok, usage
            for line in ev.split(b"\n"):
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if first_tok is None and any((c.get("delta") or {}).get(k) for c in ch for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    first_tok = time.time()
                if strip_usage and d.get("usage") and not ch:
                    return False
            return True
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            first = first or time.time()
            total += len(chunk)
            out = chunk
            if sse:
                buf += chunk
                parts = buf.split(b"\n\n")
                buf = parts.pop()
                out = b"".join(p + b"\n\n" for p in parts if note_event(p))
            elif raw is not None and len(raw) < 4 << 20:
                raw += chunk
            try:
                if out:
                    self.wfile.write(out); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                log(f"{method} {self.path} client disconnected")
                break
        if sse and buf:
            try:
                if note_event(buf):
                    self.wfile.write(buf); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if raw:
            try:
                usage = json.loads(raw).get("usage") or usage
            except Exception:
                pass
        c.close()
        self.close_connection = True
        if self.path.startswith("/v1/"):
            ttft = first_tok or first
            pt = ct = gt = 0
            if usage:
                pt = int(usage.get("prompt_tokens") or 0); gt = int(usage.get("completion_tokens") or 0)
                ct = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            log(f"{method} {self.path} {r.status} model={model_in!r} prompt={pt} cached={ct} gen={gt} "
                f"first-token={(ttft - t0):.2f}s total={time.time() - t0:.1f}s {total}B" if ttft else
                f"{method} {self.path} {r.status} model={model_in!r} prompt={pt} cached={ct} gen={gt} first-token=- total={time.time() - t0:.1f}s {total}B")

    def do_GET(self):
        if self.path.split("?", 1)[0].rstrip("/") == "/props":   # Pi asks /props?model=local&autoload=false
            n = self._n_ctx(); self._send_json(200, {"default_generation_settings": {"n_ctx": n}, "n_ctx": n, "modalities": {"vision": VISION, "audio": False}, "model_alias": "local", "served_tag": TAG})
            return
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def do_DELETE(self):
        self._relay("DELETE")

    def do_OPTIONS(self):
        self._relay("OPTIONS")


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", LISTEN), Proxy)
    log(f"proxy :{LISTEN} -> loom :{BACKEND} ({TAG}), model 'local' accepted")
    srv.serve_forever()
