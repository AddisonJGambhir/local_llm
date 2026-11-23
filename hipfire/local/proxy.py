#!/usr/bin/env python3
"""hipfire :1234 shim - lets harnesses that ask for model "local" (Pi, Hermes, Qwen Code...)
talk to `hipfire serve`, which only answers to its real tag.

  - POST bodies: "model" is rewritten to the served tag.
  - GET /v1/models: also lists "local".
  - GET /props: llama.cpp-style {"default_generation_settings": {"n_ctx": N}} for hermes-sync-ctx.
  - everything else (incl. SSE streams) is passed through unchanged, flushed as it arrives.
Logs one line per request to stdout: path, status, TTFT, total time, bytes.

usage: proxy.py LISTEN_PORT BACKEND_PORT SERVED_TAG
"""
import http.client
import os
VISION = os.environ.get("PROXY_VISION", "1") == "1"   # image input works on this engine/mode (hipfire maxctx turns the tower off)
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, BACKEND, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "upgrade", "te", "trailers"}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


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
        try:
            c = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=5)
            c.request("GET", "/stats")
            s = json.loads(c.getresponse().read())
            return int((s.get("loaded_config") or {}).get("max_seq") or 0)
        except Exception:
            return 0

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
            self._send_json(502, {"error": {"message": f"hipfire backend not reachable: {e}", "type": "server_error"}})
            log(f"{method} {self.path} 502 backend down")
            return
        if method == "GET" and self.path.rstrip("/") == "/v1/models" and r.status == 200:
            data = json.loads(r.read())
            ids = [m.get("id") for m in data.get("data", [])]
            if "local" not in ids and data.get("data"):
                alias = dict(data["data"][0]); alias["id"] = "local"
                data["data"].insert(0, alias)
            # Pi's pi-llama-cpp reads meta.n_ctx (llama-server's field) and falls back to 128K without it.
            n_ctx = None
            for m in data.get("data", []):
                n = m.get("context_length") or m.get("max_model_len") or n_ctx or self._n_ctx()
                n_ctx = n
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
    log(f"proxy :{LISTEN} -> hipfire :{BACKEND} ({TAG}), model 'local' accepted")
    srv.serve_forever()
