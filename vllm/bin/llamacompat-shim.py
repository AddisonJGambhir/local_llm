#!/usr/bin/env python3
"""llama.cpp-compat shim in front of vLLM.

WHY: Pi's `pi-llama-cpp` extension (and anything else written against
llama-server) probes two endpoints vLLM does not implement:

  /health   llama.cpp -> {"status":"ok"};  vLLM -> HTTP 200 with an EMPTY body,
            so the extension's res.json() throws and it reports "unreachable".
  /props    llama.cpp -> server/model properties used for role detection and
            n_ctx;  vLLM -> 404.

This shim listens on its own port, answers those two itself, and proxies
everything else (including streaming SSE) straight through. It synthesizes a
response ONLY when the upstream does not provide one, so putting it in front of
a real llama-server is also safe -- that server's own /health and /props win.

Usage: llamacompat-shim.py [--listen 1235] [--upstream 127.0.0.1:1234]
"""
import socket
import argparse
import http.client
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AP = argparse.ArgumentParser()
AP.add_argument("--listen", type=int, default=1235)
AP.add_argument("--upstream", default="127.0.0.1:1234")
AP.add_argument("--ctx", type=int, default=8192)
ARGS = AP.parse_args()
UP_HOST, UP_PORT = ARGS.upstream.split(":")
UP_PORT = int(UP_PORT)

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade"}

# llama-server answers the OpenAI routes BOTH with and without the /v1 prefix;
# vLLM only mounts them under /v1. Pi's llama-server provider posts to the
# unprefixed form, which vLLM 404s ("Error: 404 status code (no body)").
# Map the unprefixed aliases onto /v1. Deliberately NOT mapped: llama.cpp's
# native /completion, /infill, /tokenize -- different request schemas, so
# silently rerouting them would be worse than a clean 404.
ALIASES = {
    "/chat/completions": "/v1/chat/completions",
    "/completions": "/v1/completions",
    "/embeddings": "/v1/embeddings",
    "/models": "/v1/models",
    "/responses": "/v1/responses",
}


def rewrite(path):
    head, sep, qs = path.partition("?")
    target = ALIASES.get(head)
    return (target + sep + qs) if target else path



# Approved sampling (BetterBench config/qwen-thinking-xtx.json). Fill in what the client leaves out; never pass temperature 0.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def apply_sampling(body):
    try:
        j = json.loads(body)
    except ValueError:
        return body
    if not isinstance(j, dict) or not isinstance(j.get("messages"), list):
        return body
    for k, v in SAMPLING.items():
        if j.get(k) is None:
            j[k] = v
    t = j.get("temperature")
    if not isinstance(t, (int, float)) or t <= 0:
        j["temperature"] = SAMPLING["temperature"]
    return json.dumps(j).encode()

def upstream(method, path, body=None, headers=None, timeout=1800):
    conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=timeout)
    conn.request(method, path, body=body, headers=headers or {})
    return conn, conn.getresponse()


def upstream_json(path):
    """Return parsed JSON from upstream, or None if absent/unparseable."""
    try:
        conn, r = upstream("GET", path, timeout=10)
        data = r.read()
        conn.close()
        if r.status != 200 or not data.strip():
            return None
        return json.loads(data)
    except Exception:
        return None


# Upstream capability cache: vLLM has no /props, so probing it on every call
# costs a round trip and litters vLLM's log with 404s on each client startup.
# Probe once per path shape, then remember.
_UPSTREAM_HAS = {}


def upstream_has(path_key, path):
    if path_key not in _UPSTREAM_HAS:
        _UPSTREAM_HAS[path_key] = upstream_json(path)
    return _UPSTREAM_HAS[path_key]


def model_ids():
    j = upstream_json("/v1/models") or {}
    return [m["id"] for m in j.get("data", [])] or ["local"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet; vLLM already logs requests
        pass

    def _send_json(self, obj, code=200):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ---- synthesized llama.cpp endpoints -------------------------------
    def _health(self):
        # Upstream's own /health wins when it returns real JSON (llama.cpp).
        j = upstream_json("/health")
        if isinstance(j, dict) and "status" in j:
            return self._send_json(j)
        try:
            conn, r = upstream("GET", "/health", timeout=10)
            r.read(); conn.close()
            alive = r.status == 200
        except Exception:
            alive = False
        if not alive:
            return self._send_json({"status": "error"}, 503)
        return self._send_json({"status": "ok"})

    def _props(self):
        j = upstream_has("props" if "model=" not in self.path else "props_model",
                         self.path)
        if isinstance(j, dict) and ("role" in j or "default_generation_settings" in j):
            return self._send_json(j)          # real llama-server behind us

        ids = model_ids()
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        wanted = None
        for kv in qs.split("&"):
            if kv.startswith("model="):
                wanted = kv[6:]
        alias = wanted or ids[0]

        if wanted is None:
            # server-level /props. role must NOT be "router": a non-router
            # server plus max_model_len in /v1/models selects LEGACY mode,
            # which is the single-model shape vLLM actually is.
            return self._send_json({
                "role": "single",
                "max_instances": 1,
                "models_autoload": False,
                "model_alias": alias,
                "model_path": alias,
                "default_generation_settings": {"n_ctx": ARGS.ctx, "params": {}},
                "ui_settings": {},
                "build_info": "vllm-mixedrdna (llamacompat-shim)",
                "cors_proxy_enabled": False,
            })
        return self._send_json({
            "default_generation_settings": {"params": {}, "n_ctx": ARGS.ctx},
            "total_slots": 1,
            "model_alias": alias,
            "model_path": alias,
            "modalities": {"vision": True, "video": False, "audio": False},   # Qwen3.8-27B is a VL model; vLLM loads its tower by default (2026-11-22)
            "media_marker": "",
            "endpoint_slots": False,
            "endpoint_props": True,
            "endpoint_metrics": True,
            "ui": False,
            "ui_settings": {},
            "chat_template": "",
            "chat_template_caps": {},
        })

    # ---- transparent proxy ---------------------------------------------
    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP and k.lower() != "host"}
        if body and method == "POST":
            body = apply_sampling(body)
            headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
            headers["Content-Length"] = str(len(body))
        try:
            conn, r = upstream(method, rewrite(self.path), body, headers)
        except Exception as exc:
            return self._send_json({"error": f"upstream unreachable: {exc}"}, 502)

        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() in HOP or k.lower() == "content-length":
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # Token-smooth streaming: SSE events are newline-delimited. Reading
        # fixed 8 KiB blocks coalesced dozens of token events into one burst
        # (client sees 20-30 tokens arrive at once). Read per line instead and
        # flush immediately, so each engine step's delta reaches the client
        # the moment vLLM emits it.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:                       # stream SSE / long bodies
                line = r.readline()
                if not line:
                    break
                self.wfile.write(b"%X\r\n%s\r\n" % (len(line), line))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/health":
            return self._health()
        if p == "/props":
            return self._props()
        return self._proxy("GET")

    def do_POST(self):
        return self._proxy("POST")

    def do_DELETE(self):
        return self._proxy("DELETE")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.listen), Handler)
    srv.daemon_threads = True
    print(f"llamacompat-shim: 127.0.0.1:{ARGS.listen} -> {ARGS.upstream}",
          file=sys.stderr, flush=True)
    srv.serve_forever()
