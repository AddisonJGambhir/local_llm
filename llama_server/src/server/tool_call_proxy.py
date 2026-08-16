#!/usr/bin/env python3
"""Reverse proxy in front of llama-server that forces single-tool-call turns.

Cline's built-in system prompt instructs the model to batch/parallelize tool
calls, and Cline never sends the OpenAI `parallel_tool_calls` field, so
llama.cpp falls back to the chat template's own capability flag (true for
Qwen3.6) and the model does emit multiple tool_calls in one turn. Cline's
Hub-based execution pipeline only expects one tool call per turn: a second
call in the same response either crashes the Hub worker ("Hub connection
closed", code=1006) or, mid-transition into the second call, breaks llama.cpp's
grammar-constrained parser and leaks raw `<function=...><parameter=...>`
markup into the content instead of a structured tool_calls delta.

This proxy sits between Cline and llama-server and injects
`"parallel_tool_calls": false` into every chat-completions request that
declares tools, forcing exactly one tool call per turn regardless of what
Cline's prompt asks for. Verified: with this forced off, llama.cpp caps
generation to one tool call even when explicitly prompted to do two.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_FILE = os.environ.get("TOOL_CALL_PROXY_LOG")


def make_handler(upstream: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._proxy()

        def do_POST(self) -> None:
            self._proxy()

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            if self.path.startswith("/v1/chat/completions") and body:
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
                if payload is not None and payload.get("tools"):
                    payload["parallel_tool_calls"] = False
                    body = json.dumps(payload).encode("utf-8")

            if LOG_FILE and body:
                with open(LOG_FILE, "a") as f:
                    f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.command} {self.path} ====\n")
                    try:
                        f.write(json.dumps(json.loads(body), indent=2))
                    except ValueError:
                        f.write(body.decode("utf-8", errors="replace"))
                    f.write("\n")

            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length")
            }
            request = urllib.request.Request(
                upstream + self.path, data=body or None, headers=headers, method=self.command
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    self.send_response(response.status)
                    for k, v in response.headers.items():
                        if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = response.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n" % len(chunk))
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
            except OSError:
                self.send_response(502)
                self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, default=1235)
    parser.add_argument("--upstream", default="http://127.0.0.1:1234")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.listen_port), make_handler(args.upstream))
    print(f"tool_call_proxy: :{args.listen_port} -> {args.upstream} (forcing parallel_tool_calls=false)")
    sys.stdout.flush()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
