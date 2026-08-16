#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
llama_proxy.py — transparent TCP passthrough proxy in front of llama-server.

Listens on 127.0.0.1:1235, forwards all bytes both directions to llama-server
on 127.0.0.1:1234, and tees every TCP connection's traffic to disk so we can
postmortem any 400/500 response.

Why this exists:
  llama-server returns HTTP 400 with empty body for ~half of Hermes' requests
  after tool calls. Server-side log shows the 400 but doesn't reveal whether
  the request reaching the server was malformed, whether a keep-alive
  connection got into a bad state, or whether streaming framing broke. This
  proxy captures the exact bytes on the wire so we can answer that.

Usage:
  1. Start the proxy:
       /home/addison-gambhir/Desktop/local_llm/llama_server/bin/llama-proxy

  2. Point Hermes at the proxy (one-time edit; revert when done):
       sed -i 's|localhost:1234|localhost:1235|' ~/.hermes/config.yaml
     Restart Hermes.

  3. Reproduce the failure (any Hermes session that triggers the 400s).

  4. Examine the dumps:
       ls /tmp/llama-proxy/
       grep -F '!!!' /tmp/llama-proxy/index.log   # connections that 4xx/5xx'd

     Each connection produces two files:
       {conn_id}.in   — bytes Hermes sent (client → server)
       {conn_id}.out  — bytes llama.cpp returned (server → client)

  5. When done, revert Hermes config:
       sed -i 's|localhost:1235|localhost:1234|' ~/.hermes/config.yaml

Diagnostic interpretation guide:
  - If a `.in` for a 400 connection shows a malformed HTTP request line,
    truncated headers, or an invalid JSON body → client-side bug (httpx
    / openai-python / Hermes message serialization).
  - If a `.in` looks identical to a `.in` from a successful 200 connection
    on the same Hermes session → server-side bug. Likely candidates:
      (a) keep-alive: previous response in the same TCP connection left
          state that breaks the next request's parse;
      (b) streaming end-of-stream framing race;
      (c) some llama.cpp httplib quirk on consecutive POSTs.
  - If multiple connections start within milliseconds, keep-alive isn't
    being used — single-request connections, so keep-alive is ruled out.
"""

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path

UPSTREAM = ("127.0.0.1", 1234)
LISTEN = ("127.0.0.1", 1235)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
DUMP_DIR = PROJECT_ROOT / "output" / "diagnostics" / "llama-proxy" / RUN_STAMP
INDEX_LOG = DUMP_DIR / "index.log"

# Match each HTTP response status line in the server -> client stream
STATUS_RE = re.compile(rb"HTTP/\d\.\d (\d{3})")


async def tee(reader, writer, dump_path: Path):
    """Forward reader → writer while teeing every byte to dump_path."""
    with open(dump_path, "ab") as f:
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                f.write(data)
                f.flush()
                writer.write(data)
                await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def handle(client_reader, client_writer):
    peer = client_writer.get_extra_info("peername") or ("?", 0)
    conn_id = f"{int(time.time()*1000):013d}_{peer[1]:05d}"
    in_path = DUMP_DIR / f"{conn_id}.in"
    out_path = DUMP_DIR / f"{conn_id}.out"
    t0 = time.time()

    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(*UPSTREAM)
    except Exception as e:
        with open(INDEX_LOG, "a") as f:
            f.write(f"[{conn_id}] UPSTREAM_FAIL: {e}\n")
        client_writer.close()
        return

    await asyncio.gather(
        tee(client_reader, upstream_writer, in_path),
        tee(upstream_reader, client_writer, out_path),
        return_exceptions=True,
    )

    # Summarize this connection in the index
    try:
        out_bytes = out_path.read_bytes()
    except FileNotFoundError:
        out_bytes = b""
    statuses = STATUS_RE.findall(out_bytes)
    in_size = in_path.stat().st_size if in_path.exists() else 0
    out_size = out_path.stat().st_size if out_path.exists() else 0
    dur = time.time() - t0
    has_non2xx = any(int(s) >= 400 for s in statuses)

    with open(INDEX_LOG, "a") as f:
        marker = "  !!!" if has_non2xx else ""
        sc_str = ",".join(s.decode() for s in statuses) or "-"
        f.write(
            f"[{conn_id}] dur={dur:6.2f}s "
            f"in={in_size:>9}B out={out_size:>9}B "
            f"statuses=[{sc_str}]{marker}\n"
        )


async def main():
    DUMP_DIR.mkdir(parents=True, exist_ok=False)
    INDEX_LOG.write_text("")

    server = await asyncio.start_server(handle, *LISTEN)
    print(f"proxy : {LISTEN[0]}:{LISTEN[1]} → {UPSTREAM[0]}:{UPSTREAM[1]}")
    print(f"dumps : {DUMP_DIR}/")
    print(f"index : {INDEX_LOG}")
    print()
    print("To use, point Hermes at the proxy:")
    print(f"  sed -i 's|localhost:{UPSTREAM[1]}|localhost:{LISTEN[1]}|' "
          "~/.hermes/config.yaml")
    print("Then restart Hermes. Ctrl+C here when you're done capturing.")
    print()
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nproxy stopped")
