#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
watch_llama_server.py — surface the server-side reason for HTTP 400s.

Reads the managed llama-server log in real time and prints:
  • Every request line (POST /v1/chat/completions ... <status>)
  • The ~20 lines of stderr that preceded any non-2xx response, with
    template / slot / jinja / draft / error keywords highlighted.

Usage:
    1. Start the managed server with `bin/llamactl start`.
    2. Run `bin/llama-watch`.
    3. Reproduce the failure in Hermes.

The watcher highlights every 4xx/5xx response with its preceding context so
the actual rejection reason (template error, slot busy, draft cache mismatch,
context overflow) is visible — instead of an opaque "Error code: 400" from
the client side.
"""

import os
import re
import sys
import time
from collections import deque
from pathlib import Path

LOG = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    / "local-llm"
    / "llama-server.log"
)
CONTEXT_BEFORE = 25  # lines of preceding context to print for each non-2xx

# Patterns to highlight in context lines
HIGHLIGHT = re.compile(
    r"(error|fail|invalid|reject|template|jinja|slot|draft|busy|"
    r"oom|cuda|hip|rocm|nan|inf|overflow|exceed|truncat|warn)",
    re.IGNORECASE,
)
# llama-server request log line — covers httplib-style and slog formats
REQUEST_LINE = re.compile(
    r'(request:|"POST\s+/v[01]/|/v1/chat/completions|/v1/completions).*?(\b[1-5]\d{2}\b)',
    re.IGNORECASE,
)
NON_2XX = re.compile(r"\b[45]\d{2}\b")

RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def highlight(line: str) -> str:
    return HIGHLIGHT.sub(lambda m: f"{YELLOW}{m.group(0)}{RESET}", line)


def fmt_status(line: str) -> str:
    m = REQUEST_LINE.search(line)
    if not m:
        return line
    code = m.group(2)
    if code.startswith("2"):
        return f"{DIM}{line}{RESET}"
    return f"{RED}{BOLD}{line}{RESET}"


def follow(path: Path):
    """Generator yielding new lines as they're appended (tail -F semantics)."""
    while not path.exists():
        print(f"[watcher] waiting for {path} to appear...", flush=True)
        time.sleep(1)
    f = path.open("r", errors="replace")
    # Start at end of file — we only care about new events
    f.seek(0, os.SEEK_END)
    inode = path.stat().st_ino
    while True:
        line = f.readline()
        if line:
            yield line.rstrip("\n")
            continue
        # No new data — check for rotation
        time.sleep(0.2)
        try:
            new_inode = path.stat().st_ino
        except FileNotFoundError:
            continue
        if new_inode != inode:
            f.close()
            f = path.open("r", errors="replace")
            inode = new_inode


def main():
    if not LOG.exists():
        print(f"[watcher] {LOG} does not exist.")
        print("[watcher] Start the managed server first:")
        print("[watcher]   bin/llamactl start")
        sys.exit(1)

    print(f"[watcher] tailing {LOG}")
    print(f"[watcher] preceding-context window: {CONTEXT_BEFORE} lines")
    print(f"[watcher] highlighting non-2xx responses in {RED}red{RESET}, "
          f"suspect keywords in {YELLOW}yellow{RESET}")
    print()

    ring = deque(maxlen=CONTEXT_BEFORE)
    non_2xx_seen = 0

    for line in follow(LOG):
        ring.append(line)

        # Treat any line with a request-like pattern + non-2xx as a hit
        req_match = REQUEST_LINE.search(line)
        is_hit = bool(req_match and not req_match.group(2).startswith("2"))

        # Also treat bare "error"/"fail" lines as hits if they aren't already
        # part of a normal startup log (suppress the obvious benign ones)
        is_error_line = bool(
            HIGHLIGHT.search(line)
            and not any(s in line for s in (
                "loading model", "model loaded", "ggml_cuda",
                "warming up", "build = ",
            ))
        )

        if is_hit:
            non_2xx_seen += 1
            print(f"\n{'=' * 78}")
            print(f"{RED}{BOLD}[HIT #{non_2xx_seen}] non-2xx response{RESET}")
            print(f"{'=' * 78}")
            print(f"{CYAN}Preceding context ({len(ring)} lines):{RESET}")
            for r in list(ring)[:-1]:
                print(f"  {highlight(r)}")
            print(f"{CYAN}Request line:{RESET}")
            print(f"  {fmt_status(line)}")
            print()
        elif is_error_line:
            print(f"{highlight(line)}")
        # Otherwise: silent (keeps signal-to-noise high)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watcher] stopped")
