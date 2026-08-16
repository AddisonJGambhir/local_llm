#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "httpx>=0.27"]
# ///
"""
diagnose_hermes_400.py

Find why Hermes gets HTTP 400 from llama-server while a curl replay of the
captured request body returns 200.

Run with Hermes' own venv so the openai-python version matches what Hermes uses:

    /home/addison-gambhir/Desktop/hermes-agent-main/venv/bin/python3 \
        /home/addison-gambhir/Desktop/local_llm/src/diagnostics/diagnose_hermes_400.py

Three tests:
  1. Raw HTTP, non-streaming — baseline (known to return 200).
  2. Raw HTTP, streaming      — isolates whether stream=true alone breaks it.
  3. openai-python SDK call   — reproduces Hermes' code path AND captures the
                                 actual on-the-wire JSON via an httpx event
                                 hook, then diffs it against the dump.

If test 3 fails where tests 1 and 2 succeed, the diff output names the field
the SDK added/renamed that llama-server rejects.
"""

import glob
import json
import os
import sys
import urllib.error
import urllib.request

DUMP_GLOB = "/home/addison-gambhir/.hermes/sessions/request_dump_*.json"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "local"


def load_latest_dump():
    dumps = sorted(glob.glob(DUMP_GLOB), key=os.path.getmtime, reverse=True)
    if not dumps:
        sys.exit("No Hermes request dumps found at " + DUMP_GLOB)
    path = dumps[0]
    print(f"[dump] using: {path}")
    with open(path) as f:
        return path, json.load(f)


def extract_body(dump):
    body = dump["request"]["body"]
    return json.loads(body) if isinstance(body, str) else body


def post_raw(payload, *, stream):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream" if stream else "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(300)
            print(f"  HTTP {r.status} — first 300 bytes: {head!r}")
            return r.status
    except urllib.error.HTTPError as e:
        body = e.read()
        print(f"  HTTP {e.code} — error body: {body!r}")
        return e.code


def test_sdk(dump_body):
    """Reproduce Hermes' code path, capturing the actual wire payload."""
    try:
        from openai import BadRequestError, OpenAI  # type: ignore[import-not-found]
    except ImportError:
        print("  openai package not importable in this Python — re-run with Hermes' venv:")
        print("    /home/addison-gambhir/Desktop/hermes-agent-main/venv/bin/python3 "
              + os.path.abspath(__file__))
        return

    try:
        import httpx
    except ImportError:
        print("  httpx not importable — re-run with Hermes' venv.")
        return

    captured = {}

    def on_request(req):
        if "chat/completions" not in str(req.url):
            return
        try:
            captured["wire"] = json.loads(req.content)
        except Exception:
            captured["wire_raw"] = req.content[:2000]
        captured["headers"] = dict(req.headers)

    def on_response(resp):
        if "chat/completions" not in str(resp.request.url):
            return
        captured["status"] = resp.status_code
        if resp.status_code >= 400:
            resp.read()
            captured["err_body"] = resp.content[:2000]

    http = httpx.Client(
        event_hooks={"request": [on_request], "response": [on_response]},
        timeout=60.0,
    )
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=http)

    kwargs = dict(
        model=dump_body["model"],
        messages=dump_body["messages"],
        tools=dump_body.get("tools") or [],
        max_tokens=dump_body.get("max_tokens"),
        stream=True,
    )
    print(f"  calling: stream=True, tools={len(kwargs['tools'])}, msgs={len(kwargs['messages'])}")
    try:
        stream = client.chat.completions.create(**kwargs)
        chunks = 0
        for _ in stream:
            chunks += 1
            if chunks >= 3:
                break
        print(f"  SUCCESS — received {chunks}+ chunks")
    except BadRequestError as e:
        print(f"  BadRequestError: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            print(f"  response status: {resp.status_code}")
            print(f"  response text:   {resp.text!r}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

    if "wire" in captured:
        wire = captured["wire"]
        print(f"\n  --- wire payload keys: {sorted(wire.keys())}")
        print(f"  --- dump payload keys: {sorted(dump_body.keys())}")
        added = set(wire) - set(dump_body)
        removed = set(dump_body) - set(wire)
        if added:
            print(f"  ! keys ADDED by SDK (not in dump): {sorted(added)}")
            for k in sorted(added):
                print(f"      {k} = {wire[k]!r}")
        if removed:
            print(f"  ! keys REMOVED by SDK (in dump only): {sorted(removed)}")
        for k in sorted(set(wire) & set(dump_body)):
            if k in ("messages", "tools"):
                # don't dump giant fields, just length-compare
                wlen = len(wire[k]) if hasattr(wire[k], "__len__") else "?"
                dlen = len(dump_body[k]) if hasattr(dump_body[k], "__len__") else "?"
                if wlen != dlen:
                    print(f"  ! {k} length differs: dump={dlen} wire={wlen}")
                continue
            if wire[k] != dump_body[k]:
                print(f"  ! {k} differs: dump={dump_body[k]!r} wire={wire[k]!r}")
    elif "wire_raw" in captured:
        print(f"  wire payload was not JSON: {captured['wire_raw']!r}")

    if captured.get("status", 200) >= 400 and "err_body" in captured:
        print(f"\n  --- llama-server error body: {captured['err_body']!r}")


def main():
    _, dump = load_latest_dump()
    body = extract_body(dump)
    err = dump.get("error") or {}
    print(f"[dump] captured error: status={err.get('status_code')} body={err.get('body')!r}")
    print(f"[dump] body keys: {sorted(body.keys())}")
    print(f"[dump] messages={len(body['messages'])} "
          f"tools={len(body.get('tools') or [])} "
          f"max_tokens={body.get('max_tokens')}")
    print()

    print("=== Test 1: raw HTTP, non-streaming ===")
    p1 = dict(body); p1["stream"] = False
    post_raw(p1, stream=False)
    print()

    print("=== Test 2: raw HTTP, streaming ===")
    p2 = dict(body); p2["stream"] = True
    p2["stream_options"] = {"include_usage": True}
    post_raw(p2, stream=True)
    print()

    print("=== Test 3: openai-python SDK (Hermes' code path) ===")
    test_sdk(body)


if __name__ == "__main__":
    main()
