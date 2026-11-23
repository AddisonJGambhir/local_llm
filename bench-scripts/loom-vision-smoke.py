import base64
import json
import re
import struct
import sys
import urllib.request
import zlib


def image_url(rgb):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))

    width = height = 224
    rows = (b"\0" + bytes(rgb) * width) * height
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=180)


base = sys.argv[1].rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "local"
if model == "local":
    with request("/props") as response:
        properties = json.load(response)
    assert properties["modalities"]["vision"] is True, properties
    with request("/v1/models") as response:
        models = json.load(response)["data"]
    alias = next(m for m in models if m["id"] == "local")
    assert "image" in alias["architecture"]["input_modalities"], alias
    print(f"local advertises image support at {properties['n_ctx']} tokens", flush=True)

for expected, rgb, stream in (("red", (255, 0, 0), False), ("blue", (0, 0, 255), True)):
    with request("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url(rgb)}},
            {"type": "text", "text": "What is the dominant color in this image? Reply with the single color word."},
        ]}],
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "max_tokens": 512,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "stream": stream,
    }) as response:
        if stream:
            pieces = []
            done = False
            for line in response:
                if line.startswith(b"data:"):
                    if line.strip() == b"data: [DONE]":
                        done = True
                        break
                    payload = json.loads(line[5:])
                    for choice in payload.get("choices", []):
                        pieces.append(choice["delta"].get("content") or "")
            assert done, "stream did not send [DONE]"
            result = "".join(pieces)
        else:
            choice = json.load(response)["choices"][0]
            result = choice["message"]["content"]
            if not result:
                print(f"Empty final answer: {choice}", flush=True)
    print(f"{expected} input returned {result!r} ({'stream' if stream else 'JSON'})", flush=True)
    assert re.search(rf"\b{expected}\b", result, re.IGNORECASE), result
