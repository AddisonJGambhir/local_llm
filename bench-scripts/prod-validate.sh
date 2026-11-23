#!/bin/bash
# validate the production r9700-vision config: model.json (262144), --memory_bytes=31gib, pinned
B=~/Desktop/local_llm/bench-scripts
st(){ curl -s -m3 localhost:11438/v1/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["loom"].get("registration"))'; }
fresh(){ systemctl --user stop loom-vtest-proxy loom-vtest-backend; LOOM_EXTRA="--memory_bytes=31gib" KEEP=1 $B/lvtest.sh Inkling-2-Mini-r9700-c262144-d3-vision model.json 262144 Inkling-2-Mini >/dev/null 2>&1; curl -s -m30 -X POST -H 'Content-Type: application/json' -d '{"model":"Inkling-2-Mini"}' localhost:11438/v1/models/pin >/dev/null; }
fresh; echo "== text over the window (260K + 8K reply) then 250K"; python3 -I $B/loom-cap-probe.py 260000 250000; echo "   model: $(st)"
echo "== image after 250K in same server"; python3 -I $B/vision-longctx.py 1239 ~/ocean-shot.png 120000 2>&1 | tail -1 | cut -c1-140; echo "   model: $(st)"
fresh; echo "== 6 x 40K conversations then image"; python3 -I $B/vision-multiconv.py 1239 ~/ocean-shot.png 6 40000 2>&1 | tail -2 | cut -c1-140; echo "   model: $(st)"
echo "== image at 200K (expect clean refusal)"; python3 -I $B/vision-longctx.py 1239 ~/ocean-shot.png 200000 2>&1 | tail -1 | cut -c1-160; echo "   model: $(st)"
echo "== small after: $(curl -s -m120 localhost:1239/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"say hi"}],"max_tokens":300,"temperature":1.0,"top_p":0.95,"top_k":20}' | head -c 80)"
