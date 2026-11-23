#!/bin/bash
B=~/Desktop/local_llm/bench-scripts; CAP=${1:-31gib}
st(){ curl -s -m3 localhost:11438/v1/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["loom"].get("registration"))'; }
fresh(){ systemctl --user stop loom-vtest-proxy loom-vtest-backend; LOOM_EXTRA="--memory_bytes=$CAP" KEEP=1 $B/lvtest.sh Inkling-2-Mini-r9700-c262144-d3-vision model-c393216.json 393216 Inkling-2-Mini >/dev/null 2>&1; curl -s -m30 -X POST -H 'Content-Type: application/json' -d '{"model":"Inkling-2-Mini"}' localhost:11438/v1/models/pin >/dev/null; }
echo "### pin + cap $CAP"
for n in 130000 160000 262000; do fresh; echo "== image after $n"; python3 -I $B/vision-longctx.py 1239 ~/ocean-shot.png $n 2>&1 | tail -2 | cut -c1-170; echo "   model: $(st)"; done
fresh; echo "== text probes"; python3 -I $B/loom-cap-probe.py 320000 300000 285000 270000
echo "   small after: $(curl -s -m120 localhost:1239/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"say hi"}],"max_tokens":300,"temperature":1.0,"top_p":0.95,"top_k":20}' | head -c 80)"
