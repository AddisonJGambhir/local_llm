#!/bin/bash
# cap-safety.sh CAP : fresh server with --memory_bytes=CAP; push text 320K,310K,300K,290K,280K (stop at first OK) - model must stay alive
B=~/Desktop/local_llm/bench-scripts
systemctl --user stop loom-vtest-proxy loom-vtest-backend
LOOM_EXTRA="--memory_bytes=$1" KEEP=1 $B/lvtest.sh Inkling-2-Mini-r9700-c262144-d3-vision model-c393216.json 393216 Inkling-2-Mini >/dev/null 2>&1
echo "== cap $1"; python3 -I $B/loom-cap-probe.py 320000 310000 300000 290000 280000
echo "small request after: $(curl -s -m60 localhost:1239/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"say hi"}],"max_tokens":300,"temperature":1.0,"top_p":0.95,"top_k":20}' | head -c 120)"
