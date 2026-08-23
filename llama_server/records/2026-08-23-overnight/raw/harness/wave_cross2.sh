#!/usr/bin/env bash
# Crossover, corrected. The first attempt concatenated repetitive text then source,
# so the TAIL was source code for every mix below 100 and the model's continuation
# regime did not track the ratio. These prompts interleave in 200-char blocks.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
P=$LLM/llama_server/src/benchmarks/prompts
export REC_CSV="$SP/crossover2.csv"
source "$SP/wave_rec_lib.sh"
REPS=5
sweep(){ local label="$1" bindir="$2"; shift 2
  if ss -lntH 'sport = :1234' 2>/dev/null | grep -q . || pgrep -x llama-server >/dev/null; then
    log "CROSS2 $label SKIPPED: port busy"; return 1; fi
  log "=== CROSS2 $label"; echo "===== cross2 $label =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$bindir" setsid "$bindir/llama-server" \
    -m "$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift \
    --split-mode tensor --main-gpu 1 --tensor-split 50,50 \
    -c 98304 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576 \
    --host 127.0.0.1 --port 1234 $* >> "$SP/server.log" 2>&1 &
  MY_PID=$!; peakwatch "$MY_PID" & W=$!
  local ok=0
  for i in $(seq 1 400); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  (( ok==0 )) && { log "CROSS2 $label FAILED TO START"; mykill; return 1; }
  for pct in 000 025 050 075 100; do
    PROMPT_FILE="$P/ilv_$pct.txt" python3 "$SP/blockRec.py" "${label}_ilv$pct" "$REPS" "$REC_CSV" 2>&1 | tee -a "$SP/run.log"
  done
  log "CROSS2 $label done peak $(cat $SP/.peak) MiB"; mykill; }
wait_vram_free
log "CROSSOVER SWEEP 2 START (interleaved prompts)"
sweep adaptive "$A" $ADAPT
sweep n3       "$A" $N3
log "CROSSOVER SWEEP 2 COMPLETE -> $REC_CSV"
