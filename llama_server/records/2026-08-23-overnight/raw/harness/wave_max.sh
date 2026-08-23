#!/usr/bin/env bash
# MAX TG: every candidate config against the committed high-acceptance prompt
# (src/benchmarks/prompts/agentic_toolcalls.txt, ~1985 tok of repeated tool calls).
# This is the regime the unreproducible 81.67 figure came from. Upper bound, not realistic.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
export PROMPT_FILE=$LLM/llama_server/src/benchmarks/prompts/agentic_toolcalls.txt
CSV="$SP/maxtg.csv"; REPS=7
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=24000
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
MY_PID=""; W=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$W" ] && { kill "$W" 2>/dev/null; W=""; }; sleep 4; }
trap 'mykill' EXIT
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 4 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 1; done; echo "$mx" > "$SP/.peak"; }
arm(){ local label="$1" bindir="$2"; shift 2
  if ss -lntH 'sport = :1234' 2>/dev/null | grep -q . || pgrep -x llama-server >/dev/null; then
    log "MAXTG $label SKIPPED: port busy"; return 1; fi
  log "=== MAXTG arm $label"; echo "===== maxtg $label =====" >> "$SP/server.log"
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
  (( ok==0 )) && { log "MAXTG $label FAILED TO START"; mykill; return 1; }
  python3 "$SP/blockRec.py" "$label" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "MAXTG $label done peak $(cat $SP/.peak) MiB"; mykill; }

ADAPT="--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
N3="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
A=$LLM/llama.cpp-tp-adaptive/build/bin; C=$LLM/llama.cpp-tp-chain/build/bin; M=$LLM/llama.cpp-tp-mmvq/build/bin
log "MAX TG RUN START (high-acceptance prompt; 81.67 came from this regime)"
arm max_shipped_adaptive "$A" $ADAPT
arm max_shipped_n3       "$A" $N3
LLAMA_SPEC_CHAIN=1 LLAMA_META_MIRROR_OUTPUT=1 LLAMA_SCHED_POOL=8 arm max_chain_all "$C" $ADAPT
arm max_mmvq_adaptive    "$M" $ADAPT
arm max_mmvq_n3          "$M" $N3
log "MAX TG RUN COMPLETE -> $CSV"
