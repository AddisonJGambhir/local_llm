#!/usr/bin/env bash
# RECORD ATTEMPT. Exact conditions of the 81.67 t/s number:
#   -c 98304, --split-mode tensor --tensor-split 50,50, f16 KV, ub 1024,
#   matrix.SHORT_PROMPT, 2048-token generation, temp 0 + top_k 1 + ignore_eos.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
CSV="$SP/record.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
REPS=7
XTXF=/sys/class/drm/card1/device/mem_info_vram_used
CEIL=24000
MY_PID=""; WATCH=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$WATCH" ] && { kill "$WATCH" 2>/dev/null; WATCH=""; }; sleep 4; }
trap 'mykill' EXIT
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 4 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 1; done; echo "$mx" > "$SP/.peak"; }

# arm <label> <binary-dir> <spec args...>
arm(){ local label="$1" bindir="$2"; shift 2
  if ss -lntH 'sport = :1234' 2>/dev/null | grep -q . || pgrep -x llama-server >/dev/null; then
    log "RECORD $label SKIPPED: port/server busy"; return 1; fi
  log "=== RECORD arm $label"
  echo "===== $label =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$bindir" setsid "$bindir/llama-server" \
    -m "$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift \
    --split-mode tensor --main-gpu 1 --tensor-split 50,50 \
    -c 98304 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576 \
    --host 127.0.0.1 --port 1234 $* >> "$SP/server.log" 2>&1 &
  MY_PID=$!; peakwatch "$MY_PID" & WATCH=$!
  local ok=0
  for i in $(seq 1 400); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  (( ok==0 )) && { log "RECORD $label FAILED TO START"; mykill; return 1; }
  python3 "$SP/blockRec.py" "$label" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "RECORD $label done peak $(cat $SP/.peak) MiB"; mykill; }

ADAPT="--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
N3="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
A=$LLM/llama.cpp-tp-adaptive/build/bin
C=$LLM/llama.cpp-tp-chain/build/bin
M=$LLM/llama.cpp-tp-mmvq/build/bin

log "RECORD RUN START (target to beat: 81.67 t/s)"
arm baseline_shipped      "$A" $ADAPT                       # must reproduce ~81.67
arm chain_knobs_off       "$C" $ADAPT                       # regression check vs baseline
LLAMA_SPEC_CHAIN=1 LLAMA_META_MIRROR_OUTPUT=1 LLAMA_SCHED_POOL=8 \
  arm chain_all_knobs     "$C" $ADAPT
LLAMA_META_MIRROR_OUTPUT=1 \
  arm chain_mirror_only   "$C" $ADAPT
arm mmvq_adaptive         "$M" $ADAPT
arm mmvq_n3               "$M" $N3
arm baseline_n3           "$A" $N3                          # record-condition n3 reference
log "RECORD RUN COMPLETE -> $CSV"
