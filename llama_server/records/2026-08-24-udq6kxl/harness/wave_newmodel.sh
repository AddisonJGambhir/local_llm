#!/usr/bin/env bash
# UD-Q6_K_XL preset build: f16 KV, 262144 native max context, tensor split.
# Q8_0 forced 49/51; this model is 3.5 GiB smaller so weight can shift toward the
# XTX, which is the higher-bandwidth card. Phase 1 finds the best ratio that fits,
# phase 2 sweeps draft depth on it. The MTP head here is Q6_K (was Q8_0), so
# acceptance is re-measured rather than carried over.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
CSV="$SP/newmodel.csv"
X=/sys/class/drm/card1/device/mem_info_vram_used
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
MY_PID=""; W=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$W" ] && { kill "$W" 2>/dev/null; W=""; }; sleep 5; }
trap 'mykill' EXIT
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $X)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > 23900 )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
phase(){ local label="$1" ts="$2" reps="$3"; shift 3
  log "=== $label  ts=$ts  $*"
  echo "===== $label =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 $* >> "$SP/server.log" 2>&1 &
  MY_PID=$!; peakwatch "$MY_PID" & W=$!
  local ok=0
  for i in $(seq 1 500); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  (( ok==0 )) && { log "  DID NOT LOAD (peak $(cat $SP/.peak) MiB)"; mykill; return 1; }
  log "  loaded, peak $(cat $SP/.peak) MiB"
  python3 "$SP/blockA.py" "$label" "120000" "$reps" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "  $label done, peak $(cat $SP/.peak) MiB"; mykill; }

F16D="--spec-draft-type-k f16 --spec-draft-type-v f16"
log "PHASE 1: split ratio (n4 fixed)"
phase r50_n4 50,50 6 --spec-type draft-mtp --spec-draft-n-max 4 $F16D
phase r52_n4 52,48 6 --spec-type draft-mtp --spec-draft-n-max 4 $F16D
log "PHASE 2: draft depth at 50/50"
phase r50_n3 50,50 8 --spec-type draft-mtp --spec-draft-n-max 3 $F16D
phase r50_n5 50,50 8 --spec-type draft-mtp --spec-draft-n-max 5 $F16D
log "NEW MODEL TUNE COMPLETE -> $CSV"
