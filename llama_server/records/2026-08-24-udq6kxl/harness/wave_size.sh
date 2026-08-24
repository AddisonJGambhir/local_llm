#!/usr/bin/env bash
# SIZING ONLY. Find the most XTX-heavy tensor-split that fits at c=262144 f16 KV.
# Previous attempt failed on every ratio because MTMD_BACKEND_DEVICE was unset,
# so clip.cpp put the vision tower on the FIRST device (Vulkan0 = XTX), costing
# ~1.1 GiB there. Every shipped preset sets this; my ad-hoc launch did not.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
R=/sys/class/drm/card0/device/mem_info_vram_used
OUT="$SP/sizing.csv"
export MTMD_BACKEND_DEVICE=Vulkan1          # <-- the fix: tower on the R9700
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "ratio,ctx,loaded,peak_xtx,peak_r9700,steady_xtx" > "$OUT"
MY_PID=""; W=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$W" ] && { kill "$W" 2>/dev/null; W=""; }; sleep 5; }
trap 'mykill' EXIT
peakwatch(){ local pid=$1 mx=0 mr=0 over=0 u v; echo 0 > "$SP/.peak"; echo 0 > "$SP/.peakr"
  while kill -0 "$pid" 2>/dev/null; do
    u=$(( $(cat $X)/1048576 )); v=$(( $(cat $R)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    (( v > mr )) && { mr=$v; echo "$mr" > "$SP/.peakr"; }
    if (( u > 23800 )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.4; done }
probe(){ local ts="$1" ctx="$2"
  for i in $(seq 1 60); do (( $(cat $X)/1048576 < 3000 )) && break; sleep 2; done
  log "sizing: ts=$ts ctx=$ctx"
  echo "===== sizing $ts $ctx =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c "$ctx" --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max 4 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!; peakwatch "$MY_PID" & W=$!
  local ok=0
  for i in $(seq 1 500); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  local steady=0
  if (( ok==1 )); then
    # short real generation so buffers are actually touched, then read steady state
    curl -s -m 120 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write one paragraph about GPU memory.","n_predict":128,"temperature":0}' >/dev/null 2>&1
    steady=$(( $(cat $X)/1048576 ))
    log "  LOADED  peak_xtx=$(cat $SP/.peak) steady=$steady peak_r9700=$(cat $SP/.peakr)"
  else
    log "  FAILED  peak_xtx=$(cat $SP/.peak)"
  fi
  echo "$ts,$ctx,$ok,$(cat $SP/.peak),$(cat $SP/.peakr),$steady" >> "$OUT"
  mykill; }
log "SIZING PROBE (tower on Vulkan1)"
probe 45,55 262144
probe 48,52 262144
probe 50,50 262144
log "SIZING PROBE COMPLETE"
cat "$OUT"
