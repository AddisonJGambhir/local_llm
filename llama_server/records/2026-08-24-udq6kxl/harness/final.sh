#!/usr/bin/env bash
# Decide the preset: ratio (49/51 vs 50/50) then draft depth at the winner.
# Settled VRAM: 49/51=21,052  50/50=23,963 (ceiling 24,200). 51/49+ saturate the
# XTX and give it no extra work, so they are out.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
CSV="$SP/final.csv"
export MTMD_BACKEND_DEVICE=Vulkan1
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
MY_PID=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 45); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  for i in $(seq 1 60); do (( $(cat $X)/1048576 < 2600 )) && break; sleep 2; done; }
trap 'mykill' EXIT
run(){ local label="$1" ts="$2" nmax="$3" dkv="$4" reps="$5"
  mykill
  log "=== $label ts=$ts n=$nmax dkv=$dkv"
  echo "===== $label =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" \
    --spec-draft-type-k "$dkv" --spec-draft-type-v "$dkv" \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!
  local ok=0 over=0 u mx=0
  for i in $(seq 1 900); do
    kill -0 "$MY_PID" 2>/dev/null || break
    u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    if (( u > 24200 )); then over=$((over+1)); else over=0; fi
    (( over >= 10 )) && { log "  ABORT sustained $u"; kill -9 "$MY_PID"; break; }
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    sleep 1; done
  (( ok==0 )) && { log "  DID NOT LOAD peak=$mx"; mykill; return 1; }
  log "  loaded ${mx} MiB"
  python3 "$SP/blockA.py" "$label" "120000" "$reps" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "  $label done"; mykill; }
log "##### RATIO DECIDER #####"
run F_r49_n4 49,51 4 f16 8
run F_r50_n4 50,50 4 f16 8
log "##### DEPTH SWEEP at 49,51 #####"
run F_r49_n2 49,51 2 f16 8
run F_r49_n3 49,51 3 f16 8
run F_r49_n5 49,51 5 f16 8
log "##### DRAFT KV at 49,51 #####"
run F_r49_n4_q8 49,51 4 q8_0 8
log "##### FINAL SWEEP COMPLETE #####"
