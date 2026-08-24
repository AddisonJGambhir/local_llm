#!/usr/bin/env bash
# Push as much weight onto the XTX as fits. 52/48 previously "failed" only because
# MTMD_BACKEND_DEVICE was unset and the vision tower landed on the XTX (~1.1 GiB).
# Size first, then decode-test whatever fits, most-XTX-heavy first.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
R=/sys/class/drm/card0/device/mem_info_vram_used
SZ="$SP/sizing2.csv"; TG="$SP/ratio_tg.csv"
export MTMD_BACKEND_DEVICE=Vulkan1
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$SZ" ] || echo "ratio;ctx;loaded;peak_xtx;peak_r9700;steady_xtx" > "$SZ"
MY_PID=""; W=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$W" ] && { kill "$W" 2>/dev/null; W=""; }; sleep 5; }
trap 'mykill' EXIT
watch_(){ local pid=$1 mx=0 mr=0 over=0 u v; echo 0 > "$SP/.peak"; echo 0 > "$SP/.peakr"
  while kill -0 "$pid" 2>/dev/null; do
    u=$(( $(cat $X)/1048576 )); v=$(( $(cat $R)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    (( v > mr )) && { mr=$v; echo "$mr" > "$SP/.peakr"; }
    if (( u > 23850 )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.4; done }
up(){ local ts="$1" nmax="$2"
  for i in $(seq 1 60); do (( $(cat $X)/1048576 < 3000 )) && break; sleep 2; done
  echo "===== $ts n$nmax =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!; watch_ "$MY_PID" & W=$!
  for i in $(seq 1 500); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && return 0
    kill -0 "$MY_PID" 2>/dev/null || return 1
    sleep 1; done
  return 1; }
FITS=()
for TS in 53,47 52,48 51,49; do
  log "sizing $TS"
  if up "$TS" 4; then
    curl -s -m 180 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a paragraph about GPU memory bandwidth.","n_predict":256,"temperature":0}' >/dev/null 2>&1
    st=$(( $(cat $X)/1048576 ))
    log "  $TS LOADED peak=$(cat $SP/.peak) steady=$st r9700=$(cat $SP/.peakr)"
    echo "$TS;262144;1;$(cat $SP/.peak);$(cat $SP/.peakr);$st" >> "$SZ"
    (( st < 23400 )) && FITS+=("$TS")
  else
    log "  $TS FAILED peak=$(cat $SP/.peak)"
    echo "$TS;262144;0;$(cat $SP/.peak);$(cat $SP/.peakr);0" >> "$SZ"
  fi
  mykill
done
log "fits with margin: ${FITS[*]:-none}"
for TS in "${FITS[@]:-}"; do
  [ -z "$TS" ] && continue
  log "=== decode $TS n4"
  if up "$TS" 4; then
    python3 "$SP/blockA.py" "r${TS%%,*}_n4" "120000" 8 "$TG" 2>&1 | tee -a "$SP/run.log"
    log "  done peak $(cat $SP/.peak)"
  fi
  mykill
done
log "XTX-HEAVY SWEEP COMPLETE"
