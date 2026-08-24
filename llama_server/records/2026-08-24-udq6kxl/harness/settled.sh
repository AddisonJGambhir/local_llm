#!/usr/bin/env bash
# VRAM spikes during load then drops and stabilizes. Earlier probes recorded the
# SPIKE and called it steady (peak==steady in every row -- read at the same instant),
# and aborted on transient spikes, which wrongly rejected 51/49 and 52/48.
# This: tolerates spikes, aborts only if SUSTAINED over the real 24,200 ceiling,
# and records the value after it stops moving.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
R=/sys/class/drm/card0/device/mem_info_vram_used
export MTMD_BACKEND_DEVICE=Vulkan1
OUT="$SP/settled.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "ratio;spike_xtx;settled_xtx;settled_r9700;loaded" > "$OUT"
MY_PID=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 45); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  for i in $(seq 1 60); do (( $(cat $X)/1048576 < 2600 )) && break; sleep 2; done; }
trap 'mykill' EXIT
# wait until value unchanged (+/-8 MiB) for 12 consecutive 1s samples
settle_read(){ local f=$1 last=-999 same=0 cur
  for i in $(seq 1 120); do
    cur=$(( $(cat $f)/1048576 ))
    if (( cur > last-8 && cur < last+8 )); then same=$((same+1)); else same=0; fi
    (( same >= 12 )) && { echo "$cur"; return 0; }
    last=$cur; sleep 1
  done; echo "$cur"; }
one(){ local ts="$1"
  mykill
  log "ratio $ts"
  echo "===== settled $ts =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max 4 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!
  local spike=0 ok=0 over=0 u
  for i in $(seq 1 900); do
    kill -0 "$MY_PID" 2>/dev/null || break
    u=$(( $(cat $X)/1048576 )); (( u > spike )) && spike=$u
    # abort only on SUSTAINED overage of the real ceiling
    if (( u > 24200 )); then over=$((over+1)); else over=0; fi
    (( over >= 10 )) && { log "  ABORT: sustained ${u} > 24200 for 10s"; kill -9 "$MY_PID"; break; }
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    sleep 1
  done
  local sx=0 sr=0
  if (( ok==1 )); then
    curl -s -m 240 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Explain GPU memory bandwidth.","n_predict":256,"temperature":0}' >/dev/null 2>&1
    sx=$(settle_read "$X"); sr=$(settle_read "$R")
    log "  $ts spike=${spike} SETTLED=${sx} r9700=${sr}"
  else
    log "  $ts FAILED spike=${spike}"
  fi
  echo "$ts;$spike;$sx;$sr;$ok" >> "$OUT"
  mykill
}
log "SETTLED VRAM CHECK (spike tolerated, ceiling 24200 sustained)"
for TS in 49,51 50,50 51,49 52,48 53,47; do one "$TS"; done
log "SETTLED CHECK COMPLETE"
column -s';' -t "$OUT"
