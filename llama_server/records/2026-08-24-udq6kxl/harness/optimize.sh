#!/usr/bin/env bash
# FULL OPTIMIZATION for the UD-Q6_K_XL preset: f16 KV, 262144 context, fastest decode.
#   A) decode vs split ratio among ratios that fit (45/55 done: 43.28)
#   B) draft depth sweep at the winning ratio (n2..n5 -- real use is 2-4)
#   C) draft KV type at the winner
#   D) if the XTX-heavy ratio wins but is too tight at 262144, retry it at 245760
#      to see whether trading 6% context buys a faster ratio
# All at 120k depth, novel content, 8 reps, acceptance recorded.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
CSV="$SP/opt.csv"
export MTMD_BACKEND_DEVICE=Vulkan1
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
MY_PID=""; W=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""
  [ -n "$W" ] && { kill "$W" 2>/dev/null; W=""; }; sleep 5; }
trap 'mykill' EXIT
watch_(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $X)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > 23850 )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.4; done }
run(){ # run <label> <ts> <ctx> <nmax> <draftkv> <reps>
  local label="$1" ts="$2" ctx="$3" nmax="$4" dkv="$5" reps="$6"
  for i in $(seq 1 90); do (( $(cat $X)/1048576 < 3000 )) && break; sleep 2; done
  log "=== $label  ts=$ts ctx=$ctx n=$nmax dkv=$dkv"
  echo "===== $label =====" >> "$SP/server.log"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c "$ctx" --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" \
    --spec-draft-type-k "$dkv" --spec-draft-type-v "$dkv" \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!; watch_ "$MY_PID" & W=$!
  local ok=0
  for i in $(seq 1 500); do
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  if (( ok==0 )); then log "  DID NOT LOAD peak=$(cat $SP/.peak)"; mykill; return 1; fi
  log "  loaded peak $(cat $SP/.peak) MiB"
  python3 "$SP/blockA.py" "$label" "120000" "$reps" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "  $label done peak $(cat $SP/.peak) MiB"; mykill; }

log "########## PHASE A: split ratio at n4 ##########"
run A_r48_n4 48,52 262144 4 f16 8
run A_r50_n4 50,50 262144 4 f16 8
log "########## PHASE B: draft depth ##########"
for N in 2 3 5; do run B_r48_n$N 48,52 262144 $N f16 8; done
log "########## PHASE C: draft KV type ##########"
run C_r48_n4_q8d 48,52 262144 4 q8_0 8
log "########## PHASE D: context trade for XTX-heavy ratio ##########"
run D_r51_c245k 51,49 245760 4 f16 8
run D_r52_c229k 52,48 229376 4 f16 8
log "########## OPTIMIZATION COMPLETE ##########"
