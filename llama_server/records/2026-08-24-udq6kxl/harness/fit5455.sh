#!/usr/bin/env bash
# >=50% on the XTX only. Two questions left:
#   1. raw decode at DEPTH (d65k) for 50/50 .. 54/46 -- d0 said more XTX = faster,
#      confirm it holds once KV is large
#   2. which of these actually fit at c=262144 f16 KV. Earlier single readings were
#      non-monotonic (51/49 24,251 vs 52/48 24,067), so measure each TWICE.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
export MTMD_BACKEND_DEVICE=Vulkan1
RD="$SP/rawdecode.csv"; VR="$SP/vram50plus.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$VR" ] || echo "ratio,pass,settled_xtx,loaded" > "$VR"
idle(){ for i in $(seq 1 90); do (( $(cat $X)/1048576 < 2600 )) && break; sleep 2; done; }
# ---------- part 1: raw decode at depth ----------
cell(){ local ts="$1" dep="$2"
  idle; log "raw decode ts=$ts d=$dep"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b 2048 -ub 1024 \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
      -p 0 -n 256 -d "$dep" -r 3 -o csv > "$SP/.rd2" 2>>"$SP/run.log" ) &
  local p=$!; local mx=0 over=0 u
  while kill -0 $p 2>/dev/null; do u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    if (( u > 24200 )); then over=$((over+1)); else over=0; fi
    (( over >= 10 )) && { log "  ABORT $u"; kill -9 $p; break; }; sleep 0.5; done
  wait $p 2>/dev/null
  local tg=$(python3 -c "
import csv
for r in csv.DictReader(open('$SP/.rd2')):
    if r['n_gen'] not in ('0',''): print(f\"{float(r['avg_ts']):.2f}\"); break" 2>/dev/null)
  echo "${ts//\//-},$dep,${tg:-FAIL},$mx" >> "$RD"
  log "  -> tg=${tg:-FAIL} peak=$mx"; }
log "SKIP raw decode"

# ---------- part 2: does it fit at 262144, measured twice ----------
MY_PID=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 45); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""; idle; }
settle_read(){ local last=-999 same=0 cur
  for i in $(seq 1 120); do cur=$(( $(cat $X)/1048576 ))
    if (( cur > last-8 && cur < last+8 )); then same=$((same+1)); else same=0; fi
    (( same >= 12 )) && { echo "$cur"; return 0; }; last=$cur; sleep 1; done; echo "$cur"; }
fit(){ local ts="$1" pass="$2"
  mykill; log "fit-check $ts pass$pass"
  LD_LIBRARY_PATH="$BD" setsid "$BD/llama-server" -m "$MODEL" --alias local \
    --device Vulkan0,Vulkan1 -ngl 99 -fit off -t 16 -tb 32 --parallel 1 \
    --flash-attn on --no-mmap --no-context-shift --ctx-checkpoints 8 \
    --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
    -c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 \
    --batch-size 2048 --cache-ram 24576 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 \
    --mmproj "$LLM/llama_server/models/mmproj-BF16.gguf" \
    --host 127.0.0.1 --port 1234 >> "$SP/server.log" 2>&1 &
  MY_PID=$!
  local ok=0 over=0 u
  for i in $(seq 1 900); do kill -0 "$MY_PID" 2>/dev/null || break
    u=$(( $(cat $X)/1048576 ))
    if (( u > 24200 )); then over=$((over+1)); else over=0; fi
    (( over >= 10 )) && { log "  sustained $u > 24200"; kill -9 "$MY_PID"; break; }
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    sleep 1; done
  local sx=0
  if (( ok==1 )); then
    curl -s -m 240 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Explain memory bandwidth.","n_predict":192,"temperature":0}' >/dev/null 2>&1
    sx=$(settle_read); log "  $ts pass$pass SETTLED=$sx"
  else log "  $ts pass$pass FAILED"; fi
  echo "${ts//\//-},$pass,$sx,$ok" >> "$VR"; mykill; }
log "FIT CHECK AT 262144 (twice each)"
for P in 1 2; do for TS in 54,46 55,45; do fit "$TS" "$P"; done; done
log "XTX50PLUS COMPLETE"
