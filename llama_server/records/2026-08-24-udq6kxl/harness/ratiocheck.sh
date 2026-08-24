#!/usr/bin/env bash
# Careful isolated sizing. Prior numbers were non-physical: Q6 at 48/52 = 20,279
# but 50/50 = 23,638 and 51/49 = 24,552 -- a 2.6 GiB jump per ratio point, while
# Q8_0 (3.5 GiB LARGER) runs 49/51 at ~23,000. Something in the earlier probe was
# wrong. This waits for a true idle baseline between every run and logs the
# allocation llama.cpp itself reports, not just sysfs.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
R=/sys/class/drm/card0/device/mem_info_vram_used
export MTMD_BACKEND_DEVICE=Vulkan1
OUT="$SP/ratiocheck.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "ratio;base_xtx;peak_xtx;steady_xtx;peak_r9700;loaded" > "$OUT"
MY_PID=""
mykill(){ [ -z "$MY_PID" ] && return 0; kill "$MY_PID" 2>/dev/null
  for i in $(seq 1 45); do kill -0 "$MY_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$MY_PID" 2>/dev/null; wait "$MY_PID" 2>/dev/null; MY_PID=""; }
trap 'mykill' EXIT
settle(){ local i last=-1 cur
  for i in $(seq 1 90); do
    cur=$(( $(cat $X)/1048576 ))
    [ "$cur" = "$last" ] && (( cur < 2600 )) && { echo "$cur"; return 0; }
    last=$cur; sleep 2
  done; echo "$cur"; }
one(){ local ts="$1"
  mykill
  local base=$(settle)
  log "ratio $ts  (idle baseline ${base} MiB)"
  echo "===== ratiocheck $ts =====" >> "$SP/server.log"
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
  local mx=0 mr=0 ok=0 u v
  for i in $(seq 1 600); do
    kill -0 "$MY_PID" 2>/dev/null || break
    u=$(( $(cat $X)/1048576 )); v=$(( $(cat $R)/1048576 ))
    (( u > mx )) && mx=$u; (( v > mr )) && mr=$v
    (( u > 24100 )) && { log "  HARD ABORT ${u}"; kill -9 "$MY_PID"; break; }
    curl -s -m 2 http://127.0.0.1:1234/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    sleep 0.5
  done
  local st=0
  if (( ok==1 )); then
    curl -s -m 240 http://127.0.0.1:1234/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Explain GPU memory bandwidth in one paragraph.","n_predict":256,"temperature":0}' >/dev/null 2>&1
    st=$(( $(cat $X)/1048576 )); u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    log "  $ts LOADED  peak=${mx} steady=${st} r9700=${mr}"
  else
    log "  $ts FAILED  peak=${mx}"
  fi
  echo "$ts;$base;$mx;$st;$mr;$ok" >> "$OUT"
  mykill
}
log "ISOLATED RATIO CHECK (Q6, c=262144, f16 KV)"
for TS in 48,52 49,51 50,50 51,49; do one "$TS"; done
log "ISOLATED RATIO CHECK COMPLETE"
column -s';' -t "$OUT"
