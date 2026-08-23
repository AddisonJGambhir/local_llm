#!/usr/bin/env bash
# llama-bench spikes to 24.4 GiB on tensor 49/51 regardless of batch size.
# Shift weights onto the R9700 (32 GiB, 18 GiB free) until it fits, THEN sweep batch.
# Step from safest ratio upward, recording peak, so we never approach the line blind.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
OUT="$SP/tensor_pp.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23400
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "split,ratio,ub,b,depth,pp,tg,peak_xtx_mib,status" > "$OUT"
wait_vram_free(){ local i u
  for i in $(seq 1 90); do u=$(( $(cat $XTXF)/1048576 ))
    (( u < 3500 )) && return 0; sleep 2; done; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 1 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.25; done; echo "$mx" > "$SP/.peak"; }
cell(){ local ts="$1" ub="$2" b="$3" dep="$4"
  wait_vram_free
  log "ratio-pp: ts=$ts ub=$ub b=$b d=$dep"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
      -p 8192 -n 128 -d "$dep" -r 3 -o csv > "$SP/.rpp" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak")
  if (( rc != 0 )); then
    echo "tensor,${ts/,//},$ub,$b,$dep,,,$pk,CEILING" >> "$OUT"; log "  too big: peak $pk"; return 1; fi
  python3 - "$SP/.rpp" "${ts/,//}" "$ub" "$b" "$dep" "$pk" "$OUT" <<'PY'
import csv,sys
pp=tg=""
for r in csv.DictReader(open(sys.argv[1])):
    v=float(r["avg_ts"])
    if r.get("n_prompt","0") not in ("0",""): pp=f"{v:.2f}"
    else: tg=f"{v:.2f}"
line=f"tensor,{sys.argv[2]},{sys.argv[3]},{sys.argv[4]},{sys.argv[5]},{pp},{tg},{sys.argv[6]},OK"
open(sys.argv[7],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}
log "TENSOR RATIO PROBE START (find a ratio that fits, then sweep batch)"
# phase 1: how far must weights move to the R9700? baseline batch, d0
for TS in 25,75 30,70 35,65 40,60; do
  cell "$TS" 1024 2048 0 || { log "  -> $TS does not fit, stopping probe"; break; }
done
log "TENSOR RATIO PROBE COMPLETE"
