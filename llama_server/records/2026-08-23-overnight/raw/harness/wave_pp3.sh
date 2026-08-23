#!/usr/bin/env bash
# Confirm and refine the 1369.93 t/s prefill peak at ub1536/b4096 (+10.4%).
# The pp curve is non-monotone; every prior sweep used powers of two and stepped
# over 1536. More reps here because this is a record claim.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
OUT="$SP/pp_record.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23500
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
cell(){ local ub="$1" b="$2" r="$3"
  
# Wait for the previous wave's VRAM to actually be released. These waves are
# chained by appending to script tails, so they start immediately after the
# previous server exits -- and a llama-bench launched into 19 GiB of not-yet-freed
# VRAM trips the ceiling guard in under 2 seconds. This cost the first depth curve.
wait_vram_free(){
  local f=/sys/class/drm/card1/device/mem_info_vram_used i u
  for i in $(seq 1 90); do
    u=$(( $(cat $f)/1048576 ))
    (( u < 3500 )) && { echo "[$(date +%H:%M:%S)] vram settled at ${u} MiB" >> "$SP/run.log"; return 0; }
    sleep 2
  done
  echo "[$(date +%H:%M:%S)] WARNING: vram still ${u} MiB after 180s" >> "$SP/run.log"
}
wait_vram_free

log "pp3: ub=$ub b=$b r=$r"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode row \
      -p 8192 -n 0 -r "$r" -o csv > "$SP/.pp3" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak")
  (( rc != 0 )) && { log "  FAILED rc=$rc peak=$pk"; return; }
  python3 - "$SP/.pp3" "$ub" "$b" "$pk" "$OUT" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[float(r["avg_ts"]) for r in rows if r.get("n_prompt","0") not in ("0","")]
if v:
    line="confirm,row,-,%s,%s,p8192,%.2f,%.2f,%s"%(sys.argv[2],sys.argv[3],st.mean(v),st.pstdev(v),sys.argv[4])
    open(sys.argv[5],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}
log "PP CONFIRM START (claim: 1369.93 @ ub1536 b4096)"
cell 1536 4096 5        # the claim, more reps
cell 1280 4096 3
cell 1792 4096 3
cell 1536 6144 3
cell 1536 8192 3
cell 1024 4096 3        # is b4096 alone the win, or is 1536 special?
log "PP CONFIRM COMPLETE"
bash "$SP/wave_cross2.sh"
