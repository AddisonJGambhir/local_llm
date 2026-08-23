#!/usr/bin/env bash
# Push below ub1024: pp is monotone DECREASING in ubatch once b=4096.
# it was running -- bash reads scripts by byte offset, so the insert desynced it).
# The cell that survived CONFIRMED the record: 1353.06 t/s at ub1536/b4096, r=5.
# These are the neighbours plus the ub1024/b4096 control that separates
# "b4096 is the win" from "ub1536 is specifically special".
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
OUT="$SP/pp_record.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23500
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
wait_vram_free(){ local i u
  for i in $(seq 1 90); do u=$(( $(cat $XTXF)/1048576 ))
    (( u < 3500 )) && { log "vram settled at ${u} MiB"; return 0; }; sleep 2; done
  log "WARNING vram still ${u} MiB"; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
cell(){ local ub="$1" b="$2" r="$3"
  log "pp4: ub=$ub b=$b r=$r"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode row \
      -p 8192 -n 0 -r "$r" -o csv > "$SP/.pp6" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak")
  (( rc != 0 )) && { log "  FAILED rc=$rc peak=$pk"; return; }
  python3 - "$SP/.pp6" "$ub" "$b" "$pk" "$OUT" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[float(r["avg_ts"]) for r in rows if r.get("n_prompt","0") not in ("0","")]
if v:
    line="pp6,row,-,%s,%s,p8192,%.2f,%.2f,%s"%(sys.argv[2],sys.argv[3],st.mean(v),st.pstdev(v),sys.argv[4])
    open(sys.argv[5],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}

celld(){ local ub="$1" b="$2" dep="$3" r="$4"
  log "pp5-depth: ub=$ub b=$b d=$dep r=$r"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode row \
      -p 8192 -n 0 -d "$dep" -r "$r" -o csv > "$SP/.pp6" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak")
  (( rc != 0 )) && { log "  FAILED rc=$rc peak=$pk"; return; }
  python3 - "$SP/.pp6" "$ub" "$b" "$dep" "$pk" "$OUT" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[float(r["avg_ts"]) for r in rows if r.get("n_prompt","0") not in ("0","")]
if v:
    line="pp6d,row,-,%s,%s,d%s,%.2f,%.2f,%s"%(sys.argv[2],sys.argv[3],sys.argv[4],st.mean(v),st.pstdev(v),sys.argv[5])
    open(sys.argv[6],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}

wait_vram_free
log "PP6 START (record now 1648.96 @ ub1024 b8192)"
cell  512 8192 3
cell 1024 16384 3
cell  512 16384 3
cell 2048 16384 3
cell 1024 8192 5
# does the b4096 win survive at working depth? the record is a d0 number.
celld 1024 8192 65536 3
celld 1024 16384 65536 3
celld  512 8192 65536 3
log "PP6 COMPLETE"
