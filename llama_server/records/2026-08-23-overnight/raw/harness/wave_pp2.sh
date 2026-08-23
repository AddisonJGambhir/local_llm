#!/usr/bin/env bash
# Refine the prefill optimum. Row split only (it won by a wide margin and stays
# at ~13.5 GiB; the layer/tensor cells at b4096 hit the VRAM guard at 24.5 GiB).
# ub2048/b4096 = 1257.21 is the current record; ub4096 collapses to 937.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
OUT="$SP/pp_record.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23500
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
cell(){ local bn="$1" bd="$2" ub="$3" b="$4" pn="$5"
  log "pp2 cell: ub=$ub b=$b p=$pn"
  ( LD_LIBRARY_PATH="$bd" "$bd/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode row \
      -p "$pn" -n 0 -r 3 -o csv > "$SP/.pp2" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak" 2>/dev/null)
  (( rc != 0 )) && { log "  FAILED rc=$rc peak=$pk"; return; }
  python3 - "$SP/.pp2" "$bn" "$ub" "$b" "$pn" "$pk" "$OUT" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[float(r["avg_ts"]) for r in rows if r.get("n_prompt","0") not in ("0","")]
if v:
    line="%s,row,-,%s,%s,p%s,%.2f,%.2f,%s"%(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],st.mean(v),st.pstdev(v),sys.argv[6])
    open(sys.argv[7],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}
A=$LLM/llama.cpp-tp-adaptive/build/bin
log "PP REFINE START (best so far 1257.21 @ ub2048 b4096)"
for UB in 1536 2048 3072; do
  for B in 2048 4096; do
    (( B < UB )) && continue
    cell shipped "$A" $UB $B 8192
  done
done
# does a longer prompt amortize better?
cell shipped "$A" 2048 4096 16384
log "PP REFINE COMPLETE"
