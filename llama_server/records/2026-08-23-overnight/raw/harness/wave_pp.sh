#!/usr/bin/env bash
# PREFILL RECORD ATTEMPT. Target to beat: 1240.47 t/s (Q8_0, vulkan-pr, row split,
# d0, peak 13,346 MiB). llama-bench, no speculation (method rule: llama-bench does
# not run the MTP draft path, so it is the right tool for raw pp ranking).
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
OUT="$SP/pp_record.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used
CEIL=24000
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "build,split,ratio,ub,b,ctx,pp_mean,pp_sd,peak_xtx_mib" > "$OUT"

BP=""; W=""
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 4 )) && { echo "CEILING $u -> kill bench $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 1; done; echo "$mx" > "$SP/.peak"; }

# cell <buildname> <bindir> <splitmode> <ratio|-> <ub> <b> <ctx>
cell(){
  local bn="$1" bd="$2" sm="$3" ts="$4" ub="$5" b="$6" cx="$7"
  local tscsv="${ts//,//}"   # 28,37 -> 28/37 so it does not split the csv column
  local extra=""; [ "$ts" != "-" ] && extra="--tensor-split $ts --main-gpu 1"
  log "pp cell: $bn $sm $ts ub=$ub b=$b c=$cx"
  ( LD_LIBRARY_PATH="$bd" "$bd/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode "$sm" $extra \
      -p 8192 -n 0 -r 3 -o csv > "$SP/.ppcell" 2>>"$SP/run.log" ) &
  BP=$!; peakwatch $BP & W=$!
  wait $BP; local rc=$?
  [ -n "$W" ] && kill $W 2>/dev/null
  local pk=$(cat "$SP/.peak" 2>/dev/null)
  if (( rc != 0 )); then echo "$bn,$sm,$tscsv,$ub,$b,$cx,,,$pk" >> "$OUT"; log "  FAILED rc=$rc peak=$pk"; return; fi
  python3 - "$SP/.ppcell" "$bn" "$sm" "$tscsv" "$ub" "$b" "$cx" "$pk" "$OUT" <<'PY'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[float(r["avg_ts"]) for r in rows if r.get("n_prompt","0") not in ("0","")]
if v:
    line="%s,%s,%s,%s,%s,%s,%.2f,%.2f,%s"%(*sys.argv[2:8], st.mean(v), st.pstdev(v), sys.argv[8])
    open(sys.argv[9],"a").write(line+"\n"); print("  ->", line, flush=True)
PY
}

A=$LLM/llama.cpp-tp-adaptive/build/bin
C=$LLM/llama.cpp-tp-chain/build/bin
log "PP RECORD RUN START (target to beat: 1240.47 t/s)"
# 1) reproduce the record config
cell shipped "$A" row    -     1024 2048 32768
# 2) does a bigger ubatch/batch buy prefill at small ctx (VRAM is free there)?
cell shipped "$A" row    -     2048 4096 32768
cell shipped "$A" row    -     4096 4096 32768
# 3) layer split, same sweep (it was within 1% of row)
cell shipped "$A" layer  28,37 2048 4096 32768
cell shipped "$A" layer  28,37 4096 4096 32768
# 4) tensor split for reference (wins decode, loses pp at d0)
cell shipped "$A" tensor 50,50 2048 4096 32768
# 5) chain build: sched pool + split memoization are prefill-relevant too
cell chain   "$C" row    -     2048 4096 32768
cell chain   "$C" row    -     4096 4096 32768
log "PP RECORD RUN COMPLETE -> $OUT"
bash "$SP/wave_ceiling.sh"
