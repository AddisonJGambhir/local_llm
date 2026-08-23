#!/usr/bin/env bash
# Raw depth curve + ubatch sweep. llama-bench has NO -c; context is derived
# from -p + -n + -d, which is why the first attempt died instantly.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23500
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
run(){ local tag="$1"; shift
  log "bench: $tag"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
      -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split 49,51 "$@" \
      -o csv > "$SP/bench_$tag.csv" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  log "  $tag rc=$rc peak $(cat $SP/.peak) MiB"
  [ -s "$SP/bench_$tag.csv" ] && python3 -c "
import csv,sys
for r in csv.DictReader(open('$SP/bench_$tag.csv')):
    print('   ', r.get('n_prompt'),'/',r.get('n_gen'),'d'+str(r.get('n_depth','0')),
          '->', round(float(r['avg_ts']),2),'t/s', flush=True)" 2>/dev/null
}
log "WAVE BENCH START"
run depthcurve -b 2048 -ub 1024 -p 1024 -n 256 -d 0,65536,131072,196608 -r 3
run ub1024     -b 2048 -ub 1024 -p 8192 -n 256 -r 3
run ub2048     -b 2048 -ub 2048 -p 8192 -n 256 -r 3
log "WAVE BENCH COMPLETE"
bash "$SP/wave_pp2.sh"
bash "$SP/wave_cross.sh"
