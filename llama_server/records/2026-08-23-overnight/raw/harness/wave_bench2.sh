#!/usr/bin/env bash
# Raw depth curve, no speculation. ONE llama-bench invocation PER depth --
# passing -d 0,65536,131072,196608 together makes it allocate for the largest
# depth up front, which blew the VRAM ceiling. 196608 does not fit at 50/50.
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
log "DEPTH CURVE START (raw, no speculation)"
for D in 0 32768 65536 98304 131072; do
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b 2048 -ub 1024 \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split 50,50 \
      -p 512 -n 256 -d $D -r 3 -o csv > "$SP/bench_d$D.csv" 2>>"$SP/run.log" ) &
  p=$!; peakwatch $p & w=$!
  wait $p; rc=$?; kill $w 2>/dev/null
  pk=$(cat $SP/.peak)
  if (( rc==0 )) && [ -s "$SP/bench_d$D.csv" ]; then
    python3 -c "
import csv
for r in csv.DictReader(open('$SP/bench_d$D.csv')):
    kind='pp' if r.get('n_prompt','0') not in ('0','') else 'tg'
    print(f\"  d$D {kind} {float(r['avg_ts']):.2f} t/s  peak $pk MiB\", flush=True)"
  else log "  d$D FAILED rc=$rc peak=$pk"; fi
done
log "DEPTH CURVE COMPLETE"
bash "$SP/wave_pp3.sh"
