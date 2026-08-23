#!/usr/bin/env bash
# M5: the three GGML_VK_* env vars the ~45-knob null table never covered.
# First attempt died instantly: launched into 22.9 GiB of not-yet-freed VRAM from
# the previous server, so the ceiling guard killed every cell in under 2 seconds.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
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
run(){ local tag="$1"
  wait_vram_free
  log "M5: $tag"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b 4096 -ub 1024 \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode row \
      -p 4096 -n 256 -r 3 -o csv > "$SP/m5_$tag.csv" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  if (( rc==0 )) && [ -s "$SP/m5_$tag.csv" ]; then
    python3 -c "
import csv
for r in csv.DictReader(open('$SP/m5_$tag.csv')):
    k='pp' if r.get('n_prompt','0') not in ('0','') else 'tg'
    print(f'   $tag {k} {float(r[\"avg_ts\"]):.2f} t/s', flush=True)"
  else log "  $tag FAILED rc=$rc peak=$(cat $SP/.peak)"; fi
}
log "M5 WAVE START (untested Vulkan env vars)"
run baseline
GGML_VK_DISABLE_FUSION=1    run disable_fusion
GGML_VK_DISABLE_COOPMAT=1   run disable_coopmat
GGML_VK_DISABLE_MULTI_ADD=1 run disable_multi_add
log "M5 WAVE COMPLETE"
