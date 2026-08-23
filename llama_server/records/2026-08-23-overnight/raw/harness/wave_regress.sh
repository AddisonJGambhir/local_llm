#!/usr/bin/env bash
# Why is tensor prefill 393 t/s when MASTER_RESULTS records 940.78 for
# Vulkan tensor 50/50 at d0? Test the two candidate causes:
#   (a) --main-gpu 1 (shipped preset) vs --main-gpu 0
#   (b) tp-adaptive regressed vs the vulkan-tp build it was derived from
# NOTE: -ts takes SLASHES. Commas make llama-bench sweep values, which silently
# produced single-GPU runs in every earlier tensor attempt tonight.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
X=/sys/class/drm/card1/device/mem_info_vram_used
OUT="$SP/regress.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "build,split,ts,main_gpu,ub,b,pp,peak_xtx" > "$OUT"
wait_free(){ for i in $(seq 1 90); do (( $(cat $X)/1048576 < 3500 )) && return 0; sleep 2; done; }
cell(){ local bname="$1" bd="$2" sm="$3" ts="$4" mg="$5" ub="$6" b="$7"
  wait_free
  local extra=""; [ "$ts" != "-" ] && extra="--tensor-split $ts --main-gpu $mg"
  log "regress: $bname $sm ts=$ts mg=$mg ub=$ub b=$b"
  ( LD_LIBRARY_PATH="$bd" "$bd/llama-bench" -m "$MODEL" -ngl 99 -ub "$ub" -b "$b" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode "$sm" $extra \
      -p 8192 -n 0 -r 3 -o csv > "$SP/.rg" 2>>"$SP/run.log" ) &
  local p=$!; local mx=0 over=0
  while kill -0 $p 2>/dev/null; do
    local u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    if (( u > 24000 )); then over=$((over+1)); else over=0; fi
    (( over >= 4 )) && { log "  ABORT >24000"; kill -9 $p; break; }
    sleep 0.5
  done
  wait $p 2>/dev/null
  local pp=$(python3 -c "
import csv
for r in csv.DictReader(open('$SP/.rg')):
    if r['n_prompt'] not in ('0',''): print(f\"{float(r['avg_ts']):.2f}\"); break
" 2>/dev/null)
  echo "$bname,$sm,$ts,$mg,$ub,$b,${pp:-FAIL},$mx" >> "$OUT"
  log "  -> pp=${pp:-FAIL}  peak=$mx"
}
A=$LLM/llama.cpp-tp-adaptive/build/bin
V=$LLM/llama.cpp-vulkan-tp/build/bin
log "REGRESSION PROBE START (target: explain 393 vs recorded 940.78)"
cell tp-adaptive "$A" tensor 50/50 1 1024 2048
cell tp-adaptive "$A" tensor 50/50 0 1024 2048
cell tp-adaptive "$A" row    -     0 1024 2048
if [ -x "$V/llama-bench" ]; then
  cell vulkan-tp "$V" tensor 50/50 1 1024 2048
  cell vulkan-tp "$V" tensor 50/50 0 1024 2048
else
  log "vulkan-tp llama-bench not built - skipping cross-build comparison"
fi
log "REGRESSION PROBE COMPLETE"
