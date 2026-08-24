#!/usr/bin/env bash
# RAW DECODE across split ratios. No speculation -- llama-bench does not run the
# MTP draft path at all, which is exactly what we want: draft depth is not the
# thing being optimised, and at IRL acceptance (0.5-0.6) it barely moves tg.
# NOTE: llama-bench -ts takes SLASHES; commas make it sweep values as separate runs.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
OUT="$SP/rawdecode.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "ratio,depth,tg,peak_xtx" > "$OUT"
cell(){ local ts="$1" dep="$2"
  for i in $(seq 1 90); do (( $(cat $X)/1048576 < 2600 )) && break; sleep 2; done
  log "raw decode ts=$ts d=$dep"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b 2048 -ub 1024 \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 \
      --split-mode tensor --main-gpu 1 --tensor-split "$ts" \
      -p 0 -n 256 -d "$dep" -r 3 -o csv > "$SP/.rd" 2>>"$SP/run.log" ) &
  local p=$!; local mx=0 over=0 u
  while kill -0 $p 2>/dev/null; do
    u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    if (( u > 24200 )); then over=$((over+1)); else over=0; fi
    (( over >= 10 )) && { log "  ABORT sustained $u"; kill -9 $p; break; }
    sleep 0.5
  done
  wait $p 2>/dev/null
  local tg=$(python3 -c "
import csv
for r in csv.DictReader(open('$SP/.rd')):
    if r['n_gen'] not in ('0',''): print(f\"{float(r['avg_ts']):.2f}\"); break
" 2>/dev/null)
  echo "${ts//\//-},$dep,${tg:-FAIL},$mx" >> "$OUT"
  log "  -> tg=${tg:-FAIL}  peak=$mx"
}
log "RAW DECODE vs SPLIT RATIO (no speculation)"
for TS in 45/55 47/53 49/51 50/50 51/49 52/48; do cell "$TS" 0; done
log "--- now at depth 65536 ---"
for TS in 49/51 50/50 51/49; do cell "$TS" 65536; done
log "RAW DECODE COMPLETE"
column -s, -t "$OUT"
