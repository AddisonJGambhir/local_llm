#!/usr/bin/env bash
# Does the batch-size prefill win transfer to LAYER split?
# row: +33% d0 / +62% d65k. tensor: completely flat (all 6 combos within 2%).
# Layer splits at layer boundaries like row does, so it plausibly behaves like row.
# NOTE: -ts takes SLASHES in llama-bench; commas are a value sweep.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
X=/sys/class/drm/card1/device/mem_info_vram_used
OUT="$SP/layer_pp.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "split,ts,ub,b,depth,pp,peak_xtx" > "$OUT"
wait_free(){ for i in $(seq 1 90); do (( $(cat $X)/1048576 < 3500 )) && return 0; sleep 2; done; }
run(){ local ts="$1" dep="$2"
  wait_free
  log "layer-pp: ts=$ts d=$dep (sweeping ub/b)"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 \
      -ub 512,1024 -b 2048,8192 -fa 1 -ctk f16 -ctv f16 -mmp 0 \
      --split-mode layer --tensor-split "$ts" \
      -p 8192 -n 0 -d "$dep" -r 3 -o csv > "$SP/.lp" 2>>"$SP/run.log" ) &
  local p=$!; local mx=0 over=0
  while kill -0 $p 2>/dev/null; do
    local u=$(( $(cat $X)/1048576 )); (( u > mx )) && mx=$u
    if (( u > 24000 )); then over=$((over+1)); else over=0; fi
    (( over >= 4 )) && { log "  ABORT >24000"; kill -9 $p; break; }
    sleep 0.5
  done
  wait $p 2>/dev/null
  python3 - "$SP/.lp" "$ts" "$dep" "$mx" "$OUT" <<'PY'
import csv,sys
n=0
for r in csv.DictReader(open(sys.argv[1])):
    if r["n_prompt"] in ("0",""): continue
    line=f"layer,{sys.argv[2]},{r['n_ubatch']},{r['n_batch']},{sys.argv[3]},{float(r['avg_ts']):.2f},{sys.argv[4]}"
    open(sys.argv[5],"a").write(line+"\n"); print("  ->",line,flush=True); n+=1
print(f"  ({n} cells)")
PY
}
log "LAYER PREFILL SWEEP START"
run 28/37 0
run 28/37 65536
log "LAYER PREFILL SWEEP COMPLETE"
