#!/usr/bin/env bash
# Does the prefill win transfer to TENSOR split? The +33%/+62% was all measured on
# --split-mode row. The shipped decode preset is -sm tensor 49/51, which allocates
# prefill buffers very differently -- an earlier tensor cell at b4096 hit 24,343 MiB
# and was killed. Step batch up carefully and watch VRAM.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
LLM=/home/addison-gambhir/Desktop/local_llm
MODEL=$LLM/llama_server/models/Qwen3.8-27B-Q8_0.gguf
BD=$LLM/llama.cpp-tp-adaptive/build/bin
OUT="$SP/tensor_pp.csv"
XTXF=/sys/class/drm/card1/device/mem_info_vram_used; CEIL=23900
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
[ -f "$OUT" ] || echo "split,ratio,ub,b,depth,pp,tg,peak_xtx_mib,status" > "$OUT"
wait_vram_free(){ local i u
  for i in $(seq 1 90); do u=$(( $(cat $XTXF)/1048576 ))
    (( u < 3500 )) && { log "vram settled ${u} MiB"; return 0; }; sleep 2; done; }
peakwatch(){ local pid=$1 mx=0 over=0 u; echo 0 > "$SP/.peak"
  while kill -0 "$pid" 2>/dev/null; do u=$(( $(cat $XTXF)/1048576 ))
    (( u > mx )) && { mx=$u; echo "$mx" > "$SP/.peak"; }
    if (( u > CEIL )); then over=$((over+1)); else over=0; fi
    (( over >= 2 )) && { echo "CEILING $u -> kill $pid" >> "$SP/run.log"; kill -9 "$pid"; break; }
    sleep 0.5; done; echo "$mx" > "$SP/.peak"; }
cell(){ local ub="$1" b="$2" dep="$3"
  wait_vram_free
  log "tensor-pp: ub=$ub b=$b d=$dep"
  ( LD_LIBRARY_PATH="$BD" "$BD/llama-bench" -m "$MODEL" -ngl 99 -b "$b" -ub "$ub" \
      -fa 1 -ctk f16 -ctv f16 -mmp 0 --split-mode tensor --main-gpu 1 --tensor-split 49,51 \
      -p 8192 -n 128 -d "$dep" -r 3 -o csv > "$SP/.tpp" 2>>"$SP/run.log" ) &
  local p=$!; peakwatch $p & local w=$!
  wait $p; local rc=$?; kill $w 2>/dev/null
  local pk=$(cat "$SP/.peak")
  if (( rc != 0 )); then
    echo "tensor,49/51,$ub,$b,$dep,,,$pk,CEILING_OR_FAIL" >> "$OUT"
    log "  FAILED rc=$rc peak=$pk  <-- does not fit"; return; fi
  python3 - "$SP/.tpp" "$ub" "$b" "$dep" "$pk" "$OUT" <<'PY'
import csv,sys
pp=tg=""
for r in csv.DictReader(open(sys.argv[1])):
    v=float(r["avg_ts"])
    if r.get("n_prompt","0") not in ("0",""): pp=f"{v:.2f}"
    else: tg=f"{v:.2f}"
line=f"tensor,49/51,{sys.argv[2]},{sys.argv[3]},{sys.argv[4]},{pp},{tg},{sys.argv[5]},OK"
open(sys.argv[6],"a").write(line+"\n"); print("  ->",line,flush=True)
PY
}
log "TENSOR PREFILL TRANSFER TEST START"
# baseline first, then step batch up; ub512 and ub1024 at each
for D in 0 65536; do
  cell 1024 2048 $D      # current preset setting
  cell  512 4096 $D
  cell  512 8192 $D      # the row-split winner
  cell 1024 8192 $D
done
log "TENSOR PREFILL TRANSFER TEST COMPLETE"
