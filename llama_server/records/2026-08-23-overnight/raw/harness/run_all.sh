#!/usr/bin/env bash
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
source "$SP/lib.sh"
CSV="$SP/depth_ab.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT

COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
DEPTHS="0,60000,120000"
REPS=7

phase(){ # phase <label> <spec args>
  local label="$1"; shift
  log "=== PHASE $label : starting server"
  echo "===== $label =====" >> "$SP/server.log"
  if ! start_server "$COMMON $*"; then
    log "PHASE $label FAILED TO START (see server.log)"; mykill; return 1
  fi
  log "PHASE $label up (load peak $(cat $SP/.peak 2>/dev/null) MiB) - running depths $DEPTHS x $REPS"
  python3 "$SP/blockA.py" "$label" "$DEPTHS" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "PHASE $label done, peak XTX $(cat $SP/.peak 2>/dev/null) MiB"
  mykill
}

log "OVERNIGHT RUN START"
phase adaptive_n12_q8draft  "--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase mtp_n3_q8draft        "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase adaptive_n12_f16draft "--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k f16 --spec-draft-type-v f16"
log "DEPTH A/B COMPLETE -> $CSV"

# ---- Block B: raw depth curve, no speculation (bounds the above) ----
log "=== BLOCK B: llama-bench raw depth curve"
( "$BENCH" -m "$MODEL" -ngl 99 -c 262144 -b 2048 -ub 1024 -fa 1 -ctk f16 -ctv f16 \
    --no-mmap --split-mode tensor --main-gpu 1 --tensor-split 49,51 \
    -p 1024 -n 256 -d 0,65536,131072,196608 -r 3 -o csv > "$SP/blockB.csv" 2>>"$SP/run.log" ) &
BP=$!; MY_PID=$BP; peakwatch $BP & GUARD_PID=$!
wait $BP; log "BLOCK B done rc=$? peak $(cat $SP/.peak 2>/dev/null) MiB"
MY_PID=""; [ -n "$GUARD_PID" ] && kill $GUARD_PID 2>/dev/null; GUARD_PID=""

# ---- Block D1: ubatch 1024 vs 2048 at c=98304 ----
log "=== BLOCK D1: ubatch sweep at c=98304"
for UB in 1024 2048; do
  ( "$BENCH" -m "$MODEL" -ngl 99 -c 98304 -b 2048 -ub $UB -fa 1 -ctk f16 -ctv f16 \
      --no-mmap --split-mode tensor --main-gpu 1 --tensor-split 49,51 \
      -p 8192 -n 256 -r 3 -o csv > "$SP/blockD1_ub$UB.csv" 2>>"$SP/run.log" ) &
  BP=$!; MY_PID=$BP; peakwatch $BP & GUARD_PID=$!
  wait $BP; log "D1 ub$UB done peak $(cat $SP/.peak 2>/dev/null) MiB"
  MY_PID=""; [ -n "$GUARD_PID" ] && kill $GUARD_PID 2>/dev/null; GUARD_PID=""
done
log "OVERNIGHT RUN COMPLETE"
