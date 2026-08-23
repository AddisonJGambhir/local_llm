#!/usr/bin/env bash
# n4+f16 is +7.3% over n3+f16 at 120k, and the depth ranking inverts vs d0.
# wave2 only sweeps n5/n2 with q8_0 draft KV, but f16 is the better draft KV.
# Find where the fixed-depth optimum actually sits, at working depth, with f16.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"; source "$SP/lib.sh"
CSV="$SP/depth_ab.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
F16="--spec-draft-type-k f16 --spec-draft-type-v f16"
phase(){ local label="$1"; shift
  log "=== N5WAVE $label"; echo "===== $label =====" >> "$SP/server.log"
  start_server "$COMMON $*" || { log "$label FAILED TO START"; mykill; return 1; }
  python3 "$SP/blockA.py" "$label" "120000" 15 "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "$label done peak $(cat $SP/.peak) MiB"; mykill; }
log "N5 WAVE START (fixed-depth optimum at 120k with f16 draft KV)"
phase mtp_n5_f16draft "--spec-type draft-mtp --spec-draft-n-max 5 $F16"
phase mtp_n6_f16draft "--spec-type draft-mtp --spec-draft-n-max 6 $F16"
phase mtp_n8_f16draft "--spec-type draft-mtp --spec-draft-n-max 8 $F16"
log "N5 WAVE COMPLETE"
bash "$SP/wave_m5.sh"
