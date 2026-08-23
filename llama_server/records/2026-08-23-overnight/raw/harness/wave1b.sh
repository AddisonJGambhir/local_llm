#!/usr/bin/env bash
# Wave 1b: top up the DECISION cell (120k) to n=22 per config. Appends to depth_ab.csv.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"; source "$SP/lib.sh"
CSV="$SP/depth_ab.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
phase(){ local label="$1"; shift
  log "=== WAVE1b $label (120k, 15 reps)"
  echo "===== $label topup =====" >> "$SP/server.log"
  start_server "$COMMON $*" || { log "WAVE1b $label FAILED TO START"; mykill; return 1; }
  python3 "$SP/blockA.py" "$label" "120000" 15 "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "WAVE1b $label done peak $(cat $SP/.peak) MiB"; mykill
}
log "WAVE 1b START (sample-size top-up at the decision depth)"
phase adaptive_n12_q8draft  "--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase mtp_n3_q8draft        "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase adaptive_n12_f16draft "--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k f16 --spec-draft-type-v f16"
log "WAVE 1b COMPLETE"
bash "$SP/wave_pp4.sh"
bash "$SP/wave_cross2.sh"
