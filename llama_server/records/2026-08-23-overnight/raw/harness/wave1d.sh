#!/usr/bin/env bash
# Wave 1d: PR #25666 (no MMVQ in speculative steps on AMD). Compare against
# wave1's same-flag rows on the shipped binary. n3 gets the fix on EVERY verify
# (4 rows); adaptive n12 only on rounds at depth <= 7 (8 rows).
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
export BIN_OVERRIDE=/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-mmvq/build/bin/llama-server
source "$SP/lib.sh"
CSV="$SP/mmvq_ab.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
DKV="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
DEPTHS="-1,0,120000"; REPS=7
phase(){ local label="$1"; shift
  log "=== WAVE1d $label"; echo "===== $label =====" >> "$SP/server.log"
  start_server "$COMMON $*" || { log "WAVE1d $label FAILED TO START"; mykill; return 1; }
  log "$label up (peak $(cat $SP/.peak) MiB)"
  python3 "$SP/blockA.py" "$label" "$DEPTHS" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "$label done peak $(cat $SP/.peak) MiB"; mykill
}
log "WAVE 1d START (PR #25666 mmvq)"
phase mmvq_n3        "--spec-type draft-mtp --spec-draft-n-max 3 $DKV"
phase mmvq_adaptive  "--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 $DKV"
log "WAVE 1d COMPLETE -> $CSV"
