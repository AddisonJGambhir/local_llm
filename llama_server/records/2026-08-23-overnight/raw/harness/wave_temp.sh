#!/usr/bin/env bash
# Draft depth at the PRODUCTION sampler (temp 1.0 / top-p 0.95 / top-k 20).
# The whole n-sweep ran at temp 0. Temperature already measured as dropping
# acceptance 0.602 -> 0.549, and deeper drafts compound that: each extra draft
# token is another chance for a stochastic pick to diverge. n5 won at temp 0;
# it may well lose here. Range 2-4 is where real use lives.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"; source "$SP/lib.sh"
CSV="$SP/temp_nsweep.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
DKV="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
phase(){ local label="$1"; shift
  log "=== TEMPSWEEP $label"; echo "===== $label =====" >> "$SP/server.log"
  start_server "$COMMON $*" || { log "$label FAILED TO START"; mykill; return 1; }
  IRL_SAMPLER=1 python3 "$SP/blockA.py" "$label" "120000" 10 "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "$label done peak $(cat $SP/.peak) MiB"; mykill; }
log "TEMP-1.0 N SWEEP START (production sampler, 120k, 10 reps)"
phase t1_n2 "--spec-type draft-mtp --spec-draft-n-max 2 $DKV"
phase t1_n3 "--spec-type draft-mtp --spec-draft-n-max 3 $DKV"
phase t1_n4 "--spec-type draft-mtp --spec-draft-n-max 4 $DKV"
phase t1_n5 "--spec-type draft-mtp --spec-draft-n-max 5 $DKV"
log "TEMP-1.0 N SWEEP COMPLETE -> $CSV"
