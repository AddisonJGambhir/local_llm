#!/usr/bin/env bash
# Wave 1c: RECORD ATTEMPT. PR #27173 chain build vs the current shipped build.
# depth -1 = repetitive regime (the regime the 81.67 t/s record was set in)
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
export BIN_OVERRIDE=/home/addison-gambhir/Desktop/local_llm/llama.cpp-tp-chain/build/bin/llama-server
source "$SP/lib.sh"
CSV="$SP/chain_ab.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
DEPTHS="-1,0,120000"
REPS=7

phase(){ local label="$1"; shift
  log "=== WAVE1c $label"
  echo "===== $label =====" >> "$SP/server.log"
  start_server "$COMMON $SPEC" || { log "WAVE1c $label FAILED TO START"; mykill; return 1; }
  log "$label up (peak $(cat $SP/.peak) MiB)"
  python3 "$SP/blockA.py" "$label" "$DEPTHS" "$REPS" "$CSV" 2>&1 | tee -a "$SP/run.log"
  log "$label done peak $(cat $SP/.peak) MiB"; mykill
}

log "WAVE 1c START (PR #27173 chain build record attempt)"
# arm A: new binary, all knobs OFF -> must match the shipped build (regression check)
phase chain_build_knobs_off
# arm B: the PR's own recommended config
export LLAMA_SPEC_CHAIN=1 LLAMA_META_MIRROR_OUTPUT=1 LLAMA_SCHED_POOL=8
phase chain_on_mirror_pool
unset LLAMA_SPEC_CHAIN LLAMA_META_MIRROR_OUTPUT LLAMA_SCHED_POOL
# arm C: isolate the lm-head mirror (the piece that targets our no-P2P topology)
export LLAMA_META_MIRROR_OUTPUT=1
phase mirror_only
unset LLAMA_META_MIRROR_OUTPUT
log "WAVE 1c COMPLETE -> $CSV"
