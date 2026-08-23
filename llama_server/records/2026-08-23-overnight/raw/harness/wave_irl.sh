#!/usr/bin/env bash
# Why does the bench say ~42 t/s at 120k when real use shows ~27?
# Isolate ONE variable: the sampler chain. Everything else matches production
# exactly (shipped binary, adaptive n12, c=262144, tensor 49/51, q8_0 draft KV).
# Note: under -sm tensor the backend sampler is disabled and sampling falls to
# CPU, so a full temp/top-p/top-k chain costs more per token than argmax.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"; source "$SP/lib.sh"
CSV="$SP/irl_sampler.csv"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SP/run.log"; }
trap 'mykill' EXIT
COMMON="-c 262144 --cache-type-k f16 --cache-type-v f16 --ubatch-size 1024 --cache-ram 24576"
SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
log "WAVE IRL START (sampler chain isolation at 120k)"
start_server "$COMMON $SPEC" || { log "IRL FAILED TO START"; exit 1; }
log "up (peak $(cat $SP/.peak) MiB)"
python3 "$SP/blockA.py" bench_sampler_temp0 "120000" 7 "$CSV" 2>&1 | tee -a "$SP/run.log"
IRL_SAMPLER=1 python3 "$SP/blockA.py" irl_sampler_temp1 "120000" 7 "$CSV" 2>&1 | tee -a "$SP/run.log"
log "WAVE IRL COMPLETE -> $CSV"; mykill
