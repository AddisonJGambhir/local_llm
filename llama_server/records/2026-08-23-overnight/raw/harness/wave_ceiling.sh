#!/usr/bin/env bash
# Speculative ceiling probe on the degenerate repeated-sentence prompt.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
export REC_CSV="$SP/ceiling.csv"
export PROMPT_FILE=/home/addison-gambhir/Desktop/local_llm/llama_server/src/benchmarks/prompts/repetitive_sentence.txt
source "$SP/wave_rec_lib.sh"
log "CEILING PROBE START (degenerate prompt; upper bound only)"
arm ceil_shipped_adaptive "$A" $ADAPT
arm ceil_shipped_n3       "$A" $N3
log "CEILING PROBE COMPLETE -> $REC_CSV"
