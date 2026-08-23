#!/usr/bin/env bash
# LLAMA_SPEC_CHAIN crashes on this fork (GGML_ASSERT(buffer), chain needs out_ids
# API absent from our base). MIRROR and SCHED_POOL do not touch the chain graph.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
source "$SP/wave_rec_lib.sh"
log "REC2 START (chain-free #27173 knobs)"
LLAMA_META_MIRROR_OUTPUT=1 LLAMA_SCHED_POOL=8 arm chain_mirror_pool "$C" $ADAPT
LLAMA_SCHED_POOL=8                            arm chain_pool_only   "$C" $ADAPT
log "REC2 COMPLETE"
