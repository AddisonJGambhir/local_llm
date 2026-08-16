#!/usr/bin/env bash
# Qwen3.6 27B Q4_K_M — VULKAN, 7900 XTX ALONE, q8_0 KV, 128k. Decode-optimised.
#
# *** SINGLE-CARD ON PURPOSE. Do not "fix" this by adding the R9700. ***
# This model is 15.93 GiB and fits on one card. Splitting it across both costs
# ~34% of decode, because every token then makes a cross-card round trip.
# Measured 2026-08-16, 128k / q8_0 KV, same harness:
#
#   backend  placement      pp      tg
#   Vulkan   XTX only     779.9   70.44   <- this profile
#   Vulkan   R9700 only   863.7   50.26
#   Vulkan   both         776.2   46.76
#   ROCm     XTX only     876.7   45.34
#   ROCm     R9700 only   898.3   37.81
#   ROCm     both        1393.2   34.83   <- the dual profile, prefill-optimised
#
# Two things drive this:
#   1. Crossing cards is what costs, not how much crosses. With 56 of 65 layers
#      on the XTX and f16 KV, decode is still only 47.2 t/s. Any layer on the
#      second card forces the per-token round trip.
#   2. Vulkan is 55% faster than ROCm on the SAME card for this model
#      (70.44 vs 45.34). The largest backend gap measured on this box.
#
# q8_0 KV is the exception to the f16-everywhere rule, and it is forced: f16 at
# 128k needs ~25.6 GiB and the XTX has ~21.5 usable after the desktop's ~3 GiB.
# f16 DOES fit at 64k (800 pp / 71.35 tg) but that is ~1% faster for half the
# context — not worth it. 96k with f16 does not fit.
#
# Choose this profile for generation-heavy work; choose the ROCm dual profile
# when prefill dominates (1393 vs 780 pp, i.e. +79%).

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-36-27b-q4km-vulkan-xtx-only-q8-128k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_CONTEXT=131072
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576

# Forced, not chosen — see header. f16 does not fit on one card at 128k.
LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=2

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="auto"

LLAMA_STARTUP_TIMEOUT=300
LLAMA_TIMEOUT=3600

LLAMA_EXTRA_ARGS=(
    --batch-size 2048
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
