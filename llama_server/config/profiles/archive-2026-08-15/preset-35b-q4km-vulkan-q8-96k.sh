#!/usr/bin/env bash
# Vulkan preset: 35B A3B MoE Q4_K_M (the heavy/best quant), MTP, q8_0 KV, 96k context.
#
# Verified by direct load 2026-08-11 at 96k (ubatch 512 + MTP):
#   procVRAM 22.25 GiB   procGTT 0.21   card 23.37/23.98 (0.61 free)
#   throughput 281.5 pp / 189.8 tg -- identical to the 32k figure, no degradation.
#
# NOTE: ctx_headroom.py's binary search called 88k a spill (procGTT 0.32, just over
# its 0.30 threshold) and reported an 80k ceiling. A direct load at 96k measures
# 0.21 GTT, so that borderline reading was noise-sensitive -- the scan is
# conservative here. Trust the direct load; re-verify if the desktop's GPU usage
# changes materially.
#
# If this profile ever slows down mid-session, suspect a GTT spill first and drop
# to 80k or 64k. The ROCm sibling runs 70k: ROCm carries ~0.47 GiB more VRAM for
# the same config, so it has less room.
#
# The README's older "136k @ q8_0" figure for this model predates MTP and
# ubatch 512; both consume headroom.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-35b-q4km-mtp-vulkan-q8-96k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_CONTEXT=98304

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
# Vulkan optimum: 3 (189.9 t/s vs 188.2 at 2, 176.5 at 4, 147.4 at 6).
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
