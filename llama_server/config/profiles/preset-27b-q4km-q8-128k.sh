#!/usr/bin/env bash
# Vulkan preset: dense 27B Q4_K_M, MTP, q8_0 KV, 128k context.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-27b-q4km-mtp-vulkan-q8-128k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_CONTEXT=131072

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
# Dense 27B peaks at 4 (76.3 t/s vs 72.9 at 2); measured 78.0 vs 72.6 at 128k.
LLAMA_SPEC_DRAFT_N_MAX=4
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
