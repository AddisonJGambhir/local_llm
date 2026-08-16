#!/usr/bin/env bash
# ROCm preset: dense 27B Q4_K_M, MTP, q8_0 KV, 128k context.
# ROCm optimum measured 2026-08-11: n-max 2 (48.5 t/s). Note this DIFFERS from the
# Vulkan optimum of 4 -- ROCm's acceptance falls off faster with depth here.
# Vulkan is ~57% faster on this model (76.3 t/s) -- use this only if you need ROCm.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-27b-q4km-mtp-rocm-q8-128k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE=""
LLAMA_CONTEXT=131072

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=2
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
