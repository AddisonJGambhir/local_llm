#!/usr/bin/env bash
# ROCm preset: 35B A3B MoE IQ4_NL, MTP, q8_0 KV, 256k context.
# ROCm optimum measured 2026-08-11: n-max 3 (145.0 t/s) ~= n-max 2 (144.5); 4+ falls away.
# Vulkan is ~41% faster on this model (204.9 t/s) -- use this only if you need ROCm.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-35b-iq4-mtp-rocm-q8-256k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-MTP-UD-IQ4_NL.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE=""
LLAMA_CONTEXT=262144

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
