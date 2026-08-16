#!/usr/bin/env bash
# Advised Vulkan preset: 35B MoE IQ4_NL, MTP, q8_0 KV, 256k context.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-35b-iq4-mtp-vulkan-q8-256k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-MTP-UD-IQ4_NL.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_CONTEXT=262144

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
