#!/usr/bin/env bash
# Advised Vulkan preset: 35B MoE Q4_K_M, MTP, q5_1, 190k context.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-35b-q4km-mtp-vulkan-q5_1-190k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp-beellama-032-hip/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_CONTEXT=194560

LLAMA_CACHE_TYPE_K="q5_1"
LLAMA_CACHE_TYPE_V="q5_1"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=2
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""

# Pinned: this profile inherits from default.sh, whose ubatch moved to 512 on
# 2026-08-11. Its q5_1 KV / high-context VRAM profile was NOT re-validated at 512,
# so it stays at the previously measured value.
LLAMA_UBATCH=192
