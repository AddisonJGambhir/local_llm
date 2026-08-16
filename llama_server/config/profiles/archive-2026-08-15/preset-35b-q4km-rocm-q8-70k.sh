#!/usr/bin/env bash
# ROCm preset: 35B A3B MoE Q4_K_M (heavy/best quant), MTP, q8_0 KV, 70k context.
#
# ROCm carries ~0.47 GiB more VRAM than Vulkan for the same config (22.50 vs
# 22.03 GiB at 80k), so it has correspondingly less margin. The 80k ceiling was
# measured on Vulkan; this profile is held 10k below it to keep real headroom on
# ROCm. The Vulkan sibling (preset-35b-q4km-vulkan-q8-80k.sh) stays at 80k.
# ROCm optimum n-max 2 (140.5) ~= 3 (139.1); 3 chosen to match the other MoE presets.
# Vulkan is ~27% faster on this model (189.9 t/s) -- use this only if you need ROCm.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

PROFILE_NAME="preset-35b-q4km-mtp-rocm-q8-70k"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE=""
LLAMA_CONTEXT=71680

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""
