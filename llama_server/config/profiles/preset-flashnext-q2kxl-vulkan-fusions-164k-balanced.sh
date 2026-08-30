# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, 163840 ctx -- 2026-08-30
# BALANCED preset: near-50 decode AND near-800 prefill, 1.25x over 128K.
# ============================================================================
# Measured (5-pass median shallow; 8,948-token deep probe):
#   shallow tg 49.8583  (min 49.7849, max 49.9635, spread 0.1786)
#   deep pp    767.47   deep tg 38.8108
#   R9700 30,902 MiB (1,298 spare under 32,200)
#   XTX   23,692 MiB (508 spare under 24,200)
#   No CPU expert offload. Verified clean, no guard strikes.
#
# PICK THIS OVER THE 262K PRESET when prefill matters. Same split, same -ot,
# same binary; only ctx and ubatch differ.
#
# SIBLING POINTS ON THE SAME CURVE (all verified clean):
#   ctx      ub    tg      pp     deep tg   R9700 spare   XTX spare
#   262,144  128   50.03   586    41.02     1,131         428
#   229,376  128   50.13   585    41.00     1,539         710
#   163,840  512   49.86   767    38.81     1,280         542
#   131,072  512   48.22   816    39.42     1,740         738
#
# ub 512 is the practical prefill ceiling on this box. ub 1024 and ub 2048
# breach at every context tried -- 131,072 with ub 2048 reached R9700 32,601
# against the 32,200 guard. Do not raise it.
#
# ALLOCATION VARIES RUN TO RUN by up to ~1 GiB, and the XTX also drives the
# desktop (1,332-2,100 MiB depending on what is open). Treat any config with
# under ~500 MiB of spare on either card as a coin flip, not a config.
#
# BUILD: requires the patched experimental worktree
# llama.cpp-qwen4exp-vulkan-fusions (detached at aa1b06d0c, dirty). Do not
# reset, clean or checkout that worktree. Rebuild with:
#   CCACHE_DIR=/tmp/flashnext-ccache cmake --build \
#     llama.cpp-qwen4exp-vulkan-fusions/build-vulkan --target llama-server -j 16
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device). --load-mode mmap is
# MANDATORY. --tensor-read-lazy was part of every validated run.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-fusions-164k-balanced"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=163840
LLAMA_UBATCH=512

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="none"
LLAMA_DRAFTER=""

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="on"

LLAMA_STARTUP_TIMEOUT=420
LLAMA_EXTRA_ARGS=(
    --split-mode layer
    --tensor-split 400,600
    --batch-size 2048
    --flash-attn on
    --load-mode mmap
    --tensor-read-lazy on
    -ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'
)
