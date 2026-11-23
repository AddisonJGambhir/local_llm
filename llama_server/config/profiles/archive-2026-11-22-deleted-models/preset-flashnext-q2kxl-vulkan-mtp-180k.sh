#!/usr/bin/env bash
# Qwen3.8-Flash-Next 125B-A6B UD-Q2_K_XL -- Vulkan layer split + MTP.
# REJECTED FOR PRODUCTION: retained only as an experimental/reference profile.
# Real temperature=1/top-p=0.95 traffic measured 21-28 tg at 12-16K depth,
# versus roughly 39-40 tg without MTP.  Greedy microbenchmarks overstated its
# value by producing artificially favorable speculative acceptance.
#
# This profile requires llama.cpp-flashnext-full: the Vulkan fusion stack plus
# the quantized-MTP skinny-inject guard.  Without that guard, the Q4_K_M MTP
# sidecar reaches the mat-vector path with reversed operands and asserts.
#
# Measured at 11,568 prompt tokens with 128 generated:
#   ctx 184320, ub192, split 400,600: 333.24 pp / 36.04 tg
# The smaller ctx 163840 tuning point reached about 477 pp / 40.4 tg; allocating
# the requested 180K window costs throughput even at the same shallow depth.
# Context is shipped at 180 Ki tokens (184,320).  This deliberately sits below
# the 200,704 allocation that put both boards at their limits once MTP was
# loaded. Vulkan may briefly report almost no headroom during initialization;
# settled residency is the useful signal on this backend.  The validation run
# settled at 32,568 MiB R9700 / 23,196 MiB XTX and served without migration.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-full-fusions-mtp-180k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-flashnext-full/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=184320
LLAMA_UBATCH=192
LLAMA_CACHE_RAM=8192

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_P_MIN=0.7
LLAMA_SPEC_DRAFT_TYPE_K="f16"
LLAMA_SPEC_DRAFT_TYPE_V="f16"
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
    -ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'
    --spec-draft-model "$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/MTP-Q4_K_M/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf"
    -otd 'token_embd\.weight=CPU,output\.weight=CPU'
)
