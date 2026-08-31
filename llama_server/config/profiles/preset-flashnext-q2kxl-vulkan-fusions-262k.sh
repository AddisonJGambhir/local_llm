# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, populated 200K
# Corrected 2026-08-30 after actually reading the context.
# ============================================================================
# The previous 262,144 / ub512 profile was validated with only 8,948 live
# prompt tokens. Its 855 pp / 49.5 tg figures were shallow measurements and
# did not reveal Vulkan residency migration. They are withdrawn.
#
# AUTHORITATIVE FULL-DEPTH RESULT -- q8_0 K/V, EVERY EXPERT IN VRAM:
#   allocated context     200,704
#   actual prompt depth   199,320 tokens, followed by 256 generated tokens
#   pp                    347.8360 t/s
#   tg at 199K depth       13.5271 t/s
#   R9700                  30,304-30,310 MiB during the entire request
#   XTX                    23,524-23,620 MiB during the entire request
#   R9700 GTT              19-44 MiB
#   XTX GTT                1,748-1,773 MiB (normal desktop footprint)
#   migration              none
# Run: /tmp/fn-sweep/deep200k-ctx200704-ts400-ub128-none
# Record: records/2026-08-30-flashnext-deep-residency/RESULTS.md
#
# WHY UBATCH 128: ub256 migrated roughly 1 GiB from R9700 VRAM into GTT late
# in the prompt in two separate full-depth attempts, including with the same
# 200,704 allocation. ub192 completed, but bought only +2.4% pp while losing
# 2.4% tg and safety margin. ub128 is the best all-round populated-200K mode.
#
# WHY SPLIT 400,600: at ub128 it improved both full-depth pp and tg over
# split 375,625 (338.99 pp / 13.41 tg), while staying resident. Ratios 390-405
# map to the same whole-layer placement. 410 moves one more layer to the XTX
# and does not fit safely.
#
# NO CPU EXPERT OFFLOAD. Only the two embedding tables live on CPU. CPU expert
# layers hurt both pp and tg. Do not raise allocated context or ubatch based on
# startup VRAM: migration appears only after actually populating deep context.
#
# BUILD: requires the patched experimental worktree
# llama.cpp-qwen4exp-vulkan-fusions (detached at aa1b06d0c, dirty; Vulkan
# mat-vec epilog fusion, fused REPEAT+MUL+ADD, skinny inject prefill path,
# radix TOP_K). Do NOT reset, clean or checkout it. Rebuild with:
#   CCACHE_DIR=/tmp/flashnext-ccache cmake --build \
#     llama.cpp-qwen4exp-vulkan-fusions/build-vulkan --target llama-server -j 16
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device). --load-mode mmap is
# MANDATORY. --tensor-read-lazy was part of every validated run.
#
# MTP / SPECULATION: closed. The draft costs 2.43 GiB resident and does not fit
# on the XTX at this split. All five zero-VRAM n-gram modes were also measured and all
# lose: on the deep agentic prompt, none 39.23 vs ngram-simple 27.26 and
# ngram-mod 31.53.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-fusions-populated-200k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=200704
LLAMA_UBATCH=128
# Match the validated runs and the binary default. The shared profile's 24 GiB
# host prompt cache creates avoidable RAM/swap pressure; 8 GiB still permits
# idle prompt checkpoints without changing the live slot's KV capacity.
LLAMA_CACHE_RAM=8192

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
