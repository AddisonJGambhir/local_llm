# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, 262144 ctx -- 2026-08-30
# MAX CONTEXT + 800+ PREFILL. 262,144 is the model's trained maximum
# (qwen4exp.context_length); nothing above it without RoPE extension.
# ============================================================================
# Measured, 9-pass median shallow tg, 8,948-token deep probe, q8_0 K and V,
# EVERY EXPERT IN VRAM:
#   tg      49.5362  (min 48.9522, max 50.2466, spread 1.2944)
#   pp     855.2231
#   deep tg 41.6166
#   VRAM while actually generating, sampled every 6s for a minute, no drift:
#     R9700 32,419 MiB of 32,624 physical
#     XTX   23,521 MiB of 24,560 physical
#
# DECODE-FIRST VARIANT -- same everything, LLAMA_UBATCH=512:
#   tg 50.1408 (9-pass median), pp 585.63, deep tg 41.0227
#   R9700 31,069, XTX 23,520
#   Trades 269 pp for 0.6 t/s. Use only if decode is all you care about.
#
# ubatch is the ONLY pp lever: pp is 581.5 / 581.3 / 585.6 at ctx 131,072 /
# 196,608 / 262,144 with ub 128. Allocated context does not affect pp at all.
#
# NO CPU EXPERT OFFLOAD -- it is a dead end here, measured twice over:
#   ub 512 with blk.11+28+35 experts on CPU -> pp 579.59, tg 35.98.
#   That is WORSE pp than ub 128 all-GPU (585.63) and 13 t/s worse decode.
#   CPU offload costs prefill as much as decode. Never trade experts for
#   ubatch. If you must shed VRAM, shed context instead.
#
# NEVER MOVE EXPERTS ACROSS CARDS: -ot ...=Vulkan0/1 on one layer's expert
# matrices measured 34.75 / 38.80 / 38.70 t/s against 49.48 unsplit.
#
# TENSOR-SPLIT IS QUANTISED TO WHOLE LAYERS. 390,610 / 395,605 / 400,600 /
# 405,595 all produce byte-identical card totals (32,419 / 23,513) because all
# four assign 19 layers to the XTX. Their apparent tg differences (48.31 to
# 49.89 on 3-pass medians) are run-to-run noise, not tuning. 410,590 assigns
# 20 and breaches the XTX at 24,484. Do not "tune" between 390 and 405.
#
# RUN-TO-RUN VARIANCE IS ~1.5 t/s on a 3-pass median. Use 9 passes before
# believing any difference smaller than that.
#
# ub 256 BREACHES where ub 512 DOES NOT (XTX 24,277 vs 23,521). Vulkan buffer
# sizing is not monotonic in ubatch on this model. Measure, never extrapolate.
#
# GUARDS: R9700 32,500 (headless card, no desktop, 124 MiB physical margin),
# XTX 24,200 (drives the desktop, which itself holds 1,228-2,100 MiB). The
# guard now enforces configured ceilings only AFTER the server reports ready;
# during load it enforces physical limits, because allocation spikes high and
# settles. Peak-during-load is not a valid measurement -- steady state while
# prefilling or generating is.
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
# MTP / SPECULATION: closed. The draft costs 2.43 GiB resident and there is
# ~200 MiB spare. All five zero-VRAM n-gram modes were also measured and all
# lose: on the deep agentic prompt, none 39.23 vs ngram-simple 27.26 and
# ngram-mod 31.53.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-fusions-262k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
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
