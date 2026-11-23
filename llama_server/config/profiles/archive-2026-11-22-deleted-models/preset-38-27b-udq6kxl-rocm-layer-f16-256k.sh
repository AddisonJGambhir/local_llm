# ============================================================================
# Qwen3.8-27B UD-Q6_K_XL -- ROCm LAYER split. The PREFILL counterpart to the
# tensor-split preset (entry 10). Same binary, same model, same MTP n3.
#
# PREFILL (sampler-independent, ctx 262144, 10,776-token prompt):
#   tensor 50,50   411 pp
#   THIS (layer)  1013 pp     <- 2.5x
#
# DECODE, measured at Unsloth's thinking sampler (temp 1.0 / top_p 0.95 /
# top_k 20 / min_p 0 / presence 0), ctx 262144, aggregate throughput, 5 seeds:
#
#   prompt      tensor+MTP3   THIS layer+MTP3
#   code  10k      50.46           33.74
#   code  40k      50.85           32.07
#   prose 10k      33.15           21.83
#   prose 40k      28.02           18.18
#
# So the trade is 2.5x prefill for ~35% less decode -- steeper than the 18% an
# earlier temp-0.7 measurement suggested. Worth it only when prompt processing
# genuinely dominates: large tool results, repeated long-context reads with
# little generation. For anything generation-heavy, use entry 10.
#
# It does leave the XTX far cooler: ~20,900 MiB versus 24,325 for the tensor
# preset, which sits only ~235 MiB from the card's 24,560 physical.
#
# WHY NOT DFlash2 / DSpark HERE: measured and rejected. At ctx 262144 under the
# correct sampler, DFlash2 LOSES to plain MTP n3 in every cell -- 32.04 vs 33.74
# (code 10k), 28.71 vs 32.07 (code 40k), 16.62 vs 21.83 (prose 10k), 13.34 vs
# 18.18 (prose 40k) -- because its acceptance is roughly half MTP's. Its n-max
# knob is inert (n7 and n12 give identical accept counts). DSpark DOES fit -- the earlier
# "won't load" was a placement failure, not capacity (25 GiB of weights in 56 GiB
# of VRAM). It needs --spec-draft-device ROCm1 plus q8_0 KV at ctx 262144; small
# --tensor-split nudges do not help (layer-boundary rounding). Once running it is
# the WORST of the three and the hungriest: at q8_0 KV, code10k tg 27.17 vs
# DFlash2 30.52 vs MTP 32.14, and it takes 32,312 of the R9700's 32,624 MiB.
# Neither can run under --split-mode tensor at all; both abort in the Meta
# Backend (ggml-backend-meta.cpp:543 / :537).
# See records/2026-09-03-27b-rocm-speculation/SPECULATION-BY-CONTENT.md
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# ROCm1 = R9700 (ROCm0 is the 7900 XTX).
export MTMD_BACKEND_DEVICE=ROCm1

PROFILE_NAME="preset-38-27b-UDQ6KXL-rocm10-LAYER-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-rocm10/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-UD-Q6_K_XL.gguf"
LLAMA_BACKEND="rocm"

LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="mtp"
# MTP DRAFT DEPTH IS CONTEXT-DEPENDENT. Measured 2026-09-03, ROCm 10,
# -sm tensor 50,50, 10,776-token prompt, n_predict 256, temp 0.7/k20/p0.95,
# seed 42, 3 passes. Prefill flat (409-412 pp) everywhere -- decode-only.
#
#   n-max      @ctx 32768      @ctx 262144 (THIS PRESET)
#     2            -               51.87
#     3         52.44              57.31   <- PEAK, and already what was shipped
#     4         58.92              56.21
#     5         62.63              54.90   <- -4.2%: WORSE at full context
#     6         59.52                -
#     7         54.70                -
#     9         52.86                -
#    12         60.20                -
#
# n-max 3 was re-measured twice at 262144 (57.28, 57.31). THE ORIGINAL SETTING
# WAS ALREADY OPTIMAL -- a 2026-09-03 attempt to raise it to 5 was a regression
# and was reverted.
#
# n-max 5 wins by +19.4% at ctx 32768 and LOSES by 4.2% at ctx 262144. A shallow
# sweep does not transfer. `--spec-draft-n-max` is only a CEILING -- the adaptive
# PROFIT controller picks real depth, and its behaviour changes with the KV
# allocation. **Re-measure at 262144; never tune this at a smaller context.**
LLAMA_SPEC_DRAFT_N_MAX=3
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
LLAMA_CHAT_TEMPLATE_KWARGS='{"preserve_thinking":true,"reasoning_effort":"xhigh"}'

LLAMA_STARTUP_TIMEOUT=420
LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --split-mode layer
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
)
