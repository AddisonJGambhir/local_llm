# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, 131072 ctx -- 2026-08-30
# ============================================================================
# pp 654.0 t/s, tg 42.1/42.5 @d0, tg 28.8 @d9060. All experts in VRAM; only
# the PLE n-gram table is on CPU (mmap, file-backed).
#
# THE HEADLINE: this preset is the shipped IQ3_XXS-82k preset's SPEED at
# 1.6x its CONTEXT. Same Q8_0 KV, same decode, pp within 5%.
#   IQ3_XXS 82k ub1024:  pp 688, tg 42.5, tg@d9060 29.2
#   Q2_K_XL 128k ub1024: pp 654, tg 42.1/42.5, tg@d9060 28.8
#   Q2_K_XL 256k ub256:  pp 440, tg 41.4/42.3, tg@d9060 29.4  (see variant)
#
# WHY Q2_K_XL FITS 128K WHEN IQ3_XXS CANNOT: the PLE table is a fixed 26.8 GiB
# in every unsloth quant, so only the body varies. Q2_K_XL's body is 46.7 GiB
# vs IQ3_XXS's 49.5. At ctx 131072 with ub 1024 the R9700 needs that missing
# ~1.8 GiB for KV + the ctx-scaled KQ mask; IQ3_XXS has only 66 MiB left at
# ub 256 and cliffs past it. Measured settled VRAM (Q2, 128k, ub 1024):
#   R9700 32,146 MiB (444 spare of 32,590)
#   XTX   22,880 MiB (1,420 spare under the 24,300 ceiling)
# Healthy margins on both cards; the fit has reproduced 3/3 loads.
#
# WHY NOT IQ3_XXS AT 128K: it works at ub 256 only (tg 40.5, tg@d9060 28.5,
# pp 425) with 66 MiB of R9700 margin. It is the fallback if Q2 body quality
# proves unacceptable, but on the standard probes (mmap/RSS essay, 9,060-token
# filler refusal) Q2 output was coherent and correct.
#
# 256K VARIANT (validated, zero offload, ub 256):
#   LLAMA_CONTEXT=262144 LLAMA_UBATCH=256
#   R9700 32,525 (65 spare), XTX 22,531. pp 440, tg 41.4/42.3, tg@d9060 29.4.
#   The KQ mask is ctx x ubatch x 4 bytes per attention-hosting card, so 256K
#   cannot afford ub 1024. Use when context matters more than prefill.
#   262,144 is the model's trained maximum (qwen4exp.context_length).
#
# MTP: DO NOT USE. Re-verified 2026-08-30 at 128k on the mtp-fix fork: tg
# 18.8 with n1 vs 28.3 off (77% acceptance). The draft reads its full 2.43 GiB
# every forward; the loss persists at any target speed and the draft displaces
# 2.43 GiB of context VRAM. Closed permanently on this hardware.
#
# EXPERT OFFLOAD (if a future variant needs room): each expert layer moved to
# CPU frees 962 MiB on its host card and costs ~7% tg (measured 40.3 -> 37.4).
# Cross-card expert moves cost 8.5% and whole-layer moves 32%; both are worse
# than or equal to CPU, so offload overflow to CPU only, never rebalance
# across cards.
#
# BUILD: llama.cpp-qwen4exp/build-vulkan @ aa1b06d0c = PR #27880 (contains the
# qwen4exp arch PR #27742). Mainline llama.cpp CANNOT load this model.
#
# --device IS MANDATORY (Vulkan2 is the CPU device). --load-mode mmap is
# MANDATORY (see the IQ3_XXS preset for the OOM history). --tensor-read-lazy
# was part of every validated run.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-LAYER-128k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=131072
LLAMA_UBATCH=1024

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
    --tensor-split 385,615
    --batch-size 2048
    --flash-attn on
    --load-mode mmap
    --tensor-read-lazy on
    -ot 'per_layer_token_embd\.weight=CPU'
)