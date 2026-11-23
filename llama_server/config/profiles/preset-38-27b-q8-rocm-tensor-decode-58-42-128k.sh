# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q8_0.gguf
# BUILD    llama.cpp   backend ROCm0,ROCm1
# SPLIT    tensor 58,42   (XTX,R9700)   main-gpu 1   ubatch 256
# CONTEXT  131072   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower on ROCm1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# ROCm TENSOR, Q8_0 -- FASTEST Q8 DECODE ON THIS BOX. MEASURED 2026-08-22
# ============================================================================
# 29.35 t/s raw decode, ~45-48 t/s with MTP, at 131,072 context.
# Sibling of preset-38-27b-q6k-rocm-tensor-decode-58-42-188k.sh (Q6_K, 192512).
#
# Q6_K is 13% faster on decode (33.24 vs 29.35) because decode is bandwidth
# bound and Q6_K is 21.3 GiB against Q8_0's 27.7 GiB. Use this only if you want
# Q8_0 weights. Note the inverse holds for PREFILL, where Q8_0 is ~1.2% FASTER
# because prefill is compute bound and Q6_K's dequant costs more per byte.
#
# ============================================================================
# WHY ts 58/42 -- MEASURED SEPARATELY ON THIS QUANT
# ============================================================================
# Not inherited from the Q6_K profile. Swept independently (llama-bench,
# -p 0 -n 512 -r 3, decode only):
#     ts 50/50   26.36      ts 58/42   28.82  <- peak
#     ts 54/46   27.82      ts 62/38   28.33
#                           ts 66/34   27.61
# Same optimum as Q6_K and Q6_K_P. Three models agree, so this is a property of
# the card pair (7900 XTX ~960 GB/s + R9700 ~640 GB/s), not of any quant.
#
# +11.4% over the ts 1,1 / 50/50 that the older tensor profiles used.
#
# ============================================================================
# CONTEXT: 131072, MEASURED ON AN IDLE BOX
# ============================================================================
# Live server, MTP n3, real 300-token generation, server-only VRAM
# (peak whole-card minus the desktop baseline sampled immediately before load):
#     c 114688   tg 45.46   peak 23,105   server-only 21,492
#     c 122880   tg 44.04   peak 23,523   server-only 21,899
#     c 131072   tg 47.67   peak 23,812   server-only 22,017   <- SHIPPED
# 388 MiB of margin against the 24,200 ceiling at a ~1,795 MiB desktop.
#
#   *** THIS NEEDS YOUR DESKTOP UNDER ~2.18 GiB. Drop to 122880 (or 114688)
#       if it is busier than that. ***
#
# MEASUREMENT WARNING, learned the hard way: an earlier attempt put this same
# context at 24,263 MiB and "over the line". That run was invalid -- a leaked
# llama-bench from a previous test was still holding 7.4 GiB, so every probe
# started 7 GiB in the hole. ALWAYS check the idle baseline is sane (<3 GiB)
# before trusting a VRAM measurement. The probe script now aborts if it is not.
#
# ============================================================================
# WHAT THIS PROFILE IS NOT FOR
# ============================================================================
# PREFILL. ROCm tensor encodes 71,449 tokens in 211.6s (337.7 t/s). Vulkan
# layer/row does the same prompt in 72-73s (~980 t/s), 2.9x faster. If you feed
# long prompts, use a prefill profile; tensor split is a decode tool here.
#
# MTP figures above are indicative only. Draft acceptance is content dependent
# (measured 46.8 t/s prose vs 59.4 code on identical settings at temp 0) and at
# temp 1.0 the same config ranged 26-68 t/s. Rank on the raw number.
# ============================================================================

source "$LLAMA_SERVER_ROOT/config/profiles/preset-38-27b-q6k-rocm-dual-f16-256k-mmproj.sh"

# +1.4% decode, free. Measured at -r 5 across three ratios and both quants:
#     ts 54/46  31.86 -> 32.75    ts 58/42  32.78 -> 33.24
#     ts 62/38  31.87 -> 32.25    Q8 58/42  28.49 -> 29.35
# Positive everywhere, prefill unaffected (428.46 -> 429.31), so there is no
# trade. This is the meta backend's CUDA-graph overhead, which PR #19378's
# author suspected but never quantified; it costs DECODE, not prefill.
export GGML_CUDA_DISABLE_GRAPHS=1

# MUST override the model: this sources the Q6_K base profile for its tuned
# cache/spec settings, and that parent sets LLAMA_MODEL to the Q6_K file.
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"

PROFILE_NAME="preset-38-27b-q8-mtp-rocm-TENSOR-DECODE-58-42-128k"
LLAMA_FIT=off
LLAMA_CONTEXT=131072
LLAMA_UBATCH=256
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_P_MIN=0.0

LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --device ROCm0,ROCm1
    --split-mode tensor
    # 58/42, not 1,1. See the sweep above -- this is the measured decode peak.
    --tensor-split 58,42
    --main-gpu 1
    --batch-size 6144
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --no-mmproj-offload
    --reasoning-preserve
)
