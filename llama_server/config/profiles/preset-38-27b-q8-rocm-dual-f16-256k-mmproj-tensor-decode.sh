# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q8_0.gguf
# BUILD    llama.cpp   backend ROCm0,ROCm1
# SPLIT    tensor 45,55   (XTX,R9700)   main-gpu 1   ubatch 512
# CONTEXT  245760   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower on ROCm1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# ts 50,50 WAS TRIED AND REVERTED, 2026-08-19 -- VRAM, not speed
# ============================================================================
# It was FASTER on both axes (llama-bench, -d 65536 -p 10240 -n 256, ub 512):
#     ts 45/55  pp 274.91  tg 22.68   <- SHIPPED (this file)
#     ts 50/50  pp 300.16  tg 24.29   +9.2% pp, +7.1% tg; live MTP 44.8 -> 48.0
#     ts 55/45  pp 282.31  tg 25.42   best decode, never fit at all
#
# IT DOES NOT FIT UNDER REAL LOAD. The load-time plateau probe said 22,670 MiB
# at c=245000 and that number is a TRAP: serving adds ~1,800 MiB on top. In
# production the XTX sat at 24,487 MiB of 24,560 -- 73 MiB free, past the 24,300
# ceiling, with the kernel's svm_range_evict_svm_bo_worker already firing. The
# server had not spilled to GTT yet, but there was no margin left for the
# desktop, which moves ~500 MiB on its own within an hour.
#
# DO NOT re-ship 50/50 on the strength of the speed numbers without re-measuring
# VRAM WHILE THE SERVER IS SERVING, not at the load-time plateau. If context
# ever drops to ~229376 or lower, 50/50 becomes worth re-testing -- the speed
# win is real and mechanism-backed (the R9700 is behind a Gen4 x4 chipset link,
# so tensor-mode all-reduce traffic sent its way crosses a bottleneck; balance
# beats weighting toward it).
# ============================================================================

#!/usr/bin/env bash
# Qwen3.8-27B Q8_0 tensor-parallel decode profile, measured 2026-08-19.
# This is an INTERACTIVE/DECODE alternative to the balanced layer profile. It
# trades about half of marginal long-context prefill for materially lower
# generation latency. Keep the layer profile for ingestion-heavy requests.
#
# Fixed long-context protocol, llama.cpp b10358 / 030ebb558, f16 KV:
#   config                     pp10240 @ d65536   tg2048 @ d65536
#   layer 1/1, ub256, mg1      509.40 / 508.56    18.44 / 17.63
#   tensor 45/55, ub256, mg1   263.93 / 263.22    22.48 / 22.64
#   tensor 45/55, ub512, mg1   275.20             22.63
# Final tensor gains 25.5% raw decode against the layer mean and loses 45.9%
# prefill. The tensor split sweep was 35/65=273.89/20.47,
# 40/60=265.99/21.56, and 45/55=263.93/22.48 (pp/tg), so 45/55 is the
# decode winner. main-gpu 0 measured 263.21/22.40: no improvement, worse
# placement for the desktop GPU, rejected.
#
# UBATCH SWEEP at tensor 45/55 (same long protocol):
#   ub128  235.14 pp / 22.57 tg
#   ub256  263.93, 263.22 / 22.48, 22.64
#   ub288  225.98 pp / 22.37 tg   (the warned discontinuity is real)
#   ub512  275.20 pp / 22.63 tg   <- SHIPPED
# Decode is tied across the useful points; ub512 adds 4.4% prefill over the
# repeated ub256 mean. It was therefore validated again on the live server.
#
# Live MTP at safe c=245760, fixed four natural prompts x 400 tokens x 2 passes:
#   layer 1/1 n3:     34.352 t/s, 53.46% acceptance, peak XTX 23,225 MiB
#   tensor 45/55 n3, ub512: 44.813 t/s, 53.98% acceptance, peak 21,908 MiB
# Tensor gains 30.5%; paired 95% CI is +8.77 to +12.15 t/s. Relative to its
# unspeculated 22.63 t/s, MTP adds 98.0%. Draft n2 at ub256 raised acceptance to
# 60.11% but fell to 41.517 t/s. n4 fell to 44.087 t/s and its paired CI versus
# n3 crossed zero, so n3 remains shipped.
#
# The original layer server at c=260096 reached 24,197 MiB on request two and
# was stopped by the exact-PID watcher before crossing the hard 24,200 ceiling.
# This profile intentionally uses c=245760, the context used for the valid live
# A/B, instead of retaining the former layer profile's unsafe c=262144 setting.
#
# NOT MEASURED: vision-request throughput or transient VRAM; live MTP at
# c=262144; draft-depth interaction with ub512; concurrent-request throughput.

source "$LLAMA_SERVER_ROOT/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh"

PROFILE_NAME="preset-38-27b-q8-mtp-rocm-dual-f16-256k-mmproj-tensor-decode"
LLAMA_FIT=off
LLAMA_CONTEXT=245760
LLAMA_UBATCH=512
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_P_MIN=0.0

LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --device ROCm0,ROCm1
    --split-mode tensor
    --tensor-split 45,55
    --main-gpu 1
    --batch-size 6144
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --no-mmproj-offload
    --reasoning-preserve
)
