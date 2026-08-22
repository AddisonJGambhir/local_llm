# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q6_K.gguf
# BUILD    llama.cpp   backend ROCm0,ROCm1
# SPLIT    tensor 58,42   (XTX,R9700)   main-gpu 1   ubatch 256
# CONTEXT  192512   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower NOT routed - defaults to first device, the XTX)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# ROCm TENSOR, Q6_K -- FASTEST DECODE ON THIS BOX. MEASURED 2026-08-21
# ============================================================================
# 53-54 t/s with MTP at 163,840 context. The split ratio is the whole point:
# ts 58/42 was never tested before today and it beats every decode number in a
# 78-run matrix.
#
# ============================================================================
# WHY 58/42 -- THE MATRIX STEPPED OVER THE PEAK
# ============================================================================
# The 72-run matrix only tested tensor at 50/50 and 70/30 on ROCm. 70/30 came
# out WORSE than 50/50 (29.65 vs 30.27), which made 50/50 look optimal. It was
# not: the peak sits between them and both probes missed it.
#
# Filled in 2026-08-21 (llama-bench, -r 3, ub 1024, f16 KV, -p 2048 -n 256):
#     ts 50/50   pp 434.71   tg 30.19
#     ts 55/45   pp 425.40   tg 32.25
#     ts 58/42   pp 414.85   tg 32.84   <- SHIPPED. real interior maximum
#     ts 60/40   pp 412.18   tg 32.56
#     ts 62/38   pp 411.77   tg 32.01
#     ts 65/35   pp 402.38   tg 31.39
# +8.8% decode over 50/50 for 4.6% of prefill. Points on both sides fall away,
# so this is a genuine peak, not the edge of the tested range.
#
# DO NOT COPY THIS RATIO TO THE VULKAN PROFILES. Vulkan tensor peaks at 70/30
# (29.15 tg) and measures only 25.63 at 60/40. The two backends want different
# splits; the ratio is not portable.
#
# ============================================================================
# CONTEXT: 192512 SHIPPED -- SIZED TO FILL THE CARD TO 24.1 GiB
# ============================================================================
# Ceiling for this box is 24,200 MiB (owner's number). This profile is
# deliberately sized to sit just under it rather than banking headroom.
#
# Live server, MTP n3, real 400-token generation. server-only = peak whole-card
# minus the desktop baseline sampled immediately before load:
#     c 131072   tg 54.46   peak 21,420   server-only 19,392   acc 65.1%
#     c 163840   tg 53.16   peak 22,890   server-only 20,864   acc 61.7%
#     c 180224   tg 49.72   peak 23,518   server-only 21,593   acc 55.3%
#     c 188416   tg 49.31   peak 23,931   server-only 21,980   acc 55.4%
#     c 192512   tg 49.27   peak 24,109   server-only 22,158   acc 55.1%  <- SHIPPED
#     c 196608   tg 54.26   peak 24,305   server-only 22,317   acc 64.1%  <- over 24,200
#
# 44.6 MiB of server-only VRAM per 1k tokens.
#
# READ THE DECODE COLUMN WITH THE ACCEPTANCE COLUMN. tg tracks draft acceptance,
# which is stochastic at temp 1.0 and ranged 55.1-65.1% across these six runs.
# c=196608 posting the HIGHEST tg (54.26) at the LARGEST context is the tell:
# that is a lucky acceptance draw, not a context effect. Real decode here is
# ~49-54 t/s at any of these sizes and context is effectively free. Do NOT read
# the 53.16 -> 49.27 drop as a cost of context; it is sampling noise.
#
# ONLY 91 MiB OF MARGIN. This is a deliberate choice to use the card, not a
# safe default:
#   - The desktop baseline was 1,925-2,026 MiB during these probes. It has been
#     as high as 3,000 MiB on this box within a single day. If yours grows past
#     ~2.05 GiB this profile will cross 24,200.
#   - Above the ceiling nothing crashes. Throughput silently collapses via GTT
#     spillover (measured elsewhere: 79% of prefill lost, no failed health
#     check). If this profile ever feels slow, check peak VRAM FIRST.
#
#   *** FALL BACK TO 163840 (peak 22,890, 1,310 MiB of margin) IF THE DESKTOP
#       IS BUSY, OR IF THROUGHPUT EVER LOOKS WRONG. ***
#
# ============================================================================
# WHAT THIS PROFILE IS NOT FOR
# ============================================================================
# PREFILL. ROCm tensor prefill is ~415 t/s here; ROCm LAYER split does 1195-1229
# on the same hardware, ~3x faster. That gap is a synchronization cost in the
# meta-backend's allreduce, not an interconnect limit -- doubling the wire bytes
# (GGML_CUDA_AR_BF16_THRESHOLD=0) changes prefill by 0.1%, and so does a 32x
# change in chunk size, a 16x change in ubatch, and swapping the allreduce
# algorithm entirely. See docs/pr25051-run-2026-08-21/.
#
# Use LAYER split for prompt-heavy work. Use THIS for interactive/agent decode.
#
# QUANT: Q6_K ONLY. Every number here is Qwen3.8-27B-Q6_K.gguf (21,811 MiB).
# Q8_0 puts ~3,417 MiB MORE on the XTX at this ratio and would top out at a far
# lower context. Neither the ratio nor the context ceiling has been verified on
# Q8_0 -- re-measure before reusing either.
# ============================================================================

source "$LLAMA_SERVER_ROOT/config/profiles/preset-38-27b-q6k-rocm-dual-f16-256k-mmproj.sh"

# +1.4% decode, free. Measured at -r 5 across three ratios and both quants:
#     ts 54/46  31.86 -> 32.75    ts 58/42  32.78 -> 33.24
#     ts 62/38  31.87 -> 32.25    Q8 58/42  28.49 -> 29.35
# Positive everywhere, prefill unaffected (428.46 -> 429.31), so there is no
# trade. This is the meta backend's CUDA-graph overhead, which PR #19378's
# author suspected but never quantified; it costs DECODE, not prefill.
export GGML_CUDA_DISABLE_GRAPHS=1

PROFILE_NAME="preset-38-27b-q6k-mtp-rocm-TENSOR-DECODE-58-42-188k"
LLAMA_FIT=off
LLAMA_CONTEXT=192512
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
    --reasoning-preserve
)
