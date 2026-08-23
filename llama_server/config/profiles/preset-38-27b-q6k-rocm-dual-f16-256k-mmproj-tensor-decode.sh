# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q6_K.gguf
# BUILD    llama.cpp   backend ROCm0,ROCm1
# SPLIT    tensor 1,1   (XTX,R9700)   main-gpu 1   ubatch 256
# CONTEXT  245760   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower on ROCm1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
#!/usr/bin/env bash
# Qwen3.8-27B Q6_K tensor-parallel decode profile, measured 2026-08-19.
# This is an INTERACTIVE/DECODE alternative to the balanced layer profile. It
# is not a universal replacement: tensor mode materially improves generation
# latency by making both GPUs work on each tensor, but roughly halves marginal
# long-context prefill throughput.
#
# Fixed long-context protocol, llama.cpp b10358 / 030ebb558, f16 KV:
#   config                    pp10240 @ d65536   tg2048 @ d65536
#   layer 38/27, ub256, mg1   565.85 / 566.55    20.04 / 19.99
#   tensor 1/1, ub256, mg1    294.53             27.33
# Tensor therefore gains 36.4% raw decode and loses 48.0% prefill against the
# paired layer mean. ub128/256/288/512 gave tg 27.27/27.33/27.34/27.21;
# ub256 remains the balanced choice. Tensor 34/31 reached tg 27.92 but its live
# server hit 24,226 MiB on the XTX, above the hard 24,200 MiB ceiling: rejected.
#
# Live MTP, fixed four natural prompts x 400 tokens x two passes:
#   layer 38/27 n3: 32.231 t/s, 51.62% acceptance, 8 samples
#   tensor 1/1 n3: 43.576 t/s, 51.01% acceptance, 8 samples (+35.2%)
# Paired 95% CI for the tensor gain: +9.35 to +13.34 t/s. An additional n3
# repeat makes the 16-sample tensor mean 43.265 t/s. n2 averaged 44.173 across
# 16 samples, only +2.1%; its paired CI crossed zero, so n3 remains shipped.
# n4 averaged 42.231. High-p-min tests raised acceptance but cut throughput.
#
# VRAM: live tensor 1/1 peaked 22,211-22,306 MiB at c=245760. main-gpu 0 was
# measured at 294.35 pp / 27.29 tg (no speed gain) and has much worse XTX
# headroom, so mg1 stays. Tensor initialization was verified from the server's
# Meta-device/AllReduce log lines, not inferred from /health.
#
# NOT MEASURED: vision-request throughput or transient VRAM; tensor mode at a
# context above 245760; concurrent-request throughput (--parallel remains 1).

source "$LLAMA_SERVER_ROOT/config/profiles/preset-38-27b-q6k-rocm-dual-f16-256k-mmproj.sh"

PROFILE_NAME="preset-38-27b-q6k-mtp-rocm-dual-f16-256k-mmproj-tensor-decode"
LLAMA_FIT=off
LLAMA_CONTEXT=245760
LLAMA_UBATCH=256
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_P_MIN=0.0

LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --device ROCm0,ROCm1
    --split-mode tensor
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 6144
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --reasoning-preserve
)
