# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q8_0.gguf
# BUILD    llama.cpp-vulkan-tp   backend Vulkan0,Vulkan1
# SPLIT    tensor 49,51   (XTX,R9700)   main-gpu 1   ubatch 512
# CONTEXT  262144   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   no
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# VULKAN TENSOR, Q8_0, FULL 262144 CONTEXT AT f16 KV. MEASURED 2026-08-22
# ============================================================================
# The maximum context this model supports (262144 native), on Q8_0 weights, with
# a full-precision f16 KV cache. Nothing is quantized away to make it fit.
#
# TEXT ONLY -- no vision projector. Every VRAM number below was measured without
# --mmproj. To add vision: set MTMD_BACKEND_DEVICE=Vulkan1 to park the 889 MiB
# tower on the R9700, or pass --no-mmproj-offload to keep it in system RAM.
# Either leaves the XTX budget untouched, which is the constraint here.
#
# NON-STANDARD BUILD -- llama.cpp-vulkan-tp (worktree, branch pr25051). Stock
# Vulkan cannot tensor-split usefully; see the Q6_K siblings for the rebuild.
#
# ============================================================================
# WHY ts 49,51 -- THE CEILING IS A CLIFF, NOT A SLOPE
# ============================================================================
# Peak XTX under a real 60k-depth generation with MTP on, c=262144, f16 KV:
#     ts 45/55   22,477 MiB
#     ts 46/54   22,632 MiB
#     ts 47/53   22,774 MiB
#     ts 48/52   23,040 MiB
#     ts 49/51   23,148 MiB   <- SHIPPED, 1,052 MiB under the 24,200 ceiling
#     ts 50/50   24,290-24,365 MiB   <- KILLED by the watchdog, 3 separate runs
# 45 -> 49 costs 671 MiB across four ratio points. 49 -> 50 costs 1,142 MiB in
# ONE. Do not interpolate near the top; 50/50 also failed at c=245760 (24,461).
#
# WHY THE MOST XTX-WEIGHTED RATIO THAT FITS: raw decode rises monotonically with
# XTX share. RAW (MTP OFF), 4 reps each at 60k depth, sd 0.008-0.015:
#     ts 45/55   18.67 t/s
#     ts 47/53   18.80 t/s
#     ts 48/52   18.95 t/s   <- MEASURED
#     ts 49/51   ~19.1 t/s   <- EXTRAPOLATED from the +0.7%/step trend, NOT measured
# If you want a measured number instead of an extrapolated one, set
# --tensor-split 48,52: it costs ~0.8% decode and buys 108 MiB more margin.
#
# IGNORE MTP THROUGHPUT WHEN COMPARING RATIOS. The same four configs measured
# 47.22 / 37.92 / 36.28 / 40.96 t/s with MTP on -- an ordering that is pure draft
# acceptance noise. Identical configs produced 43.88 and 51.45 on separate runs.
# Raw decode reproduces to +/-0.01; MTP does not reproduce at all.
#
# ============================================================================
# WHAT THIS COSTS YOU
# ============================================================================
# ~19 t/s raw at 60k depth. The ROCm Q6_K decode profile does 29.62 at the same
# depth (preset-38-27b-q6k-rocm-tensor-decode-58-42-188k.sh). Full 262144 on
# Q8_0 weights at f16 KV costs roughly 36% of decode, because fitting it forces
# an R9700-weighted split and that card has ~2/3 the bandwidth of the XTX.
#
# Pick this when you need the whole context window at full KV precision on Q8_0
# weights. Pick the ROCm Q6_K profile when you want speed.
#
# MEASURED DEPTH BEHAVIOUR (Q6_K, both backends, 4 reps per cell, raw decode):
#     depth    ROCm f16   VK f16    ROCm q8_0   VK q8_0
#     16k       31.91      28.43      30.28      28.38
#     32k       30.94      27.48      28.67      27.73
#     60k       29.62      26.25      26.14      26.69
#     90k       29.09      25.87      25.22      26.35
# With f16 KV, ROCm leads at every depth and never crosses over. With q8_0 KV,
# ROCm decays much faster (-16.7% vs -8.8% from 16k to 90k) and Vulkan overtakes
# it around 50k. Any "Vulkan beats ROCm at depth" claim is q8_0-specific.
# ============================================================================

#!/usr/bin/env bash

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# The PR build links its libs from a directory that no longer exists in its
# baked-in RUNPATH; point the loader at the real location or the server dies
# instantly with "libllama-server-impl.so: cannot open shared object file".
export LD_LIBRARY_PATH="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROFILE_NAME="preset-38-27b-q8-mtp-vulkan-TENSOR-FULL-256k-f16"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
# +3.3% tg d0 / +2.5% tg d65536 on this model, prefill flat (2026-08-29, r=3,
# devices pinned). Upstream reports -8% on DENSE models -- measured positive here.
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576
LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

# n3 confirmed on this model family; n_max>3 does not pay because the MTP head
# is trained one layer deep (gguf: qwen35.nextn_predict_layers = 1), so every
# draft token past the first is an extrapolation. Per-position acceptance
# measured (0.787, 0.585, 0.421). n_max>8 also CRASHES under -sm tensor
# (ggml-backend-meta.cpp:1576) while working fine under -sm layer.
LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
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
    --split-mode tensor
    --main-gpu 1
    --tensor-split 49,51
    --batch-size 2048
    --no-mmap
    # Vision on the CPU (2026-11-22): the tower stays in system RAM, so the XTX budget above is untouched.
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --no-mmproj-offload
    # --reasoning-preserve REMOVED: the PR #25051 base (2026-06-29) predates
    # this flag and rejects it with "invalid argument".
)
