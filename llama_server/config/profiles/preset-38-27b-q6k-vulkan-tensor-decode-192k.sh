# ============================================================================
# VULKAN TENSOR-PARALLEL, Q6_K -- FAST + LONG CONTEXT (q8_0 KV). 2026-08-20
# ============================================================================
# The decode-oriented split at 192k of context. Ships ALONGSIDE the f16 variant
# (preset-38-27b-q6k-vulkan-tensor-decode-f16-144k.sh), not instead of it:
#     THIS ONE   q8_0 KV   c 196608   192k of context, 10.7% less prefill at depth
#     f16 twin   f16  KV   c 147456   144k, full-precision KV, faster prefill
# Decode is a tie between them. Choose on context vs KV precision, nothing else.
#
# q8_0 KV is what makes 192k reachable at all: it holds 192k in LESS memory than
# f16 holds 128k (20,024 MiB vs 20,839). f16 cannot reach 160k here at any
# decode-friendly split -- c163840 with f16 was killed at 24,290 MiB whole-card.
#
# NON-STANDARD BUILD -- uses llama.cpp-vulkan-tp (git worktree, branch pr25051),
# NOT llama.cpp/build-vulkan. Stock Vulkan CANNOT tensor-split: -ts is silently
# ignored and every weight lands on device 0. Rebuild recipe:
#   cd llama.cpp && git fetch origin pull/25051/head:pr25051
#   git worktree add ../llama.cpp-vulkan-tp pr25051
#   cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# MODEL: unsloth/Qwen3.8-27B-GGUF, Qwen3.8-27B-Q6_K.gguf, 21.30 GiB, imatrix.
#
# ============================================================================
# THE THREE Q6_K VULKAN TENSOR PROFILES
# ============================================================================
#   decode q8_0   ts 70/30  c 196608  ub  512  q8_0 KV   697 pp / 28.65 tg
#   decode f16    ts 68/32  c 147456  ub  512  f16  KV   718 pp / 28.37 tg
#   context f16   ts 50/50  c 262144  ub 1024  f16  KV   889 pp / 23.17 tg
# pp/tg above are UNSPECULATED llama-bench figures (see the note on measurement
# below). The two decode profiles differ only in KV precision and what that buys:
# q8_0 reaches 192k, f16 stops at 144k.
#
# ============================================================================
# MEASURE DECODE WITH MTP OFF. MTP NUMBERS ARE NOT COMPARABLE.
# ============================================================================
# Draft acceptance is stochastic at temp 1.0 and ranged **47% to 68% across
# repeat runs of identical configs** during this tuning session. Decode t/s
# tracks acceptance closely, so the same config produced 49.3 and 56.7 t/s on
# consecutive runs. Any config-vs-config decode comparison taken with MTP on is
# measuring the sampler, not the configuration.
#
# All pp/tg numbers in these headers are therefore llama-bench with speculation
# OFF, -r 3. MTP is a multiplier applied afterwards and roughly doubles decode in
# practice (~50-56 t/s observed on the live server), but that range is indicative
# only -- do not use it to choose between configs.
#
# UNSPECULATED, -sm tensor, ub 512, -r 3, mg 1:
#     ts 70/30  f16   d0      pp 710.56 +/- 0.43   tg 29.11 +/- 0.28
#     ts 70/30  q8_0  d0      pp 696.78 +/- 1.16   tg 28.65 +/- 0.28
#     ts 68/32  f16   d0      pp 717.81 +/- 1.24   tg 28.37 +/- 0.14
#     ts 70/30  f16   d32768  pp 572.20 +/- 0.72   tg 27.26 +/- 0.27
#     ts 70/30  q8_0  d32768  pp 511.02 +/- 2.66   tg 27.38 +/- 0.17
# READ THIS OFF THE TABLE: q8_0 KV is decode-NEUTRAL (a tie at depth, -1.6% at
# d0) and costs 10.7% of prefill at 32k depth. It buys context, not speed. An
# earlier version of this header claimed q8_0 won on decode; that was MTP
# acceptance noise and it was wrong.
#
# ============================================================================
# HOW TO SIZE THIS CARD: USE SERVER-ONLY VRAM, NOT THE WHOLE-CARD NUMBER
# ============================================================================
# The XTX drives the desktop, so `mem_info_vram_used` is server + desktop, and
# the desktop is NOT a constant: it moved 1,696 -> 2,332 MiB during one tuning
# session (Firefox, VS Code, a qemu VM), and this repo has recorded 3.0 GiB.
# Sizing from whole-card readings taken at different times makes one config look
# irreproducible -- ts 70/30 @ 147456 measured 23,092, then 24,034, then 24,282
# MiB whole-card. Nothing about the server changed; the desktop grew under it.
#
# THE RULE:   server_only + worst_case_desktop <= 24,200
#             server_only <= 21,200 MiB   (allowing 3.0 GiB of desktop)
# Measure server_only as (peak whole-card under sustained load) - (baseline with
# no server running, sampled immediately before launch).
#
# server_only, ts 70/30, ub 512, real 512-token generation:
#     f16   c 131072   20,839 MiB
#     f16   c 147456   21,564 MiB   <- over budget at ts 70/30; needs ts 68/32
#     f16   c 163840   KILLED by the watchdog at 24,290 whole-card
#     q8_0  c 163840   19,268 MiB
#     q8_0  c 196608   20,024 / 20,114 MiB   (two runs, 90 MiB apart)
#     q8_0  c 229376   20,810 MiB
#
# ============================================================================
# WHY ts ~70/30 -- DECODE HAS AN INTERIOR MAXIMUM
# ============================================================================
# llama-bench, -sm tensor, ub 1024, f16 KV, -p 8192 -n 256, r=2:
#     ts 50/50  pp 889.60  tg 23.17      ts 70/30  pp 729.82  tg 29.15  <- peak
#     ts 60/40  pp 811.35  tg 25.63      ts 75/25  pp 701.73  tg 29.14
#     ts 65/35  pp 770.92  tg 27.47      ts 80/20  pp 670.47  tg 28.09
#                                        ts 85/15  pp 642.86  tg 26.23
# Past 70/30 the XTX does too much of the work to be helped by the R9700.
#
# WHY ub 512: at ts 70/30 c131072, ub 512 peaked 22,437 MiB whole-card vs 23,385
# for ub 1024, for 2.7% less prefill. For a decode profile that trade is right;
# the 256k context sibling keeps ub 1024 because prefill is its point. Do not
# "unify" the two.
#
# EXPERIMENTAL: PR #25051 is an unmerged DRAFT based on 2026-06-29. Quantized KV
# is documented as unsupported under -sm tensor; this build accepts and runs it.
# ============================================================================

#!/usr/bin/env bash

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700. --main-gpu does NOT cover the vision tower; clip.cpp only
# honours this env var and otherwise grabs the FIRST GPU, the constrained XTX.
export MTMD_BACKEND_DEVICE=Vulkan1

# This build's binaries were linked in /tmp/llamacpp-vkpr and carry a baked-in
# RUNPATH pointing there; that directory is gone. The libs live next to the
# binary, so point the loader at them. Without this the server dies instantly:
#   error while loading shared libraries: libllama-server-impl.so
export LD_LIBRARY_PATH="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROFILE_NAME="preset-38-27b-q6k-mtp-vulkan-TENSOR-DECODE-q8kv-192k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q6_K.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=196608
LLAMA_UBATCH=512

# See the Q8_0 sibling for the full derivation: a cached prompt state carries
# --ctx-checkpoints copies of the recurrent state, and on this hybrid model the
# default of 32 makes every entry so large that server_prompt_cache::alloc()
# silently refuses to cache anything, costing a full reprocess on every return.
LLAMA_CACHE_RAM=24576

# q8_0 KV is the load-bearing choice in this profile -- see the sizing block
# above. It is what makes 192k fit at all.
LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

# n3 confirmed on THIS quant, not inherited. Measured at ts 70/30 c 131072:
#     n_max 3   tg 54.52   acceptance 59.7%   mean accepted length 2.79
#     n_max 4   tg 51.39   acceptance 51.0%   mean accepted length 3.04
# n4 drafts a longer chain and lands a smaller fraction of it.
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
    # Parks compute buffers + projector on the roomier R9700.
    --main-gpu 1
    --tensor-split 70,30
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    # --reasoning-preserve REMOVED: the PR #25051 base (2026-06-29) predates
    # this flag and rejects it with "invalid argument".
)
