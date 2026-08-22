# ============================================================================
# VULKAN TENSOR-PARALLEL, Q6_K -- MAX CONTEXT + MAX PREFILL. MEASURED 2026-08-20
# ============================================================================
# One of three Q6_K Vulkan tensor profiles. Same model, same build, different
# corner of the tradeoff (pp/tg are UNSPECULATED llama-bench, -r 3):
#     THIS ONE      ts 50/50  c 262144  ub 1024  f16  KV   889 pp / 23.17 tg
#     decode q8_0   ts 70/30  c 196608  ub  512  q8_0 KV   697 pp / 28.65 tg
#     decode f16    ts 68/32  c 147456  ub  512  f16  KV   718 pp / 28.37 tg
# Pick this one for the full 256k or for time-to-first-token on a cold read: its
# prefill is 24% ahead of either decode profile. Pick a decode profile for
# interactive/agent work.
#
# DO NOT COMPARE THESE ON MTP DECODE. Draft acceptance ranged 47-68% across
# repeat runs of identical configs at temp 1.0, and decode t/s tracks it; the
# same config produced 49.3 and 56.7 t/s on consecutive runs. With MTP on, this
# profile runs ~43 t/s live and the decode profiles ~50-56, but those figures are
# indicative only. Every number in this header is speculation-OFF.
#
# NON-STANDARD BUILD -- uses llama.cpp-vulkan-tp (git worktree, branch pr25051),
# NOT llama.cpp/build-vulkan. Stock Vulkan CANNOT tensor-split: -ts is silently
# ignored and every weight lands on device 0. Rebuild recipe:
#   cd llama.cpp && git fetch origin pull/25051/head:pr25051
#   git worktree add ../llama.cpp-vulkan-tp pr25051
#   cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# MODEL: unsloth/Qwen3.8-27B-GGUF, Qwen3.8-27B-Q6_K.gguf, 21.30 GiB, imatrix.
# 5.74 GiB lighter than Q8_0, and that saving is what buys FULL 262144 context
# here -- the Q8_0 Vulkan tensor profile tops out at 200704.
#
# ============================================================================
# SPLIT SWEEP (llama-bench, -sm tensor, ub 1024, f16 KV, -p 8192 -n 256, r=2)
# ============================================================================
#     ts 50/50   pp 889.60   tg 23.17   <- pp peak, SHIPPED HERE
#     ts 60/40   pp 811.35   tg 25.63
#     ts 65/35   pp 770.92   tg 27.47
#     ts 70/30   pp 729.82   tg 29.15   <- tg peak, shipped in the decode profile
#     ts 75/25   pp 701.73   tg 29.14
#     ts 80/20   pp 670.47   tg 28.09
#     ts 85/15   pp 642.86   tg 26.23
# Prefill falls monotonically as work moves to the XTX; decode has a real
# interior maximum at 70/30 and DECLINES past it.
#
# ============================================================================
# WHY NOT SHIFT TOWARD THE XTX AT FULL CONTEXT: THERE IS A CLIFF AT ts 55
# ============================================================================
# 50/50 leaves ~1,700 MiB of XTX headroom at c=262144, which looks like room to
# buy decode by moving weights to the XTX. It is not. Measured, c=262144:
#     ts 50/50   peak XTX 22,782 MiB   pp 671   tg 43.41 (MTP)  <- SHIPPED
#     ts 55/45   peak XTX 24,180 MiB   pp 140   tg 27.00   <- COLLAPSE
# Five split points cost 1,398 MiB and put the XTX into GTT spillover. Note the
# failure mode: NOT an OOM and NOT a crash. The server starts, serves, and
# quietly runs at 21% of prefill and 62% of decode. If you tune by speed alone
# you will read this as "tensor split got slower" and never see the cause.
# Hold this card at or under 24,000 MiB. Measured either side of that line at
# c=262144: 23,953 MiB still ran clean (tg 44.76), 24,180 MiB collapsed.
#
# ============================================================================
# WHY ub 1024 (and why the decode profile uses 512)
# ============================================================================
# llama-bench pp8192 at ts 50/50, --main-gpu 1:
#     ub  512   pp 823.91   tg 23.59
#     ub 1024   pp 882.46   tg 23.61   <- SHIPPED
#     ub 2048   pp 910.88   tg 23.54
# ub 2048 is 3.2% faster on prefill and was REJECTED on memory: at c=262144 it
# peaked at 23,953 MiB versus 22,782 for ub 1024. That is +1,171 MiB for +3.2%
# prefill, and 23,953 is essentially AT the 24,000 ceiling -- 47 MiB of margin,
# and the desktop alone moves more than that.
# --main-gpu 1 does NOT park all of the ubatch compute buffer on the R9700;
# roughly a GiB of it still lands on the XTX. Measured, not assumed.
#
# ============================================================================
# MEASURED HERE (probe: 3,242-token prompt, 512-token generation, MTP n3)
# ============================================================================
#     prefill (llama-bench pp8192)     882.46 t/s   (unspeculated)
#     decode                            23.17 t/s   (unspeculated, the comparable number)
#     decode with MTP                  ~43    t/s   (live, indicative only -- see above)
#     peak XTX 22,782 MiB / R9700 20,792 MiB   (working ceiling is 24,000)
# For reference, the Q8_0 Vulkan tensor profile at its own best settings does
# 947 pp / 40.29 tg at 200704. This trades 7% of prefill for 8% of decode and
# 61,440 more tokens of context.
#
# EXPERIMENTAL: PR #25051 is an unmerged DRAFT based on 2026-06-29, so it lacks
# later fixes and rejects flags added since (see --reasoning-preserve below).
# Re-measure when it lands in master.
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

PROFILE_NAME="preset-38-27b-q6k-mtp-vulkan-TENSOR-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q6_K.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024

# See the Q8_0 sibling for the full derivation: a cached prompt state carries
# --ctx-checkpoints copies of the recurrent state, and on this hybrid model the
# default of 32 makes every entry so large that server_prompt_cache::alloc()
# silently refuses to cache anything, costing a full reprocess on every return.
LLAMA_CACHE_RAM=24576
LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

# n3 confirmed on THIS quant, not inherited. Measured at ts 70/30 c 131072:
#     n_max 3   tg 54.52   acceptance 59.7%   mean accepted length 2.79
#     n_max 4   tg 51.39   acceptance 51.0%   mean accepted length 3.04
# n4 drafts a longer chain and lands a smaller fraction of it; the extra
# verification does not pay for itself.
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
    --tensor-split 50,50
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    # --reasoning-preserve REMOVED: the PR #25051 base (2026-06-29) predates
    # this flag and rejects it with "invalid argument".
)
