#!/usr/bin/env bash
# Dual-GPU Vulkan fallback: Qwen3.8 27B Q8_0, MTP, q8_0 KV,
# native 256k context, and BF16 vision. Measured at 786.6 pp / 40.3 tg.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700. --main-gpu does NOT cover the vision tower; clip.cpp only
# honours this env var and otherwise grabs the FIRST GPU, the constrained XTX.
# The sibling ROCm profile measured this as 1.11 GiB of permanent XTX headroom
# plus the whole ~0.9 GiB vision transient, at zero text pp/tg cost.
export MTMD_BACKEND_DEVICE=Vulkan1

PROFILE_NAME="preset-38-27b-q8-mtp-vulkan-dual-q8-256k-mmproj"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=512
# Prompt-cache sizing, revised 2026-08-16 after repeated full-context reprocessing
# (119,389 tokens / 96 s, ~8 times in one session; seen on BOTH Q8 models).
#
# A cached context state = full KV + (--ctx-checkpoints) copies of the recurrent
# state. This model family is hybrid, so those checkpoints are large and the
# default of 32 makes every entry enormous:
#     Qwen3.8-27B   checkpoint 690 MiB (measured from eviction log)
#                   32 x 690 MiB = ~22 GiB of checkpoints + ~8.7 GiB KV at 256k
#     Qwen3.6-35B   checkpoint 251 MiB, KV 5.44 GiB at 256k with f16
# server_prompt_cache::alloc() SILENTLY refuses any entry larger than the whole
# budget (returns nullptr, warns only at trace level), so an over-sized context is
# never cached at all -- and returning to it costs a full reprocess.
#
# --ctx-checkpoints 8 is the load-bearing half of this fix; raising --cache-ram
# alone cannot help if a single entry still exceeds the limit.
# Tradeoff: coarser rewind granularity on mid-history edits. Those cost thousands
# of tokens; the failure being fixed costs 119,000.
LLAMA_CACHE_RAM=24576

# f16 KV, 2026-08-16. Measured on Vulkan (6226-token prefill):
#     ts 28,37  825 pp  35.03 tg  acc 53.8%   XTX 22180 / R9700 29429  <- shipped
#     ts 26,39  827 pp  34.84 tg  acc 53.6%   XTX 20377 / R9700 31231
# Equal on speed; 28,37 leaves ~2.9 GiB on the R9700 instead of ~1.4 GiB. The
# R9700 is not unlimited either -- protecting the XTX by filling the other card
# is not a safe trade.
LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

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
LLAMA_CHAT_TEMPLATE_KWARGS='{"preserve_thinking":true}'

LLAMA_STARTUP_TIMEOUT=420
LLAMA_EXTRA_ARGS=(
    # Each cached prompt state carries --ctx-checkpoints copies of the
    # recurrent state (default 32). On these hybrid models that dominates
    # the entry size and pushes it past --cache-ram, at which point
    # server_prompt_cache::alloc() silently refuses to cache the context
    # at all and returning to it costs a full reprocess.
    --ctx-checkpoints 8
    --split-mode layer
    # Parks compute buffers + projector on the roomier R9700; speed-neutral
    # but keeps ~2.5 GiB off the XTX, which also drives the desktop.
    --main-gpu 1
    --cache-reuse 256
    --tensor-split 28,37
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --reasoning-preserve
)
