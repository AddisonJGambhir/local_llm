#!/usr/bin/env bash
# Muse Glimmer 30B Q8_0 — VULKAN dual-GPU, DFlash, f16 KV, 128k native.
#
# PREFER THE ROCm PROFILE FOR THIS MODEL. Measured 2026-08-16, same harness:
#     ROCm    986 pp   33.23 tg
#     Vulkan  743 pp   31.70 tg
# ROCm is +33% on prefill and slightly ahead on decode, so there is no workload
# here where Vulkan clearly wins — unlike the Qwen3.6 MoE, where Vulkan's decode
# advantage justified shipping both. This profile exists as a fallback.
#
#   --batch-size 2048  batch is a NO-OP on Vulkan: 2048 -> 742.4 pp,
#                      6144 -> 742.9 pp. Third model where the large-batch win is
#                      ROCm-only. Left at stock rather than carrying a no-op.
#   n-max 2            same answer as ROCm: n2=30.1/31.7, n4=28.8, n8=16.7 tg.
#                      Upstream recommends 15; wrong on both backends here.
#
# Everything else is inherited from the ROCm profile — see it for the full
# evidence behind f16 KV, ubatch 512 and tensor-split 1,1.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-muse-glimmer-30b-q8-vulkan-dual-128k-dflash"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Muse-Glimmer-30B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=131072
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="dflash"
LLAMA_DRAFTER="$LLAMA_SERVER_ROOT/models/dflash-kquant.gguf"
LLAMA_DFLASH_SPEC_TYPE="draft-dflash"
LLAMA_SPEC_DRAFT_N_MAX=2

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="auto"

LLAMA_STARTUP_TIMEOUT=300
# common.sh defaults this to 0, which llama-server reads as a ZERO-second socket
# timeout: large request bodies die with an empty 400 and nothing is logged.
LLAMA_TIMEOUT=3600

LLAMA_EXTRA_ARGS=(
    --split-mode layer
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 2048
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
