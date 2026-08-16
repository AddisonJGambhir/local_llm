#!/usr/bin/env bash
# Qwen3.6 27B Q4_K_M — VULKAN dual-GPU, f16 KV, 256k. Decode-leaning of the two
# 256k profiles.
#
# Measured 2026-08-16 (6228-token prefill, f16 KV, 256k):
#   ROCm   dual ts 1,1     1407 pp   36.9 tg   <- preset "Qwen 3.6 27B" on the ROCm menu
#   Vulkan dual ts 32,33    801 pp   46.6 tg   <- this profile
#   Vulkan dual ts 40,25    790 pp   45.3 tg
#   Vulkan dual ts 48,17    does not fit (XTX over ceiling)
# Vulkan trades 43% of prefill for 26% more decode at the same context and KV
# precision. Pick by whether your turns are read-heavy or write-heavy.
#
# ts 32,33 (balanced) beats the XTX-heavy splits here, which is the OPPOSITE of
# the 128k result for this model. At 256k the f16 KV is large enough that
# crowding the XTX costs more than the split does.
#
# n-max 2, not the 4 previously carried over from a single-GPU note: measured
# 2 on both backends for this model on the dual-GPU topology.
#
# If you want maximum decode and can live with 128k and q8_0 KV, use
# preset-36-27b-q4km-vulkan-xtx-only-128k.sh instead: 70.4 tg by keeping the
# whole model on one card and avoiding the per-token cross-card round trip.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-36-27b-q4km-vulkan-dual-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=2

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="auto"

LLAMA_STARTUP_TIMEOUT=300
LLAMA_TIMEOUT=3600

LLAMA_EXTRA_ARGS=(
    --split-mode layer
    --tensor-split 32,33
    --main-gpu 1
    --batch-size 2048
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
