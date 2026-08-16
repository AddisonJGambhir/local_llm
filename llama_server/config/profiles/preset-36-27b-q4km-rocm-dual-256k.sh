#!/usr/bin/env bash
# Qwen3.6 27B Q4_K_M — ROCm dual-GPU, MTP, f16 KV, 256k.
#
# REPLACES the old single-GPU preset, which predated the R9700: it ran
# LLAMA_DEVICE="" (one card), was capped at 128k, and had LLAMA_TIMEOUT=0 so
# large pastes died with an empty 400.
#
# Measured 2026-08-16 (6223-token prefill, f16 KV, batch 6144, ubatch 512):
#   128k  n-max 2   1408 pp   36.27 tg  acc 67.9%   XTX 15057 / R9700 14809
#   256k  n-max 2   1407 pp   36.87 tg  acc 69.8%   XTX 19702 / R9700 20443  <- shipped
#   256k  n-max 3   1404 pp   37.41 tg  acc 62.3%   XTX 19780 / R9700 20515
#
# Context is free here (1408 pp at both 128k and 256k) so 256k ships. n-max 2 and
# 3 are within noise on decode; 2 has clearly better acceptance (69.8 vs 62.3%)
# and matches the archived single-GPU finding for this model on ROCm.
#
# This is the FASTEST PREFILL of any model on this box (1408 t/s vs 982 for
# Qwen3.8 27B Q8 and 986 for Muse) because it is Q4_K_M — roughly half the bytes
# to read per token. Worth keeping despite the lower quant.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-36-27b-q4km-rocm-dual-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-27B-MTP-Q4_K_M.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
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
    --device ROCm0,ROCm1
    --split-mode layer
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 6144
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
