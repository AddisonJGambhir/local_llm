#!/usr/bin/env bash
# Nemotron 3.5 Lightning 30B-A3B Q8_0 — ROCm dual-GPU, FULL 1M CONTEXT, no spec.
#
# Measured 2026-08-16: 2555 pp / 94.66 tg, XTX 19857 / R9700 29430 MiB.
# Decode is identical to the 256k profile; 1M costs ~16% of prefill only.
#
# --tensor-split 18,34 is REQUIRED at this context. At ts 1,1 a 1M KV puts
# 24426 MiB on the XTX, over the ~24200 ceiling that card needs because it also
# drives the desktop. ts 22,30 also trips the guard. 14,38 fails to load (the
# R9700 runs out). The usable window is roughly 18-20 layers on the XTX.
#
# Unsloth publish `-b 1024 -ub 256` for 1M on a SINGLE 32 GiB card. That
# constraint does not apply here: with 56 GiB across two cards the normal
# batch 6144 / ubatch 512 tuning holds and there is no reason to shrink it.
#
# MTP MUST STAY OFF — see the 256k sibling profile for the evidence (~37% of
# generations hang with speculation enabled).

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-nemotron35-30b-a3b-q8-rocm-dual-1m"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=1048576
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

# See the header: MTP hangs ~37% of generations on this model. Do not enable.
LLAMA_SPEC_MODE="none"

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
    --tensor-split 18,34
    --main-gpu 1
    --batch-size 6144
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
