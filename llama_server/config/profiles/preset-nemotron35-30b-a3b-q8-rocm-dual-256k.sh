#!/usr/bin/env bash
# NVIDIA Nemotron 3.5 Lightning 30B-A3B Q8_0 — ROCm dual-GPU, 256k, NO speculation.
#
# *** MTP MUST STAY OFF ON THIS MODEL. ***
# With --spec-type draft-mtp, ~37% of generations HANG: the server marks the slot
# idle, the GPUs go quiet, and the HTTP response is never sent (measured 3 of 8 at
# 256 tokens on a clean server; also reproduced at 512 tokens, at 1M context, and
# with both q8_0 and f16 KV). With speculation disabled: 8 of 8 succeeded at
# 96.1-96.5 t/s. Speculation is not merely unstable here, it is also SLOWER --
# the MTP runs that did complete ranged 77-121 t/s with heavy variance.
# Presumably an interaction between the MTP head and the Mamba/SSM layers.
# Re-test if you move to a newer llama.cpp: b10450 contains
# "#26623 ggml: recurrent state rollback for ggml_ssm_scan".
#
# Architecture: nemotron_h_moe, 52 layers, Mamba/SSM hybrid MoE (128 experts,
# 6 active + 1 shared), most layers carry NO attention KV at all. Consequences:
#   - KV is tiny: 816 MiB at 256k, ~3.3 GiB at 1M.
#   - Recurrent state is only 47.6 MiB (Qwen3.8's is 598 MiB), so this model is
#     largely immune to the prompt-cache-eviction problem that plagues the Qwens.
#   - No SSM CPU fallback observed on ROCm: GPUs 92%/46%, CPU load 2.5.
#
# Measured 2026-08-16 (5975-token prefill, f16 KV, batch 6144, ubatch 512):
#   256k  ts 1,1     3034 pp   94.62 tg    XTX 19873 / R9700 18644   <- this file
#   1M    ts 18,34   2555 pp   94.66 tg    XTX 19857 / R9700 29430   <- sibling profile
# Decode is unaffected by context size; 1M costs ~16% of prefill.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-nemotron35-30b-a3b-q8-rocm-dual-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
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
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 6144
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
