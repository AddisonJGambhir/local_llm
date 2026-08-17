#!/usr/bin/env bash
# hipfire preset: Qwen3.8 27B, MQ4R "Redline" speed SKU, single R9700.
#
# Registry entry (`hipfire list --remote`):
#   qwen3.8:27b-fast   14.98 GB
#   "Speed SKU of qwen3.8:27b, 0.68 GB smaller than the quality trunk. The
#    .mq4r suffix marks it an MQ4R Redline SKU, so a single-GPU load on
#    gfx1100/gfx1151/gfx1201 takes the automatic Redline PM4 route."
#
# The sibling quality trunk is `qwen3.8:27b` (15.66 GB, plain MQ4). Pull that
# instead if output quality matters more than throughput.
#
# *** THIS IS A 4-BIT MODEL. ***
# The llama.cpp side of this box runs Qwen3.8-27B at Q8_0 with f16 KV. hipfire
# has no 8-bit path for this model: the registry ships only mq4/mq4r, and
# `hipfire quantize` from a GGUF is restricted to hf4/hf6/mq4/mq6. `q8` exists
# but is safetensors-input only and documented as "reference / debug".
# Any comparison against the llama.cpp presets is therefore NOT like-for-like.

source "$HIPFIRE_SERVER_ROOT/config/env.sh"

PROFILE_NAME="hipfire-qwen38-27b-mq4r-r9700"
HIPFIRE_MODEL="qwen3.8:27b-fast"

# Single GPU. Do not set --tp here:
#   * `--tp` is EXPERT parallelism, wired only for MiniMax-M2 (arch 10) and
#     DeepSeek V4 Flash (arch 9) via load_model_ep. Qwen3.8 is neither.
#   * Pipeline parallel (`params.pp`) is wired only for Qwen3.5/3.6 HFQ
#     (arch_id 5 dense, 6 MoE/A3B) via load_qwen35_pp -- also not Qwen3.8.
#   * admissions.yml carries NO multi-GPU records; the only earned row is
#     single-GPU pp=tp=1.
#   * upstream states plainly: "No speedup is promised" for models that fit on
#     one card, and their 2x gfx1100 station measured SLOWER under PP=2.
HIPFIRE_TP=1

# Sequential behaviour, matching how the llama.cpp side runs (--parallel 1).
# Raise only if you actually want concurrent lanes.
HIPFIRE_CONTINUOUS_BATCH_SIZE=1

# 0 disables idle eviction, so the model stays resident between requests
# instead of paying a reload. Set to e.g. 900 if you want VRAM released when
# idle -- relevant here because llama-server wants the same card.
HIPFIRE_IDLE_TIMEOUT=0

# Leave unset to take hipfire's default. `contiguous` and `vmm` are the two
# documented backends; no measurement exists on this host yet.
HIPFIRE_KV_MODE=""
HIPFIRE_KV_BACKEND=""
