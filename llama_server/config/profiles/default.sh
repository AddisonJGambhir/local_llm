#!/usr/bin/env bash
# Shared defaults sourced by EVERY profile. This is not a tuned config -- it is a
# single-GPU Vulkan fallback plus the values presets inherit when they do not
# override them. Change with care: a change here silently moves any profile that
# relies on the default.
# Model was repointed 2026-08-15 (it named a deleted IQ4_NL quant, so a bare
# `llamactl start` failed). For real tuning use the presets in this directory.


PROFILE_NAME="shared-defaults-single-gpu-vulkan-fallback"

# Mainline serves qwen35moe MTP on Vulkan as of 10358; BeeLlama measured within
# 0.4% (206.0 vs 204.9 t/s) and is 2 days older, so mainline is preferred.
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-Q8_0.gguf"
LLAMA_ALIAS="local"

LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_GPU_LAYERS=99
LLAMA_CONTEXT=262144
LLAMA_THREADS="auto"
LLAMA_BATCH_THREADS="auto"
LLAMA_PARALLEL=1
# Measured 2026-08-11 on a 3080-token prompt: 192 -> 512 lifts MoE prefill from
# 1319.6 to 2332.0 t/s (+77%) with decode unchanged. 1024 reaches 2789 t/s but
# leaves only 0.83 GiB free on the card and 1.03 GiB in GTT -- too tight when the
# desktop shares the GPU. At 512: 20.53 GiB resident, 1.40 GiB free, 0.52 GTT.
LLAMA_UBATCH=1024

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"
# Host-RAM prompt cache (MiB). Hybrid DeltaNet models can only resume from stored
# checkpoints; the 8192 default prunes them to ~3 and forces full re-prefill when
# a harness rewrites mid-conversation history.
LLAMA_CACHE_RAM=24576
LLAMA_FLASH_ATTN="on"
# This hybrid DeltaNet context does not support KV shifting; the server disables it.
LLAMA_CONTEXT_SHIFT=0

LLAMA_SPEC_MODE="mtp"
# Measured optimum for this MoE is 3 (204.9 t/s vs 196.9 at 2, 199.9 at 4).
# MTP beats DFlash here by 22% and uses less VRAM, so MTP stays.
LLAMA_SPEC_DRAFT_N_MAX=2
LLAMA_SPEC_DRAFT_P_MIN=0.0
LLAMA_SPEC_DRAFT_TYPE_K="q8_0"
LLAMA_SPEC_DRAFT_TYPE_V="q8_0"
LLAMA_DRAFTER=""

LLAMA_TEMPERATURE=0.6
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0

# --reasoning replaces the old enable_thinking template kwarg.
LLAMA_REASONING="on"
LLAMA_CHAT_TEMPLATE_KWARGS='{"preserve_thinking":true}'

# Localhost is the secure default. Set an API-key file before exposing another host.
LLAMA_HOST="127.0.0.1"
LLAMA_PORT=1234
LLAMA_API_KEY_FILE=""
# Was 0, which llama-server reads as a ZERO-second socket read/write timeout:
# any request body too large for one read dies with an empty 400, nothing logged.
# Fixed in the four dual presets on 2026-08-15 but NOT here -- and since every
# preset sources this file, 0 was still the inherited default for anything that
# forgot to override it. Safe default belongs here, not patched in four places.
LLAMA_TIMEOUT=3600
LLAMA_JINJA=1
LLAMA_METRICS=1

LLAMA_STARTUP_TIMEOUT=300
LLAMA_STOP_TIMEOUT=30
LLAMA_LOG_MAX_BYTES=52428800
LLAMA_LOG_BACKUPS=5
LLAMA_SYNC_CLIENTS=1

LLAMA_EXTRA_ARGS=()
