#!/usr/bin/env bash

PROFILE_NAME="muse-glimmer-30b-q5km-vulkan-q8-128k-dflash"

LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Muse-Glimmer-30B-UD-Q5_K_M.gguf"
LLAMA_ALIAS="local"

LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0"
LLAMA_GPU_LAYERS=99
# 128k is the model's native n_ctx_train and costs almost nothing here: 3 of every 4
# layers use sliding-window attention (2048), so only 13 of 52 layers keep KV that
# scales with context. Measured 2026-08-11: 16k=19.12 GiB, 128k=19.99 GiB (+0.87)
# on Vulkan, 20.27 GiB on ROCm, 0.00 procGTT both, throughput unchanged.
# 16k was breaking agentic harnesses (Hermes requires >=64k; opencode/Hermes
# prompts hit "request exceeds the available context size").
LLAMA_CONTEXT=131072
LLAMA_THREADS="auto"
LLAMA_BATCH_THREADS="auto"
LLAMA_PARALLEL=1
LLAMA_UBATCH=512

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"
LLAMA_CACHE_RAM=""
LLAMA_FLASH_ATTN="on"
LLAMA_CONTEXT_SHIFT=1

LLAMA_SPEC_MODE="dflash"
LLAMA_DRAFTER="$LLAMA_SERVER_ROOT/models/dflash-kquant.gguf"
LLAMA_DFLASH_SPEC_TYPE="draft-dflash"
LLAMA_SPEC_DFLASH_CROSS_CTX=""
LLAMA_SPEC_DRAFT_NGL=99
# 15 is DFlash's architectural ceiling (block_size 16 = 1 anchor + 15), not a tuned optimum.
# On this gfx1100 Vulkan build a 16-wide verify pass costs 3.16x a 1-wide pass (82.2ms vs 26.0ms),
# so n_max=15 is a net loss. Measured optimum here is 3-4; AMD's own Vulkan footnotes use 2 and 4.
LLAMA_SPEC_DRAFT_N_MAX=4
LLAMA_SPEC_DRAFT_N_MIN=0

LLAMA_TEMPERATURE=0.7
LLAMA_TOP_P=1.0
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0

LLAMA_REASONING="auto"
LLAMA_CHAT_TEMPLATE_KWARGS=""

LLAMA_HOST="127.0.0.1"
LLAMA_PORT=1234
LLAMA_API_KEY_FILE=""
LLAMA_TIMEOUT=0
LLAMA_JINJA=1
LLAMA_METRICS=1

LLAMA_STARTUP_TIMEOUT=300
LLAMA_STOP_TIMEOUT=30
LLAMA_LOG_MAX_BYTES=52428800
LLAMA_LOG_BACKUPS=5
LLAMA_SYNC_CLIENTS=1

LLAMA_EXTRA_ARGS=()
