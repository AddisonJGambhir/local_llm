# ============================================================================
# Preset 15 = the CONVERSATIONAL preset. Same build as preset 14, smaller
# context allocation, much larger ubatch. For the 15-18K band real work sits in.
#
# The 200,704 allocation is what caps ubatch at 192 -- not prefill itself.
# Dropping to ctx 65536 frees VRAM on both cards and buys ubatch headroom.
#
# MEASURED 2026-09-02, llama.cpp-qsagather, ts 400,600, temp 1.0 / top_k 20 /
# top_p 0.95, seed 42, 1 warmup + 3 measured, medians, ignore_eos, n_predict 128:
#
#   @15,375 tokens          pp        tg      R9700     XTX
#     ub 192 (= preset 14) 605.7     38.20    28,726   21,774
#     ub 512               814.7     38.47    29,182   22,241
#     THIS (ub 1024)       973.2     38.11    30,157   23,233
#                          ^ +60.7% prefill, decode flat
#
#   @61,426 tokens (a FILLED 65536, validated -- not extrapolated):
#     THIS (ub 1024)       723.2     29.53    30,081   22,940
#
# XTX peaked at 23,233 of a 24,200 guard ceiling and did NOT grow with depth:
# compute buffers are allocated at load, so 22,940 at 61K is below the 23,233
# seen at 15K. 1,260 MiB of headroom at full depth.
#
# WHY NOT ub 2048 -- do not retry this:
#   Measured ubatch cost on the XTX is 1.46 MiB/unit (192->512) rising to
#   1.94 MiB/unit (512->1024). 1024->2048 projects to ~+1,987 MiB, i.e.
#   ~25,220 MiB against 24,560 MiB of PHYSICAL XTX. That is a load-time spill
#   on the card that drives the display, and the VRAM guard only arms AFTER
#   server readiness so it cannot stop it. The arm was killed during load.
#   It is not reachable by shifting --tensor-split either: both cards pay the
#   ubatch cost, and the R9700 is already at 30,157 of 32,500 at ub1024.
#   ub 1536 projects to ~24,220 -- over the guard ceiling for ~+5%. Not worth it.
#
# WHEN TO USE PRESET 14 INSTEAD: anything beyond ~60K tokens. This preset
# allocates 65536 and will not hold a longer conversation.
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device).
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-conv-65k-ub1024"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qsagather/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=65536
LLAMA_UBATCH=1024
LLAMA_CACHE_RAM=8192

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="none"
LLAMA_DRAFTER=""

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="on"

LLAMA_STARTUP_TIMEOUT=420
LLAMA_EXTRA_ARGS=(
    --split-mode layer
    --tensor-split 400,600
    --batch-size 2048
    --flash-attn on
    --load-mode mmap
    --tensor-read-lazy on
    -ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'
)
