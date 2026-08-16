#!/usr/bin/env bash
# TEXT-ONLY variant of the tuned Vulkan preset for Qwen3.6 35B-A3B Q8_0.
# Identical tuning, minus the vision projector.
#
# Measured 2026-08-16 (decode with a fixed seed, prefill through Pi):
#   this profile      93.96 t/s decode   2129 t/s prefill   XTX peak 22891 MiB
#   with vision       93.73 t/s decode   2072 t/s prefill
#
# The ~2% edge is not the point. The reason to run this is that it CANNOT hit the
# post-image prefill regression: sending one image to the multimodal profile costs
# ~31% of prefill (2452 -> 1687 t/s) until the server is restarted. A server with
# no projector loaded cannot receive an image, so it cannot enter that state.
#
# Dropping --mmproj frees the projector (~0.86 GiB) and its ~248 MiB compute
# buffer from the R9700, NOT from the XTX -- MTMD_BACKEND_DEVICE had already moved
# the vision tower off the constrained card. So this buys no XTX headroom, and the
# ~24200 MiB ceiling on that card still applies.
#
# Use the -mmproj profile when you need images; use this one otherwise.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1


PROFILE_NAME="preset-36-35b-a3b-q8-mtp-vulkan-dual-q8-256k-textonly"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024

# Prompt-cache size in SYSTEM RAM (not VRAM). Checkpoints are much cheaper on
# this model than on Qwen3.8 -- the recurrent state is 251 MiB here vs 598 MiB
# there -- so the 8192 MiB default already holds roughly 3x more of them. Set to
# 16384 for consistency with the Qwen3.8 profile and extra headroom in long
# sessions; host RAM is 59 GiB.
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

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=2
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

# NOT optional. common.sh defaults LLAMA_TIMEOUT to 0, and llama-server reads
# --timeout as a socket read/write timeout in seconds (upstream default 3600),
# not as "disabled". At 0, any request body too large to arrive in a single
# socket read is aborted with an EMPTY 400 and nothing is logged server-side --
# which silently breaks image requests and large pastes.
LLAMA_TIMEOUT=3600

LLAMA_EXTRA_ARGS=(
    # Each cached prompt state carries --ctx-checkpoints copies of the
    # recurrent state (default 32). On these hybrid models that dominates
    # the entry size and pushes it past --cache-ram, at which point
    # server_prompt_cache::alloc() silently refuses to cache the context
    # at all and returning to it costs a full reprocess.
    --ctx-checkpoints 8
    --split-mode layer
    --tensor-split 18,22
    --main-gpu 1
    --batch-size 2048
    --cache-reuse 256
    --no-mmap
    --reasoning-preserve
)

# WHEN THE DESKTOP MOVES TO THE iGPU
# ----------------------------------
# Plugging the monitor into the motherboard frees the ~1.7 GiB the compositor
# holds on the XTX, and removes the reason the ~24200 MiB ceiling exists at all
# (nothing but the server would be left on that card).
#
# What that unlocks, all of which was measured-and-rejected purely on memory:
#   ts 20,20 + ub1024 + f16 KV   hit 24388 MiB -> would land near 22700. This is
#                                the combination that seized the screen; with the
#                                desktop gone it should simply run.
#   ts 21,19 / 22,18             more layers on the higher-bandwidth XTX (~960
#                                GB/s vs ~640). ts 21,19 was the best PP split in
#                                the sweep (2153 vs 2100 at 20,20) before ubatch
#                                and KV type were raised.
#   ts 24,16                     OOM'd outright; may become feasible.
#
# Re-run scratchpad/bench/final3.sh with those splits added, and re-calibrate
# xtx_predict.py by setting DESKTOP_MIB from 1700 to whatever the card idles at.

