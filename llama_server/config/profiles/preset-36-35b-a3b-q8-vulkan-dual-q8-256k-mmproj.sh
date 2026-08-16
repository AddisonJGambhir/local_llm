#!/usr/bin/env bash
# Vulkan preset: Qwen3.6 35B-A3B Q8_0 (MoE), MTP, 256k context, BF16 vision.
# Ships alongside the ROCm preset -- neither backend wins outright.
#
#   Vulkan0 = RX 7900 XTX (24 GiB)  <- ALSO DRIVES THE DESKTOP, keep < ~24200 MiB
#   Vulkan1 = AI PRO R9700 (32 GiB)
#   Vulkan2 = iGPU -- never use
#
# Measured 2026-08-15, decode with a fixed seed (tg_clean.py), prefill through Pi:
#                        decode      prefill    XTX peak
#   this profile         93.73 t/s   2072 t/s   22751 MiB
#   best ROCm            81.46       2511       23140
#
# So Vulkan is +15% decode and -17% prefill. Equal total time when the prompt is
# about 19x the generated tokens; below that ratio this profile wins.
#   Iliad essay   (10.6k in / 2.3k out, ratio  4.6) -> Vulkan, clearly
#   Big paste     ( 13k in /   55 out, ratio  238) -> ROCm, clearly
#
# Vulkan-specific findings:
#   --batch-size 2048  batch size does NOTHING on Vulkan (2048 vs 6144 -> 2437 vs
#                      2430 pp). The large-batch win is ROCm-only, confirmed on
#                      two models. Left at stock rather than carrying a no-op.
#   --ubatch-size 1024 unlike batch, ubatch DOES help here.
#   -ctk/-ctv f16      +17% prefill on Vulkan (1772 -> 2072) at equal decode
#                      (95.13 vs 93.73, inside the 6.9% run spread). Bigger
#                      effect here than on ROCm, where f16 mainly helped decode.
#   --tensor-split 18,22  better than 1,1 on both metrics AND 1.3 GiB safer
#                      (21937 vs 23222 MiB peak at q8_0).
#   n-max 2            1=87.85, 2=95.33, 3=86.08 t/s. Same answer as ROCm.
#
# NOTE on llamactl's SPILL warning: it is WRONG on this backend. It sums
# drm-memory-gtt from /proc/<pid>/fdinfo, which RADV fills with virtual mappings
# -- the same source reported 40 GiB resident on a 24 GiB card. Proof the GTT use
# is not capacity spill: it stayed at 4.52/4.51/4.52 GiB across splits 1,1 ->
# 18,22 -> 14,26 while free XTX VRAM went from 2.23 to 5.72 GiB. RADV exposes a
# 29.71 GiB host-visible heap and the Vulkan backend uses it by design.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

# --main-gpu does not cover the vision tower; clip.cpp only honours this env var
# and otherwise picks the FIRST GPU, which is the constrained 24 GiB XTX.
export MTMD_BACKEND_DEVICE=Vulkan1

PROFILE_NAME="preset-36-35b-a3b-q8-mtp-vulkan-dual-f16-256k-mmproj"
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
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-Qwen3.6-35B-A3B-BF16.gguf"
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

# KNOWN LIMITATION -- post-image prefill regression (llama.cpp 10358).
# Sending a SINGLE image permanently drops text prefill until the server is
# restarted. Confirmed on this model 2026-08-15:
#     clean restart, no image : 2452 t/s prefill, 82.3 t/s decode
#     after ONE image request : 1687 t/s prefill (-31%), 78.5 t/s decode
# The same bug costs Qwen3.8-27B ~40% of prefill. On that model it was traced far
# enough to rule out batching, thermals, mmproj device placement, the prompt
# cache, and GTT spill -- the cause is inside llama.cpp's multimodal path.
# Workaround: restart the server after image use, or run a text-only profile
# (no --mmproj) for sessions that will not send images, which cannot hit it.
