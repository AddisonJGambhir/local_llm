#!/usr/bin/env bash
# Tuned mixed-RDNA ROCm preset: Qwen3.6 35B-A3B Q8_0 (MoE), built-in MTP,
# native 256k context, BF16 vision projector.
#
#   ROCm0 = RX 7900 XTX (gfx1100, 24 GiB)  <- ALSO DRIVES THE DESKTOP
#   ROCm1 = AI PRO R9700 (gfx1201, 32 GiB)
#
# Re-tuned 2026-08-15 with a ~45-config staged sweep measured through the real Pi
# harness. Decode was ranked with a fixed seed (tg_clean.py): driving `pi -p`
# re-randomises the seed each run, so the SAME config varies +/-20% as MTP
# acceptance follows whatever text got sampled. Prefill was measured through Pi.
#
#     vs the previously shipped config (ts 20,20 / ub512 / b6144 / q8_0 KV):
#       decode   77.32 -> 81.46 t/s   (+5.4%)
#       prefill  2109  -> 2511  t/s   (+19.1%)
#       XTX peak 23818 -> 23140 MiB   (further from the ceiling, not closer)
#
# *** HARD CONSTRAINT: keep the XTX under ~24200 MiB. ***
# It drives the desktop. At 24388 MiB the compositor was evicted to GTT and the
# screen dropped to ~3 fps. This is NOT an OOM boundary -- llama.cpp will happily
# load past it and take the session with it. scratchpad/bench/xtx_predict.py
# estimates peak usage before loading; check any new config against it.
#
# Load-bearing values:
#   --tensor-split 16,24  THE key change. Moving 2 layers to the R9700 frees
#                         ~1.8 GiB on the XTX, which is what makes ubatch 1024
#                         AND f16 KV affordable at all. At ts 20,20 that combo
#                         hits 24388 MiB and seizes the desktop. Costs a little
#                         decode (the XTX has ~960 GB/s vs the R9700's ~640) and
#                         buys far more than it costs.
#   --ubatch-size 1024    +20% prefill on its own: 256=1642, 384=1987, 512=2138,
#                         768=2396, 1024=2576. The 512 previously shipped here
#                         was carried over from Qwen3.8 untested -- that model's
#                         curve is flat past 512, this one's keeps climbing,
#                         because its hidden dim is 2048 vs 5120.
#   --batch-size 12288    2048=2112, 4096=2454, 6144=2580, 8192=2516, 12288=2685.
#   -ctk/-ctv f16         faster than q8_0 despite reading 2x the bytes: no
#                         dequant on the decode path. 83.21 vs 80.77 t/s. Only
#                         affordable at ts 18,22 -- see the constraint above.
#   --cache-reuse 256     +3% prefill (2623 -> 2702). 0/128/256/512 tested.
#   n-max 2               1=69.96, 2=81.60, 3=76.93, 4=71.88 t/s. Same answer on
#                         Vulkan. Do NOT copy n-max from the Qwen3.8 profile,
#                         which wants 3 -- its dense decode step is expensive
#                         enough to pay for a deeper draft; this MoE's is not.
#   --parallel 1          hybrid model: parallel 4 inflates recurrent state.
#   MTMD_BACKEND_DEVICE   --main-gpu does not cover the vision tower; clip.cpp
#                         only honours this env var and otherwise grabs the FIRST
#                         GPU, i.e. the constrained XTX.
#   LLAMA_TIMEOUT=3600    common.sh defaults it to 0, which llama-server reads as
#                         a ZERO-second socket timeout: large requests die with
#                         an empty 400 and nothing logged.
#
# Backend choice is a real trade, not a default. Best Vulkan is 93.73 decode /
# 2072 prefill against this profile's 81.46 / 2511, so ROCm wins only when the
# prompt is more than ~19x the generated tokens. Big pastes: ROCm. Long
# generation: use the Vulkan preset.

# RE-SPLIT 2026-08-16: ts 18,22 -> 16,24.
# The desktop's footprint on the XTX grew from ~1.7 GiB to ~3.0 GiB during the
# session, which pushed ts 18,22 to 23293 MiB and into memory pressure -- that
# config now TIMES OUT rather than merely running slow. Re-measured:
#     ts 18,22   TIMEOUT            XTX 23293 / R9700 28915
#     ts 16,24   3124 pp  90.9 tg   XTX 21535 / R9700 30637   <- shipped
#     ts 14,26   2969 pp  92.3 tg   XTX 19313 / R9700 32580   (44 MiB free, unshippable)
# The earlier 2511 pp / 81.5 tg recorded for this profile were depressed by that
# same pressure; with the split relieved it is 24% faster on prefill.
# Lesson: a tensor-split is not a fixed property of the model. It depends on what
# else is resident on the XTX, so re-check it when the desktop changes.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

# --main-gpu does not cover the vision tower; clip.cpp only honours this env var
# and otherwise picks the FIRST GPU, which is the constrained 24 GiB XTX.
export MTMD_BACKEND_DEVICE=ROCm1

PROFILE_NAME="preset-36-35b-a3b-q8-mtp-rocm-dual-f16-256k-mmproj"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.6-35B-A3B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024

# Prompt-cache size in SYSTEM RAM (not VRAM). Checkpoints are much cheaper on
# this model than on Qwen3.8 -- the recurrent state is 251 MiB here vs 598 MiB
# there -- so the 8192 MiB default already holds roughly 3x more of them. Set to
# 16384 for consistency with the Qwen3.8 profile and extra headroom in long
# sessions; host RAM is 59 GiB.
# Raised 16384 -> 24576 on 2026-08-16 after repeated full-context reprocessing
# (119,389 tokens / 96 s, ~8 times in one session). A cached state holds the full
# KV plus checkpoints: with f16 KV this model measures 5.44 GiB of KV at 262144
# tokens, so ~3.9 GiB at 190k, plus --ctx-checkpoints recurrent-state copies. One
# large context fits in 16 GiB; two do not, and Pi juggles several concurrently
# (captured prompt sizes ran 442k -> 198k -> 4k -> 73k chars). Host RAM is 59 GiB
# with ~30 GiB free, so this costs no VRAM.
# The previous 16384 was sized from a 690 MiB entry estimate taken from Qwen3.8's
# recurrent state, which ignored that every entry also carries the KV cache.
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
    --device ROCm0,ROCm1
    --split-mode layer
    --tensor-split 16,24
    --main-gpu 1
    --batch-size 12288
    --cache-reuse 256
    # Each cached prompt state carries --ctx-checkpoints copies of the recurrent
    # state (default 32). Cutting to 8 shrinks every entry substantially so more
    # contexts fit the budget, at the cost of coarser rewind granularity on
    # mid-history edits.
    --ctx-checkpoints 8
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
