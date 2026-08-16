#!/usr/bin/env bash
# Qwen3.8 27B Q8_0 — ROCm dual-GPU, PREFILL-OPTIMISED variant.
#
# Sibling of preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh, which is the
# DECODE-optimised config. Same model, same 256k context, same vision projector;
# the two differ only in KV precision and tensor split, which trade prefill
# against decode:
#
#   profile              KV     split     pp      tg      acc
#   this one (pp-opt)    q8_0   1,1      1067    31.3    49.7%
#   the f16 one (tg-opt) f16    28,37     982    40.4    66.0%   (5-seed median)
#
# So: +9% prefill, -23% decode. Pick per session — read-heavy agent turns favour
# this profile, write-heavy generation favours the f16 sibling.
#
# *** This is a deliberate exception to the "f16 KV everywhere" rule. ***
# f16 raises speculative acceptance (49.7 -> 66.0% here) and is the right default
# on every other model, but it does not fit alongside an even tensor split on
# this one: KV goes 8.7 -> ~17.4 GiB at 256k, which forces layers onto the R9700
# and costs the prefill this profile exists to keep. Do not "fix" the q8_0 below.
#
# Memory: ts 1,1 puts ~22260 MiB on the XTX. That card also drives the desktop,
# whose footprint grew from ~1.7 to ~3.0 GiB during tuning, so this sits closer
# to the ~24200 MiB ceiling than the f16 sibling (22511 at ts 28,37 but spread
# more evenly). Re-check with scratchpad/bench/xtx_guard.sh if the desktop grows.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

# Put the vision tower on the R9700. --main-gpu does NOT cover it: clip.cpp only
# honours the MTMD_BACKEND_DEVICE env var, and with it unset it falls back to
# ggml_backend_init_by_type(GPU), which picks the FIRST GPU = ROCm0 = the 24 GiB
# XTX, i.e. the one card with no room. Measured 2026-08-15:
#   unset  -> XTX 20.61 GiB, and a 4K image pushes it to a 21.27 GiB peak
#   ROCm1  -> XTX 19.50 GiB, and a 4K image leaves the XTX completely untouched
# Costs 17% on image prefill (415 -> 346 t/s) because the XTX has more bandwidth;
# buys 1.11 GiB of permanent headroom plus the whole ~0.9 GiB vision transient on
# the card that was at ~95% full. Text pp/tg are unaffected (1084 vs 1086).
export MTMD_BACKEND_DEVICE=ROCm1

PROFILE_NAME="preset-38-27b-q8-mtp-rocm-dual-256k-ppopt"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=512
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

# f16 KV + ts 28,37, 2026-08-16.  Previously q8_0 at ts 1,1 ONLY because f16 KV
# (8.7 -> ~17.4 GiB at 256k) did not fit with an even split.
#
# Ranked over 5 DIFFERENT SEEDS, not 5 repeats: this harness fixes the seed, so
# repeating a config reproduces the same number (0.1-0.5% spread) and cannot tell
# a real effect from one lucky seed. Varying the seed samples the acceptance
# distribution, which is what actually sets speculative decode rate.
#
#   ts 28,37 f16   pp 982   tg 29.1 - 40.8  median 38.5   acc 42 - 72%
#   ts 26,39 f16   pp 943   tg 36.1 - 38.8  median 37.9   acc 60 - 68%
#   ts 1,1   q8_0  pp 1067  tg 31.3 (single seed)         acc 49.7%
#
# The two splits are EQUIVALENT on decode once the seed varies. An earlier
# single-run comparison showed 28,37 at 29.23 tg and I nearly rejected it on that
# basis -- that run was simply the bottom of its distribution (its 42.0%
# acceptance matches the worst seed exactly). 28,37 wins on prefill and on memory
# balance, so it ships.
#
# Memory is the real constraint and BOTH cards matter:
#   ts 26,39 -> XTX 20711 / R9700 31497 of 32624   (R9700 96% full, unsafe)
#   ts 28,37 -> XTX 22511 / R9700 29695            (~2.9 GiB free on each)
# Protecting the XTX by exhausting the R9700 is not a safe trade.
#
# f16 at a FIXED split is worth +11% decode and +7.4 points of acceptance; the
# prefill cost seen at 26,39 comes from the split, not from f16.
LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="mtp"
LLAMA_SPEC_DRAFT_N_MAX=3
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
# socket read is aborted with an EMPTY 400 (Content-Length: 0, nothing logged
# server-side). Measured 2026-08-15 on this profile:
#   --timeout 0    4K JPEG 2.5 MB: 3/8 succeeded   4K PNG 5.2 MB: 0/4
#   --timeout 3600 4K JPEG 2.5 MB: 8/8 succeeded   4K PNG 5.2 MB: 4/4
# Genuine errors still return a JSON body (e.g. exceed_context_size_error); an
# empty 400 is the signature of this bug. Vision is the main victim, but any
# large paste hits it too.
LLAMA_TIMEOUT=3600
LLAMA_EXTRA_ARGS=(
    # Each cached prompt state carries --ctx-checkpoints copies of the
    # recurrent state (default 32). On these hybrid models that dominates
    # the entry size and pushes it past --cache-ram, at which point
    # server_prompt_cache::alloc() silently refuses to cache the context
    # at all and returning to it costs a full reprocess.
    --ctx-checkpoints 8
    --device ROCm0,ROCm1
    --split-mode layer
    --cache-reuse 256
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 6144
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    --reasoning-preserve
)

# KNOWN LIMITATION — post-image prefill regression (llama.cpp 10358, this host).
# A SINGLE image request permanently drops text prefill ~40% until the server is
# restarted. Decode is unaffected. Measured 2026-08-15:
#     text-only, 10 consecutive prefills : 1083-1090 t/s, no drift
#     after ONE 640x480 image            :  652, 652, 652, 650, 638 t/s
#     after ONE 3840x2160 image          :  650 t/s  (same magnitude)
# Trigger is any image at all, not image size. Ruled out by measurement:
#   - batching        prefill chunk sizes in the log are identical before/after
#   - thermal/clocks  both GPUs idle at 47-49 C, downclocked, 2-3% busy
#   - mmproj device   reproduces with the tower on ROCm0 and on ROCm1
#   - prompt cache    reproduces with --cache-ram 0
#   - GTT spill       procGTT stays 0.00 on both cards throughout
# Consequence: --batch-size 6144 buys +12.3% in text-only sessions but nothing
# after an image (2048 and 6144 both land at ~653). Kept because it is free.
# Workaround until upstream fixes it: restart the server after heavy image use.
