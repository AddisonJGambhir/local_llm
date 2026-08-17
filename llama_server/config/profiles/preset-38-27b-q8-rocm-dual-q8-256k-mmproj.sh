#!/usr/bin/env bash
# Tuned mixed-RDNA ROCm preset: Qwen3.8 27B Q8_0 with built-in MTP,
# f16 KV, native 256k context, and the BF16 vision projector.
#
# Hardware/device order on this host:
#   ROCm0 = RX 7900 XTX (gfx1100, 24 GiB)
#   ROCm1 = AI PRO R9700 (gfx1201, 32 GiB)
# The local llama.cpp ROCm build must contain both architectures. A 50/50 layer
# split was fastest; using the roomier R9700 as main GPU also moved scratch/output
# memory off the nearly-full XTX.
#
# Retuned 2026-08-15 (see docs/QWEN38_DUAL_GPU_AUDIT.md). Every value below was
# re-measured against alternatives rather than assumed; only --batch-size moved.
# Ranking used greedy decode: under sampled decode, moving a layer between GPUs
# perturbs kernel numerics, the token stream diverges, and MTP acceptance swings
# 44-63% for reasons unrelated to the config being tested.
#
# Load-bearing values, with what happens if you "tidy" them:
#   --batch-size 6144  968 -> 1087 pp (+12.3%) in a TEXT-ONLY session. See the
#                      post-image caveat at the bottom: once any image has been
#                      processed, 2048 and 6144 both collapse to ~653 and this
#                      setting stops mattering until the server restarts.
#                      5120=1022, 7168=1078, 8192=1085,
#                      4096=1022, 2048=968. A genuine peak, confirmed at four
#                      prompt lengths (3.7k/7.3k/14.6k/29k). ROCm-only: on Vulkan
#                      the same change does nothing (809 -> 802).
#   --ubatch-size 512  re-tested at batch 6144: 384=1082, 512=1087, 640=1043,
#                      768=1069.
#   --tensor-split      SUPERSEDED 2026-08-16: now 28,37 for f16 KV (see below).
#                      Historic note from the q8_0 era follows.
#                      33,31 looks 1.9% faster on the median but one run in three
#                      collapsed to 749 pp, and it leaves the XTX 0.66 GiB free
#                      (vs 1.63) which is too little for the vision transient.
#                      34,30 is worse still. Do not push layers onto the XTX.
#   --main-gpu 1       speed-neutral vs 0 (1086.0 vs 1086.7); kept because it
#                      parks compute buffers + mmproj on the roomier R9700 and
#                      keeps ~2.5 GiB off the constrained XTX.
#   --parallel 1       (from default.sh) this is a hybrid-attention model: at the
#                      default parallel 4 the recurrent-state buffer grows from
#                      598 MiB to 2394 MiB for no benefit at one slot.
#   n-max 3            optimal at BOTH depths. 2/4/5 lose. The verify pass runs
#                      1+n_draft tokens wide; 2->3 costs +2.2 ms/step but 3->4
#                      costs +31.6 ms/step, so 4-wide is the last free step.
#
# Pinned to llama.cpp build 10358 (030ebb558). Upstream 10450 (ece963f41) was
# built and measured here: -4.8% pp, -4.0% tg, and MTP acceptance 61.9% -> 55.7%,
# most likely from "ggml-hip: remove -funsafe-math-optimizations" (#26696).
# Re-benchmark before taking any llama.cpp update.

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

PROFILE_NAME="preset-38-27b-q8-mtp-rocm-dual-f16-256k-mmproj"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
# 512 -> 256 on 2026-08-16, together with ts 1,1. See the DEPTH RETUNE block
# below: ubatch was originally chosen at ~6k prompts, and the ranking inverts
# once there is a real KV cache to attend over.
LLAMA_UBATCH=256
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

# ============================================================================
# DEPTH RETUNE 2026-08-16 (evening):  ts 28,37 / ub 512  ->  ts 1,1 / ub 256
# ============================================================================
# Everything below this block was tuned at ~6k-token prompts with an EMPTY KV
# cache. That is not how this server is used -- real sessions run past 100k,
# where a 135,390-token prompt was logged at 491 t/s against the ~1000 t/s the
# shallow benchmarks reported. Prefill was re-tuned at depth using
# `llama-bench -d 131072 -p 2048`, which prefills 131k and then times a chunk
# on top, i.e. the marginal rate you actually feel.
#
# Measured at d131072 (f16 KV throughout, llama-bench, unspeculated):
#     ts 28,37  ub 512   pp 282.57   tg 14.97      <- previously shipped
#     ts 28,37  ub 1024  pp 251.17
#     ts 28,37  ub 2048  pp 210.74                 (bigger ubatch is WORSE)
#     ts 28,37  ub 384   pp 288.01
#     ts 28,37  ub 256   pp 286.13
#     ts 1,1    ub 512   pp 306.93   tg 15.31
#     ts 1,1    ub 384   pp 314.81
#     ts 1,1    ub 256   pp 315.79   tg 15.26      <- SHIPPED  (+11.8% / +1.9%)
#     ts 20,45  ub 512   pp 243.78                 (KV onto R9700: much worse)
#     ts 40,25  ub 512   OOM, XTX hit 24068        (do not retry)
#
# THE SPLIT MATTERS MORE THAN UBATCH AT DEPTH: 1,1 vs 20,45 is a 26% spread,
# while ubatch 256 -> 512 moves ~2%. At depth the cost is dominated by reading
# the KV cache, and KV lives on the GPU that owns its layers -- so the split is
# really a KV-read balance, not just a weight-memory balance. ts 28,37 was
# chosen for weight balance at zero depth, which is a different problem.
#
# Shallow performance is unaffected (llama-bench, d0):
#     ts 28,37 ub 512   pp2048 1058.22   pp8192 1125.91   tg 20.00
#     ts 1,1   ub 256   pp2048 1068.18   pp8192 1099.06   tg 20.33
# +0.9% / -2.4% / +1.7%. A wash, so this replaces the profile rather than
# shipping as a second preset.
#
# END-TO-END VALIDATION, this exact config with MTP and c=262144:
#     130,001-token prefill  484.6 t/s
#     decode                  29.21 t/s   (draft acceptance 70.5%)
#     PEAK XTX UNDER LOAD    24,150 MiB
#
# *** MEMORY: 150 MiB OF MARGIN. READ THIS BEFORE CHANGING ANYTHING. ***
# The ceiling on this host is 24,300 MiB (the XTX drives the desktop; past it
# the compositor is evicted to GTT and the screen drops to ~3 fps).
#   c=262144 at ts 1,1 allocates 23,927 MiB at load and peaks 24,150 UNDER LOAD.
#   The +223 MiB delta is compute buffers plus the MTP draft context. A
#   load-time-only VRAM check would call this safe when it is 150 MiB away.
# Measured context ladder at ts 1,1 + f16 + ub 256 (load-time peak):
#     196608 -> 21,754     229376 -> 22,845
#     245760 -> 23,389     262144 -> 23,927
# Scaling is linear at 33.2 MiB per 1k tokens.
# If the desktop footprint grows (it has moved 1.7 -> 3.0 GiB in one session
# before), DROP TO 245760 -- that buys 544 MiB back. Moving the display to the
# motherboard iGPU frees ~1,970 MiB and makes this comfortable instead of tight.
#
# CAVEATS, stated plainly:
#   - The 24,150 figure was measured TEXT-ONLY, without --mmproj. The vision
#     tower is pinned to ROCm1 via MTMD_BACKEND_DEVICE and a 4K image was
#     previously measured to leave the XTX untouched, so this should hold --
#     but it has NOT been re-verified at ts 1,1. Watch the XTX on first image use.
#   - The tg figures in the sweep table are UNSPECULATED (llama-bench cannot run
#     MTP). Speculated decode was validated only in the end-to-end run above.
#   - ub 256 sits inside the 65-256 range that upstream discussion #21043 flags
#     as a 40x collapse risk on hybrid Qwen models. No collapse occurs here on
#     ROCm; it is the fastest value tested. Do not "fix" it.
# ============================================================================

# Historic note (superseded by the block above):
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
LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

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
