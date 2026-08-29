# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
# BUILD    llama.cpp   backend ROCm0,ROCm1
# SPLIT    layer 34,31   (XTX,R9700)   main-gpu 1   ubatch 256
# CONTEXT  245760   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   no
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# HauhauCS "Aggressive" Q6_K_P -- TUNED AND REJECTED AS DEFAULT, 2026-08-18
# ============================================================================
# Provenance: SHA256 verified against the publisher's manifest
#   70c8139f4dcd05d3818bc9843cde7301faa44f66321e5966965929baa9c24a6a  -- MATCH.
# The release manifest's Ed25519 signature also verified. The file is 24.14 GiB
# (25,924,151,424 bytes), which lands within 960 BYTES of unsloth's UD-Q6_K_XL
# and within 896 bytes of UD-Q8_K_XL -- i.e. "K_P" is the UD recipe relabeled,
# a metadata-level difference, not a new quantization.
#
# THE VENDOR CLAIM ("HauhauCS FastMTP, up to 3.02x TG") DOES NOT REPRODUCE.
#
# VRAM: 3.04 GB bigger than plain Q6_K, so it CANNOT use the XTX-heavy splits
# that plain Q6_K wins with. Measured at c=245760 (probe aborts at 23,800):
#     30/35 -> 20,277    32/33 -> 21,985    34/31 -> 22,616  <- SHIPPED
#     36/29 -> 23,919 OVER      38/27 -> 24,226 OVER (ceiling is 24,200)
#
# SPEED, llama-bench d131072 ub256 r1 (UNSPECULATED):
#     ts 1/1    pp 319.56  tg 16.44        ts 30/35  pp 293.15  tg 16.16
#     ts 32/33  pp 318.74  tg 16.50        ts 34/31  pp 323.44  tg 16.68  <- best
# vs plain Q6_K at its best split (ts 38/27): pp 353.30 / tg 18.13.
#     => K_P is 8.7% SLOWER decode and 9.2% SLOWER prefill.
#
# REAL MTP DECODE, live server, 4 natural prompts x 400 tokens, 2 passes, idle
# machine (this is the only axis where "FastMTP" could have won):
#     K_P  (34/31): 33.53 / 35.48  -> mean 34.51 t/s, acceptance 55.8%
#     Q6_K (38/27): 34.17 / 36.25  -> mean 35.21 t/s, acceptance 57.6%
#     => a TIE inside pass-to-pass noise; no acceptance advantage whatsoever.
#
# CAUTION ON METHOD: an earlier run appeared to show K_P 12% FASTER. That was a
# contaminated baseline -- the plain-Q6_K numbers were taken while a 24 GiB
# download saturated the machine (its tell: a 24.18 t/s sample at 66.2%
# acceptance, i.e. high acceptance but low throughput = contention, not model
# behavior). Always A/B on an idle box.
#
# SHALLOW (d0, ts 34/31, ub 256, r=2): pp2048 1172.82+-2.67, pp8192 1207.90+-1.32,
# tg32 21.11+-0.01. Prefill at shallow depth ties plain Q6_K exactly (1207.99);
# the 9% prefill gap only opens up at depth. Shallow decode is 9.4% behind
# (21.11 vs 23.31), matching the depth measurement.
#
# VERDICT: keep plain Q6_K as the default. K_P costs 2.83 GiB more disk, loses
# the XTX-heavy splits, has no mmproj in its repo (vision args stripped below),
# and buys nothing. This profile exists so the result is reproducible.
# ============================================================================

#!/usr/bin/env bash
# Qwen3.8 27B "Q6_K_P" (HauhauCS, uncensored "Aggressive" finetune) with built-in
# MTP, f16 KV, native 256k context, and the BF16 vision projector. ROCm, dual GPU.
# NOT the unsloth Q6_K -- that is preset-38-27b-q6k-rocm-dual-f16-256k-mmproj.sh.
#
# Added 2026-08-17. This is a QUANT SWAP of the shipped Q8_0 preset
# (preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh) and nothing else. Every tuned
# value below was measured on the Q8_0 build; see "WHAT IS AND IS NOT MEASURED".
#
# Source: HauhauCS, file Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf,
# 25,924,151,424 bytes = 24.14 GiB. Verified from the file's own GGUF metadata
# 2026-08-20: general.author / general.organization / general.quantized_by all
# read "HauhauCS", base model Qwen/Qwen3.8-27B, Apache-2.0.
#
# CORRECTED 2026-08-20 -- this block previously read "unsloth/Qwen3.8-27B-GGUF
# ... (21.31 GiB)" and compared itself to UD-Q6_K_XL. All of that was copied
# from the unsloth Q6_K profile and none of it describes this file. It is NOT an
# unsloth quant and it is NOT 21.31 GiB.
#
# "Q6_K_P" is HauhauCS's own label, not a llama.cpp ftype. The GGUF ftype is 18
# (plain Q6_K), but the tensor mix is what makes it 2.83 GiB larger than
# unsloth's Q6_K (21.31 GiB):
#                  HauhauCS Q6_K_P    unsloth Q6_K
#     Q8_0                     279              49
#     Q6_K                     131             361
#     Q5_K                      96               0
#     F32                      360             456
# It keeps 5.7x as many tensors at Q8_0. Also note there are NO imatrix keys in
# this file, where the unsloth quant carries quantize.imatrix.file and
# .imatrix.dataset -- this one was not imatrix-calibrated.
#
# Hardware/device order on this host:
#   ROCm0 = RX 7900 XTX (gfx1100, 24 GiB, ALSO DRIVES THE DESKTOP)
#   ROCm1 = AI PRO R9700 (gfx1201, 32 GiB)
#
# ============================================================================
# WHY Q6_K: 5.74 GiB OF WEIGHTS, WHICH IS ~2.9 GiB OFF THE XTX
# ============================================================================
# Q8_0 27.05 GiB -> Q6_K 21.31 GiB. At --tensor-split 1,1 roughly half of that
# saving lands on each card, so the XTX should drop ~2.9 GiB.
#
# SUPERSEDED BY MEASUREMENT 2026-08-18 -- the projection that used to live here
# guessed ~21,000 MiB at load. That was for ts 1,1, which is no longer what this
# profile ships. Measured values are in the RETUNE block below; the headroom the
# quant bought was spent on an XTX-weighted split rather than banked as margin:
#     ts 1/1   @ 262144 -> 22,260 MiB   (full context, but pp 318)
#     ts 36/29 @ 245760 -> 23,775 MiB   <- SHIPPED (pp 354)
#     ts 36/29 @ 229376 -> 22,830 MiB   (safer fallback, same speed)
#
# SPEED EXPECTATION, ALSO UNMEASURED: decode is bandwidth-bound, and Q6_K is
# 78.8% of the bytes, so tg could gain ~20% if it scales with weight bytes.
# Prefill is compute-bound and Q6_K dequant costs more per byte than Q8_0, so pp
# may be flat or slightly down. Do not quote either number until llama-bench has
# actually run on this file. Quality: Q6_K is near-lossless vs Q8_0 on this size
# of dense model, but it is not identical -- if you are chasing an accuracy
# regression, the quant is now a variable.
#
# ============================================================================
# RETUNED AND MEASURED 2026-08-18 -- ts 1,1 -> 36,29 and c 262144 -> 245760
# ============================================================================
# The inherited values below were re-measured on THIS quant. Two changed.
#
# SPLIT SWEEP (llama-bench -d 131072 -p 2048 -n 32, ub 256, f16 KV, r=1):
#     ts 1/1     pp 318.44   tg 17.41      <- was inherited from the Q8_0 preset
#     ts 34/31   pp 323.79   tg 17.96
#     ts 36/29   pp 354.73   tg 17.56      <- SHIPPED
#     ts 38/27   pp 353.30   tg 18.13
#     ts 40/25   pp 331.61   tg 18.00
#
# WHY THIS DIRECTION WAS NEVER TESTED BEFORE: on Q8_0 the XTX-weighted splits
# did not fit (ts 40,25 OOMed at 24068 MiB). Q6_K frees 5.74 GiB of weights,
# ~2.9 GiB of it on the XTX, which unlocks the direction the Q8_0 data was
# already pointing at: loading the R9700 harder was monotonically worse
# (ts 20,45 = 243.8, ts 28,37 = 282.6, ts 1,1 = 315.8).
#
# MECHANISM, corroborated by the user's own observation: the R9700 sits pinned
# at its 300 W cap during prefill. Prefill is compute-bound, the card hits the
# power wall and downclocks, and in layer-split mode the two GPUs run
# SEQUENTIALLY per token -- so its slowdown adds directly to every token.
# Moving layers to the XTX routes work around the constrained card.
#
# UBATCH SWEEP re-run at ts 36/29 (same harness):
#     ub 128   pp 231.87   tg 17.38        (34% worse -- do not "optimize" here)
#     ub 256   pp 353.77   tg 17.49        <- SHIPPED
#     ub 384   pp 338.12   tg 17.64
#     ub 512   pp 331.06   tg 17.74
# tg drifts up with larger ubatch but only +2% across the whole range, which is
# not worth 7% of prefill. NOTE ub 256 measured 353.77 here and 354.73 in the
# split sweep -- independent runs 0.3% apart, so the pp noise floor is tight and
# the rankings above are real rather than lucky single runs.
#
# CONTEXT LADDER at ts 36/29 + ub 256 + f16 (measured to plateau, text-only):
#     213000 -> 22,224     229000 -> 22,830
#     245760 -> 23,775     253952 -> DOES NOT FIT (still climbing past 23,900)
# c=262144 does NOT fit at this split. That is the cost of the +11.5%.
#
# DECODE-WEIGHTED RETUNE, 2026-08-18. The user's criterion changed from prefill
# to "high decode preferred", so the split was re-picked on tg rather than pp.
# Depth sweep at d131072 + ub 256 (llama-bench, r=1, UNSPECULATED tg):
#     ts 1/1     pp 318.44   tg 17.41
#     ts 34/31   pp 323.79   tg 17.96
#     ts 36/29   pp 354.73   tg 17.56      (previously shipped, prefill pick)
#     ts 38/27   pp 353.30   tg 18.13      <- SHIPPED, best decode
#     ts 40/25   pp 331.61   tg 18.00
# 38/27 buys +3.2% decode for -0.4% prefill, i.e. prefill is free at this split.
#
# CONTEXT LADDER RE-RUN AT ts 38/27 (measured to plateau, text-only, idle XTX
# baseline 1,136 MiB). This corrected an earlier reading: 38/27 does NOT cost
# context relative to 36/29, it uses LESS XTX VRAM at equal context.
#     229376 -> 22,413     237568 -> 22,717
#     245760 -> 23,021     262144 -> 23,631 (fits, but only 669 MiB of margin)
#
# *** SHIPPED AT 245760: 23,021 MiB AT LOAD, ~1,179 MiB UNDER THE 24,200 CEILING ***
# That is 754 MiB MORE headroom than the 36/29 config it replaces, at the same
# context and better decode. 262144 was measured and rejected: it fits, but 669
# MiB is not enough slack on a card that also drives the desktop, whose footprint
# has moved 1.7 -> 3.0 GiB inside a single session.
# Expect roughly +223 MiB under load (compute buffers + MTP draft context, the
# delta measured on the Q8_0 preset), so a working peak near 23,250.
#
# vs the shipped Q8_0 preset (ts 1,1 ub 256 c 262144: pp 315.79 / tg 15.26):
#     +12% prefill, +15% decode, for 16k less context.
#
# NOT MEASURED HERE: MTP acceptance at this split (llama-bench cannot run
# speculative decode, so every tg above is UNSPECULATED), vision/mmproj VRAM at
# ts 36/29, and whether the split interacts with draft acceptance. Watch the XTX
# on first image use.
#
# UNRELATED BUG SEEN WHILE BENCHMARKING (does not affect the server): llama-bench
# crashes in llama_context::state_seq_set_data -> ggml_backend_cuda_buffer_set_tensor
# with an SDMA1 page fault when it restores a cached depth state on this hybrid
# model. Reproduced 4/4 on single-config runs at BOTH 300 W and 330 W, so it is
# not power-related. CORRECTED 2026-08-18: multi-config sweeps do NOT reliably
# survive it either -- a 5-config sweep with -r 2 crashed on benchmark 1/10 at
# depth run 2/2. The trigger is the state restore itself, which -r >= 2 forces
# between reps. Use -r 1 at depth, and treat any -d run as crash-prone. llama-server does not call
# that path. Power capping was tested and rejected: R9700 at 330 W (its firmware
# max) measured pp 358.95 vs 354.52 at 300 W = +1.2%, i.e. noise.
# ============================================================================

# ============================================================================
# WHAT IS AND IS NOT MEASURED IN THIS FILE
# ============================================================================
# INHERITED FROM THE Q8_0 PRESET, measured there, NOT re-measured here:
#   --tensor-split 34,31   DECODE-WEIGHTED retune 2026-08-18 -- see the retune block
#                          above. Re-measured on this quant, not assumed.
#   --ubatch-size 256      Re-measured on this quant at ts 36/29. Still optimal.
#   --batch-size 6144      +12.3% pp in text-only sessions on Q8_0. Collapses to
#                          ~653 t/s after any image until restart (see the
#                          post-image regression note at the bottom).
#   --main-gpu 1           Speed-neutral; parks compute buffers + mmproj on the
#                          roomier R9700.
#   n-max 3                Optimal at both depths on Q8_0. 2/4/5 lose.
#   --ctx-checkpoints 8    Load-bearing prompt-cache fix, see below.
#
# NOT MEASURED AT ALL ON Q6_K: pp, tg, MTP acceptance, VRAM, the context ladder,
# and whether the split is still optimal now that weights are smaller. A cheaper
# model shifts the compute/bandwidth balance, so ts 1,1 could conceivably move.
# It has NOT been checked.
#
# Pinned to llama.cpp build 10358 (030ebb558). Upstream 10450 (ece963f41) was
# built and measured on Q8_0: -4.8% pp, -4.0% tg, MTP acceptance 61.9% -> 55.7%,
# most likely from "ggml-hip: remove -funsafe-math-optimizations" (#26696).
# Re-benchmark before taking any llama.cpp update.
# ============================================================================

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

# Put the vision tower on the R9700. --main-gpu does NOT cover it: clip.cpp only
# honours the MTMD_BACKEND_DEVICE env var, and with it unset it falls back to
# ggml_backend_init_by_type(GPU), which picks the FIRST GPU = ROCm0 = the 24 GiB
# XTX. Costs ~17% on image prefill, buys ~1.1 GiB of permanent headroom plus the
# whole ~0.9 GiB vision transient on the card that drives the display.
# Less critical at Q6_K than at Q8_0 now that the XTX has room, but the image
# prefill cost is the only downside and it is already paid for in the Q8_0
# measurements, so it stays until someone measures the alternative here.
export MTMD_BACKEND_DEVICE=ROCm1

PROFILE_NAME="preset-38-27b-q6kp-hauhau-mtp-rocm-dual-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-rocm10/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=245760
LLAMA_UBATCH=256

# Prompt-cache sizing. A cached context state = full KV + (--ctx-checkpoints)
# copies of the recurrent state. This model family is hybrid, so those
# checkpoints are large (690 MiB each on Qwen3.8-27B, measured from the eviction
# log) and the default of 32 makes every entry enormous.
# server_prompt_cache::alloc() SILENTLY refuses any entry larger than the whole
# budget (returns nullptr, warns only at trace level), so an over-sized context
# is never cached at all -- and returning to it costs a full reprocess.
# --ctx-checkpoints 8 is the load-bearing half of the fix; raising --cache-ram
# alone cannot help if a single entry still exceeds the limit.
# Confirmed working in real use: the repeated full-context reprocessing that
# prompted this (119,389 tokens / 96 s, ~8 times in one session) has not
# recurred since.
LLAMA_CACHE_RAM=24576

# f16 KV throughout. On Q8_0 this was worth +11% decode and +7.4 points of MTP
# acceptance over q8_0 KV at a fixed split. KV size is independent of the weight
# quant, so 262144 at f16 costs the same ~17.4 GiB here as it does on Q8_0.
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
LLAMA_CHAT_TEMPLATE_KWARGS='{"preserve_thinking":true,"reasoning_effort":"xhigh"}'

LLAMA_STARTUP_TIMEOUT=420

# NOT optional. common.sh defaults LLAMA_TIMEOUT to 0, and llama-server reads
# --timeout as a socket read/write timeout in seconds (upstream default 3600),
# not as "disabled". At 0, any request body too large to arrive in a single
# socket read is aborted with an EMPTY 400 (Content-Length: 0, nothing logged
# server-side). Measured 2026-08-15:
#   --timeout 0    4K JPEG 2.5 MB: 3/8 succeeded   4K PNG 5.2 MB: 0/4
#   --timeout 3600 4K JPEG 2.5 MB: 8/8 succeeded   4K PNG 5.2 MB: 4/4
# Genuine errors still return a JSON body; an empty 400 is this bug's signature.
LLAMA_TIMEOUT=3600
LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --device ROCm0,ROCm1
    --split-mode layer
    --tensor-split 34,31
    --main-gpu 1
    --batch-size 6144
    --no-mmap
    --reasoning-preserve
)

# KNOWN LIMITATION — post-image prefill regression (llama.cpp 10358, this host).
# A SINGLE image request permanently drops text prefill ~40% until the server is
# restarted. Decode is unaffected. Measured on Q8_0, 2026-08-15:
#     text-only, 10 consecutive prefills : 1083-1090 t/s, no drift
#     after ONE 640x480 image            :  652, 652, 652, 650, 638 t/s
#     after ONE 3840x2160 image          :  650 t/s  (same magnitude)
# Trigger is any image at all, not image size. Ruled out by measurement:
# batching, thermals/clocks, mmproj device, prompt cache, GTT spill.
# Assumed to apply here too (it is an engine bug, not a quant effect), but not
# re-verified on Q6_K.
# Workaround until upstream fixes it: restart the server after heavy image use.
