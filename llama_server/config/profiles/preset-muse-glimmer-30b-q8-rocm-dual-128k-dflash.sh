#!/usr/bin/env bash
# Muse Glimmer 30B Q8_0 — ROCm dual-GPU, DFlash speculation, f16 KV, 128k native.
#
#   ROCm0 = RX 7900 XTX (gfx1100, 24 GiB)  <- ALSO DRIVES THE DESKTOP, keep < ~24200 MiB
#   ROCm1 = AI PRO R9700 (gfx1201, 32 GiB)
#
# Architecture: 52 layers, sliding-window attention (window 2048, pattern 4), so
# only ~13 layers keep context-scaling KV. Full 128k costs 436 MiB at q8_0 /
# ~872 MiB at f16 — KV is nearly free here, which is why f16 is affordable.
# No MTP head; speculation comes from the DFlash sidecar (dflash-kquant.gguf,
# arch `dflash`, block_size 16, target_layers [2,14,26,38,50]).
#
# Measured 2026-08-16 (5711-token prefill, 384-token decode):
#   no speculation      1107 pp   19.05 tg           <- DFlash is worth +47% decode
#   DFlash n-max 2       877 pp   28.01 tg  acc 43.6%
#   + f16 KV             873 pp   33.23 tg  acc 58.5%   <- shipped combination
#   + batch 6144         986 pp                          <- shipped
#
# Load-bearing values:
#   n-max 2            Swept 1/2/4/8/15. Acceptance collapses with depth:
#                      60.5 / 43.6 / 21.7 / 11.9 / 6.5%. UPSTREAM RECOMMENDS 15
#                      (the DFlash block size is 16) AND SO DOES THIS REPO'S OWN
#                      ARCHIVED NOTE -- both are wrong here. That note was for the
#                      Q5_K_M on a SINGLE GPU; spanning two cards makes the verify
#                      pass expensive enough to favour shallow drafts. Vulkan
#                      agrees: n2=30.1, n4=28.8, n8=16.7.
#   f16 KV             +19% decode and +15 points of DFlash acceptance over q8_0.
#                      KV precision drives how often drafter and target agree.
#                      Nearly free here because SWA keeps the cache tiny.
#   --batch-size 6144  875 -> 986 pp (+13%). 12288 is identical (988).
#   --ubatch-size 512  256=840, 512=875, 768/1024=814, 2048=705. The 6656 hidden
#                      dim makes compute buffers grow fast, so 512 wins — unlike
#                      the Qwen3.6 MoE (2048 dim) which wants 1024.
#   --tensor-split 1,1 Do NOT push layers onto the XTX despite its bandwidth
#                      advantage: 1,1=876, 30,22=816, 34,18=762 pp.
#   LLAMA_DFLASH_SPEC_TYPE=draft-dflash
#                      common.sh defaulted to `dflash`, which THIS BUILD REJECTS
#                      ("unknown speculative type"). Without this the profile does
#                      not start at all.
#
# NOTE: DFlash decode is nondeterministic even at a fixed seed (spreads of
# 5-50%, versus 0.4% with speculation off). Treat any single decode number here
# as +/-10%; the n-max ranking survives only because the acceptance trend is
# monotonic across five settings.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-muse-glimmer-30b-q8-rocm-dual-128k-dflash"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Muse-Glimmer-30B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=131072
LLAMA_UBATCH=512
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="dflash"
LLAMA_DRAFTER="$LLAMA_SERVER_ROOT/models/dflash-kquant.gguf"
LLAMA_DFLASH_SPEC_TYPE="draft-dflash"
LLAMA_SPEC_DRAFT_N_MAX=2

LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=20
LLAMA_MIN_P=0.0
LLAMA_PRESENCE_PENALTY=0.0
LLAMA_REPEAT_PENALTY=1.0
LLAMA_REASONING="auto"

LLAMA_STARTUP_TIMEOUT=300
# common.sh defaults this to 0, which llama-server reads as a ZERO-second socket
# timeout: large request bodies die with an empty 400 and nothing is logged.
LLAMA_TIMEOUT=3600

LLAMA_EXTRA_ARGS=(
    --device ROCm0,ROCm1
    --split-mode layer
    --tensor-split 1,1
    --main-gpu 1
    --batch-size 6144
    --ctx-checkpoints 8
    --cache-reuse 256
    --no-mmap
)
