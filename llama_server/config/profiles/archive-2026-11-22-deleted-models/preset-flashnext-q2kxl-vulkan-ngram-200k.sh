# ============================================================================
# Preset 16 = preset 14 + `--spec-type ngram-mod`.
#
# Same binary, same ctx 200704, same ub192, same tensor placement as preset 14.
# The ONLY change is draft-free n-gram speculative decoding. No build, no root,
# no draft model, no prefill cost.
#
# WHY IT WORKS, AND WHY IT ONLY WORKS AT DEPTH
# Speculation amortizes the per-token costs that grow with n_kv -- the CPU-side
# PLE n-gram lookup, the QSA indexer, and the mask build -- across every accepted
# token. Those costs are small at 10K and dominant at depth, so the gain tracks
# depth:
#
#   MEASURED 2026-09-02, llama.cpp-qsagather, ctx 200704, ub192, ts 400,600,
#   temp 1.0 / top_k 20 / top_p 0.95, seed 42, 1 warmup + 3 measured, medians,
#   ignore_eos, n_predict 128:
#
#   depth              spec none        ngram-mod        decode
#   10,776 (code)      609.4 / 38.51    611.2 / 39.18    +1.7%
#   12,952 (prose)     605.6 / 37.98    605.6 / 37.68    -0.8%
#   61,426 (code)      492.1 / 29.86    492.1 / 35.61    +19.3%
#
#   Prefill is IDENTICAL at 61K (492.1 both) -- this is decode-only.
#
#   The 61K result is not noise: every ngram pass (32.00 / 35.61 / 36.13) beat
#   every control pass (29.73 / 29.86 / 29.96). Worst ngram > best control by 6.8%.
#
# *** VRAM LIMIT -- READ BEFORE USING PAST ~64K ***
# ngram-mod costs +439 MiB on the XTX (23,815 vs 23,376 at 61K depth).
# The preset-14 control measured 23,983 MiB at 199K depth. 23,983 + 439 projects
# to ~24,422 against a 24,200 guard ceiling and 24,560 MiB of PHYSICAL XTX --
# roughly 138 MiB of physical margin on the card that DRIVES THE DISPLAY.
#
#   VALIDATED to 61,426 tokens. NOT validated at 199K. That projection is an
#   extrapolation, not a measurement.
#
# To clear it for full depth, run the ngram arm at deep199k.txt and read card1 at
# depth. If it does not fit, either lower --tensor-split toward 390,610 (the
# R9700 had 1,936 MiB spare) or shrink the window with --spec-ngram-mod-n-max /
# --spec-ngram-mod-n-min, then re-measure.
#
# Speculation is a runtime flag: drop `--spec-type` to revert instantly.
#
# CAVEAT ON N-GRAM NUMBERS: acceptance inflates on repetitive input, which reads
# decode high. That is why the prose control above is reported next to the code
# one, and why variance rises with it (tg spread 4.13 vs 0.23 for the control).
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device).
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-qsagather-ngram-200k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qsagather/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=200704
LLAMA_UBATCH=192
LLAMA_CACHE_RAM=8192

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

LLAMA_SPEC_MODE="ngram-mod"
LLAMA_DRAFTER=""

# default.sh forces LLAMA_SPEC_DRAFT_N_MAX=2, but the +19.3% above was measured
# with bare `--spec-type ngram-mod`, i.e. llama.cpp's own default of 3. Pin it so
# the shipped config is the one that was actually benchmarked.
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_P_MIN=0.0

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
