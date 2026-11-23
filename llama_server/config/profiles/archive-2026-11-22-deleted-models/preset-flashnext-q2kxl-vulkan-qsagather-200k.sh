# ============================================================================
# Preset 14 = preset 13 + PR 27977 commit d7ca8ea "attend the selected cells"
# (QSA gather). Decode-only by construction: the gather fires when
# 4*n_tps*width < n_kv, i.e. n_tps==1 at decode (4*2051=8204 < 200704) and
# never during ub192 prefill (4*192*2051 = 1.57M > n_kv).
#
# MEASURED @199,340 actual tokens, temp 1.0 / top_k 20 / top_p 0.95, n=3 each:
#   preset 13    pp 348.74 mean  [350.76 347.96 347.50]   tg 18.395  [17.998 18.586 18.600]
#   THIS         pp 347.83 mean  [348.32 347.62 347.55]   tg 18.591  [18.443 18.660 18.671]
#   -> tg +1.07% (higher in ALL 3 paired runs); pp -0.26%, inside noise.
#
# Honest scale: this is a ~1% decode gain, not a step change. Preset 13 remains
# the reference; use this if you want the extra 1%.
# Build: llama.cpp-qsagather (fusionscan state + d7ca8ea).
# ============================================================================
# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, populated 200K
# SCAN-FIX build. Validated 2026-08-31 at 199,340 actual prompt tokens.
# ============================================================================
# AUTHORITATIVE FULL-DEPTH RESULT (same prompt, ignore_eos, 128 generated
# tokens -- not an EOS-truncated sample):
#
#                        pp        tg        card0 VRAM   migration
#   preset 12 (ub128)  346.75    12.854      30,311       none
#   THIS (ub192)       348.09    17.910      30,564       none
#
#   decode +39.3% at real depth, prompt speed unchanged.
#
# WHY IT IS FASTER: the previous build (base aa1b06d0c) predated the merged
# upstream fix 62acc89c2 (PR 28011). llama_kv_cells::for_each_token_in walked
# all 256 LLAMA_MAX_SEQ slots for every used cell. That path is reached from
# get_prev_tokens -> build_ple, the per-layer-embedding n-gram lookup, which is
# part of the MODEL ARCHITECTURE and runs on every decode step even with
# --spec-type none. At 199K depth with one sequence that was roughly 51 million
# wasted CPU iterations per generated token. The gain scales with used cells:
# +7.4% at 11.5K, +39.3% at 199K.
#
# Also applied: 28e37926 (scan only the n-gram window) and b88668b (used-cell
# bitmap), both from PR 27977.
#
# WHY UBATCH 192, NOT 256 OR 384 -- measured at full depth, not shallow:
#   ub192  348.09 pp / 17.910 tg  card0 VRAM pinned 30,563 the whole prefill
#   ub256  349.44 pp / 17.687 tg  237 MiB migrated off card0 from 161,792 on
#   ub384  guard-killed: xtx 24,552 > 24,200 ceiling, GTT 1,346 -> 2,069 MiB
# ubatch is nearly irrelevant to prompt speed at depth (2.7 t/s across 128-256);
# attention over 200K KV dominates. Large-ubatch pp gains seen at 10K depth are
# an artifact and do NOT survive to 199K.
#
# NOT WORTH APPLYING (measured on this hardware):
#   PR 28023 indexer head slices -- noise on Vulkan/RADV (598.62 vs 597.69 pp)
#   PR 27977 QSA gather d7ca8ea  -- decode-only by construction, cannot fire at
#                                   ub128-384 prefill (4*n_tps*width > n_kv)
#
# BUILD: llama.cpp-qsagather. This is a COPY of the 7 dirty files
# from llama.cpp-qwen4exp-vulkan-fusions (Vulkan mat-vec epilog fusion, fused
# REPEAT+MUL+ADD, skinny inject prefill path, radix TOP_K) plus the three scan
# commits. The original fusions worktree was NOT modified. Those Vulkan fusions
# are worth roughly 2x prompt speed at depth and must be preserved.
#   CCACHE_DIR=/tmp/flashnext-ccache cmake --build \
#     llama.cpp-qsagather/build-vulkan --target llama-server -j 32
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device).
# NOTE: --tensor-read-lazy still exists on this base; it was REMOVED upstream.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-scanfix-qsagather-200k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qsagather/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=200704
LLAMA_UBATCH=192
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
