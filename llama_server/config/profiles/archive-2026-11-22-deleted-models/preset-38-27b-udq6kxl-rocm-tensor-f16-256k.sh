# ============================================================================
# UD-Q6_K_XL + ROCm 10 TENSOR SPLIT, f16 KV, full 262144 context -- 2026-08-29
# ============================================================================
# ROCm 10.0.0 (/opt/rocm/core-10.0) via llama.cpp-rocm10. Binaries carry
# RPATH=/opt/rocm/lib, so no LD_LIBRARY_PATH is needed.
#
# WHY THIS EXISTS: ROCm 10 decodes ~18% faster than Vulkan on this model, but
# prefills ~2x SLOWER. Measured 2026-08-29, -sm tensor 50/50, ub1024, r=3,
# devices pinned:
#           pp2048 d0   tg256 d0   pp2048 d65k   tg256 d65k
#   Vulkan+gfxq  933.63     25.38        584.99        23.14
#   ROCm 10      434.91     30.29        304.60        27.18
# Use this preset ONLY if your workload is decode-dominated. For agentic work
# with long prompts and tool results, the Vulkan preset is the better choice.
#
# WHY ts 49,51 (measured settled VRAM at c=262144, after real generation):
#   ratio    XTX settled    headroom
#   50,50       24,217      343 MiB to card top <-- SHIPPED (user accepted)
#   49,51       21,813      2,387 MiB           <-- fallback if desktop grows
#   48,52       21,597      2,603 MiB
#   46,54       21,297      2,903 MiB
# ROCm needs more XTX VRAM than Vulkan at the same ratio: the Vulkan preset
# ships 50,50 and fits. Do not copy that ratio here.
# The 49->50 step costs 2,404 MiB -- tensor-boundary rounding, same effect the
# Vulkan preset documents.
#
# CONTEXT IS THE SAME AT EVERY RATIO: 262144 is the model's architectural max
# (qwen35 context_length). The ratio buys DECODE, not context.
#
# DECODE COST OF DROPPING TO THE FALLBACK (llama-bench, r=5, ROCm 10):
#   depth      ts 50,50        ts 49,51     50,50 gain
#   d0        30.01+-0.14     29.37+-0.01      +2.17%
#   d65536    27.20+-0.18     25.97+-0.16      +4.76%
# The gain grows with depth, which is where this preset is actually used.
#
# 50,50 leaves only 343 MiB to the top of the card. Desktop baseline measured
# 3,379-3,525 MiB during this fit check (the Vulkan preset recorded 1,551-1,874),
# so this ratio WILL OOM if the desktop grows by ~350 MiB. FALLBACK: 49,51,
# which leaves 2,387 MiB.
#
# VISION: clip.cpp honours only MTMD_BACKEND_DEVICE and otherwise grabs the
# FIRST device. Set to ROCm1 (R9700) to keep ~1.1 GiB off the XTX.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# ROCm1 = R9700 (ROCm0 is the 7900 XTX).
export MTMD_BACKEND_DEVICE=ROCm1

PROFILE_NAME="preset-38-27b-UDQ6KXL-rocm10-TENSOR-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-rocm10/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-UD-Q6_K_XL.gguf"
LLAMA_BACKEND="rocm"

LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="mtp"
# *** MTP n-max IS SETTLED AT 3 -- DO NOT SWEEP IT AGAIN. *** User decision,
# 2026-09-03: "MTP 3 is the default for this model as shipped by Qwen. MTP 3 is
# final." Corrected 10-seed runs at temp 1.0 put n2/n3/n4/n5 within a few percent
# of each other on code, well inside their seed-to-seed spread -- there was never
# an n-max win to find. (An earlier "+19.4% at n-max 5" was an artifact of
# benchmarking at temp 0.7 and ctx 32768; both are wrong for this preset and both
# inflate speculative results. It was shipped, caught, and reverted.)
#
# WHAT MTP ACTUALLY BUYS (ctx 262144, Unsloth thinking sampler, 8 seeds,
# aggregate throughput). Measured 2026-09-03 -- see
# records/2026-09-03-27b-rocm-speculation/SPECULATION-BY-CONTENT.md
#
#   prompt      no-spec   MTP n3   accept   speedup
#   code  10k    29.70     50.46    56.3%    1.70x
#   code  40k    28.30     50.85    63.2%    1.80x
#   prose 10k    29.72     33.15    25.2%    1.12x
#   prose 40k    28.29     28.02    19.7%    0.99x   <- break-even
#
# Code and prose diverge WITH DEPTH: code acceptance rises (56->63%) so the win
# grows; prose acceptance falls (25->20%) until drafting exactly cancels itself.
# Without speculation tg is content-independent (29.70 vs 29.72), so all of that
# spread is draft acceptance. Keep MTP on: large win for code, free for prose.
# NOTE: an armed tool/structured grammar disables speculation outright.
#
LLAMA_SPEC_DRAFT_N_MAX=3
LLAMA_SPEC_DRAFT_TYPE_K="f16"
LLAMA_SPEC_DRAFT_TYPE_V="f16"
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
LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --split-mode tensor
    --main-gpu 1
    --tensor-split 50,50
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
)
