# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-UD-Q6_K_XL.gguf
# BUILD    llama.cpp-tp-adaptive   backend Vulkan0,Vulkan1
# SPLIT    tensor 50,50   (XTX,R9700)   main-gpu 1   ubatch 1024
# CONTEXT  262144   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower on Vulkan1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# UD-Q6_K_XL + Vulkan TENSOR SPLIT, f16 KV, full 262144 context -- 2026-08-24
# ============================================================================
# MODEL: Unsloth Dynamic v3.0 UD-Q6_K_XL (23.56 GiB) rather than Q8_0 (27.05).
# It is NOT a uniform Q6: 310 tensors stay Q8_0, 168 Q6_K, 27 Q5_K, 1 Q4_K, so
# the body is higher precision than a plain Q6_K (which has only 49 Q8_0). Built
# with the v3.0 imatrix (1251 calibration chunks vs 45 in the Q8_0/plain-Q6_K
# files). Unsloth's own chart puts it ~0.2 points of top-1% accuracy below Q8_0
# while being 3.5 GiB smaller.
#   NOTE: the MTP head here is Q6_K (Q8_0 in the old files). Measured acceptance
#   was identical (0.831 both), so the lower-precision draft head costs nothing.
#
# WHY ts 50,50 (raw decode, llama-bench -p 0 -n 256, NO speculation -- draft depth
# is not what we are maximising and IRL acceptance is only 0.5-0.6):
#   ratio   raw tg @d65k    fit-check XTX @262144   PRODUCTION XTX
#   49,51      18.63              21,052                 ~20,800
#   50,50      19.62              23,886                  23,648   <-- shipped
#   51,49      19.84              24,552 (over)              -
#   52,48      19.99              23,984                  24,193   7 MiB margin
#   53,47      20.07              24,220 (over)              -
#   54,46      20.21              24,179                     -     21 MiB margin
# Raw decode rises monotonically with weight on the XTX (the faster card):
# +5.3% from 49/51 to 50/50, +7.3% at 52/48. VRAM is NON-monotonic but fully
# reproducible (identical to the MiB across two passes) -- 51,49 and 53,47 land
# badly on tensor-boundary rounding while 52,48 fits.
#
# 52,48 was measured at 23,984 in a fit check but landed at 24,193 in production
# -- 7 MiB under the ceiling. The difference is the DESKTOP BASELINE, which moved
# between 1,551 and 1,874 MiB during a single morning. Any ratio whose margin is
# smaller than that swing is not shippable no matter what a clean-room fit check
# says. 50,50 leaves 552 MiB, which covers it.
#   FALLBACK: if the desktop grows past ~500 MiB above idle, drop to 49,51
#   (~20,800, 3.4 GiB margin) for -5.3% decode. Do NOT use 51,49 or 53,47.
#
# WHY f16 draft KV: +2.0% at n3/n4 on the Q8_0 build (n=22, p=0.005). Costs
# ~120 MiB. Draft depth 3 chosen as the conventional default; measured n2..n5
# differences at realistic acceptance are within noise, so depth is not tuned here.
#
# VISION ON THE R9700: --main-gpu does NOT cover the vision tower. clip.cpp only
# honours MTMD_BACKEND_DEVICE and otherwise grabs the FIRST device (the XTX,
# which also drives the desktop). Omitting this costs ~1.1 GiB of XTX and was
# enough on its own to make every ratio appear not to fit.
#
# CEILING: 24,200 MiB on the XTX. Measure the SETTLED value after a real
# generation, not the load reading.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700 (Vulkan0 is the 7900 XTX).
export MTMD_BACKEND_DEVICE=Vulkan1

PROFILE_NAME="preset-38-27b-UDQ6KXL-vulkan-TENSOR-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-tp-adaptive/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-UD-Q6_K_XL.gguf"
LLAMA_BACKEND="vulkan"
# +3.3% tg d0 / +2.5% tg d65536 on this model, prefill flat (2026-08-29, r=3,
# devices pinned). Upstream reports -8% on DENSE models -- measured positive here.
export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

LLAMA_SPEC_MODE="mtp"
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
