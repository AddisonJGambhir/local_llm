# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q8_0.gguf
# BUILD    llama.cpp-tp-adaptive   backend Vulkan0,Vulkan1
# SPLIT    tensor 49,51   (XTX,R9700)   main-gpu 1   ubatch 1024
# CONTEXT  262144   KV f16
# SPEC     draft-mtp-adaptive  n_max 12  n_min_adaptive 3
# VISION   yes (tower on Vulkan1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# MERGED BUILD: Vulkan TENSOR SPLIT + ADAPTIVE MTP, 256k f16 -- 2026-08-22
# ============================================================================
# NON-STANDARD BUILD. Uses llama.cpp-tp-adaptive, a worktree carrying
# PR #25051 (Vulkan tensor allreduce) with PR #27210 (adaptive MTP) ported on
# top. Neither upstream build can do both: stock #25051 has no adaptive MTP,
# and #25051 rebased onto master aborts under -sm tensor with
#   ggml-backend-meta.cpp:1156: multi buffers are not supported by the meta backend
# Rebuild recipe + patches: records/tp-trial-rebuild-recipe.md and
# records/2026-08-22-adaptive-mtp/RESULTS.md (which lists all six conflict
# resolutions). RUNPATH is absolute, so no LD_LIBRARY_PATH export is needed
# here -- unlike the older llama.cpp-vulkan-tp profiles.
#
# WHY ADAPTIVE MTP (measured, agentic prompt, 2048-token generations, c=98304):
#   tensor 50/50 draft-mtp n3          56.99 tg   622.0 pp
#   tensor 50/50 draft-mtp-adaptive    81.67 tg   622.7 pp   <- +43.3% tg, pp free
# Confirmed independently on layer split at +38.3% (n=7 each, exact permutation
# p=0.00058, ranges disjoint). Adaptive costs nothing on prefill: it only varies
# draft depth during decode, and prefill does not speculate.
#
# WHY ts 49,51 AND c=262144 (measured with REAL work, ~4.9k prompt + 512 decode,
# peak sampled continuously under load -- NOT off the load plateau):
#   49,51 -> 23,521 MiB peak, 679 MiB under the 24,200 ceiling, 50.86 tg / 809.0 pp
#   45,55 -> 22,820 MiB peak, 1,380 MiB under,                  49.14 tg / 809.8 pp
# 49,51 is both faster and comfortably inside the ceiling; 45,55 is the fallback
# if the desktop grows. Adaptive's overhead does NOT scale with context (+373 MiB
# over the same-ratio n3 build at 256k, vs +736 MiB at c=98304), so it is fixed
# draft state rather than anything per-token.
#
# VISION ON THE R9700. --main-gpu does NOT cover the vision tower; clip.cpp only
# honours MTMD_BACKEND_DEVICE and otherwise grabs the FIRST device, the
# constrained XTX. Parking the 889 MiB tower on Vulkan1 keeps it off the card
# that also drives the desktop, and the R9700 has ~5.5 GiB free at this context.
#
# EXPERIMENTAL: both PRs are unmerged. Re-measure when either lands upstream.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700 (verified: Vulkan0 is the 7900 XTX, Vulkan1 the AI PRO R9700).
export MTMD_BACKEND_DEVICE=Vulkan1

PROFILE_NAME="preset-38-27b-q8-vulkan-TENSOR-ADAPTIVE-f16-256k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-tp-adaptive/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=1024
LLAMA_CACHE_RAM=24576

LLAMA_CACHE_TYPE_K="f16"
LLAMA_CACHE_TYPE_V="f16"

# Adaptive MTP: depth starts at n_min_adaptive and climbs toward n_max as the
# controller observes acceptance. n_max 12 measured safe under -sm tensor on
# this build. Note the older finding that n_max > 8 crashed the meta backend
# applies to the STOCK #25051 draft-mtp path, not this adaptive controller.
LLAMA_SPEC_MODE="mtp-adaptive"
LLAMA_SPEC_DRAFT_N_MAX=12
LLAMA_SPEC_DRAFT_N_MIN_ADAPTIVE=3
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
LLAMA_EXTRA_ARGS=(
    --ctx-checkpoints 8
    --split-mode tensor
    --main-gpu 1
    --tensor-split 49,51
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
)
