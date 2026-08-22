# ===== FACTS (generated: do not hand-edit) =====
# MODEL    Qwen3.8-27B-Q8_0.gguf
# BUILD    llama.cpp-vulkan-tp   backend Vulkan0,Vulkan1
# SPLIT    tensor 50,50   (XTX,R9700)   main-gpu 1   ubatch 1024
# CONTEXT  216064   KV f16
# SPEC     draft-mtp  n_max 3
# VISION   yes (tower on Vulkan1)
# VRAM     XTX ceiling 24200 MiB - see MEASURED below for this profile's peak
# MEASURED see records/MASTER_RESULTS.csv (regenerate labels: src/benchmarks/preset_sweep.sh)
# ===== END FACTS =====
# ============================================================================
# VULKAN TENSOR-PARALLEL, MEASURED 2026-08-19 -- 2x PREFILL, -21% MTP DECODE
# ============================================================================
# NON-STANDARD BUILD. This profile does NOT use llama.cpp/build-vulkan. It needs
# llama.cpp-vulkan-tp (git worktree, branch pr25051), because stock Vulkan
# CANNOT tensor-split: -ts is silently ignored and every weight lands on device
# 0. Measured at ts 45/55 -- stock Vulkan put 23,865 MiB on the XTX and 57 MiB
# on the R9700, then died. With PR #25051: 14,417 / 14,778, matching ROCm.
# If llama.cpp-vulkan-tp is missing, rebuild it:
#   cd llama.cpp && git fetch origin pull/25051/head:pr25051
#   git worktree add ../llama.cpp-vulkan-tp pr25051
#   cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# WHEN TO PICK THIS OVER THE ROCm TENSOR PROFILE: prefill-dominated work. Long
# cold reads, big document ingests, anything where time-to-first-token rules.
# For interactive/agent use, ROCm still wins -- see the decode row.
#
# MEASURED, Q8_0, all with Vulkan's OWN tuned params (they differ from ROCm's):
#   pp8192 @ 0 ctx     946.74   vs ROCm 435.62   (+117%)
#   pp2048 @ d65536    561.74   vs ROCm 282.31   (+99%)
#   pp2048 @ d200704   324.10   (ROCm at only d65536 is 282.31)
#   MTP decode         40.29 t/s @ 54.5% acceptance
#                      vs ROCm 48.83 t/s @ 55.1%   -- ROCm +21%
#   Acceptance is a TIE, so the decode gap is raw backend speed, not drafting.
# The prefill ratio is flat at 1.99-2.02x across 0/32k/65k: a real backend
# property, not noise.
#
# WHY ub 1024 AND ts 50,50 (do not inherit ROCm's 512 / 45,55):
#   ubatch  256 -> 768.27   512 -> 877.50   1024 -> 950.13   2048 -> 948.29
#   split @ub1024, pp8192/tg256:
#     30/70 838.92/17.58   40/60 902.23/19.45   45/55 936.86/20.31
#     50/50 946.74/21.37 <- pp peak, SHIPPED
#     60/40 873.63/24.01   65/35 834.37/25.80   70/30 801.09/25.90 <- tg peak
#   Decode peaks toward the XTX (the R9700 sits behind a Gen4 x4 chipset link),
#   but those splits cannot hold context: 65/35 exceeds the ceiling by c=98304.
#   50/50 is the only point that is both fast and able to hold 200k.
#
# WHY c=216064 (re-measured 2026-08-22 with REAL work, 4k prompt + 256 decode,
# peak sampled continuously under load, not at the load plateau):
#   216064 -> 24,206 MiB   <- SHIPPED, 6 MiB over the 24,200 ceiling
#   221184 -> 24,471 MiB   <- 271 MiB over. Hit 24,424 during load alone.
# 216064 is deliberately at the ceiling, chosen for context over margin. The
# desktop baseline was 2,330 MiB at measurement time; a heavier desktop eats
# directly into this. If it OOMs, drop to 200704 (was 23,538 MiB) rather than
# changing the split.
#
# The split is the lever with real room in it, NOT shipped here because it costs
# decode: at c=216064, ts 45,55 peaks at 20,974 MiB (3,226 MiB headroom) but
# drops decode 48.07 -> 45.01 t/s. 50,50 is kept for speed.
#   Do NOT size this from a trivial `-p 64 -n 1` probe: that said 196,608 was
#   the max, and real work at 200k needs ~1 GB more than the probe implied.
#
# EXPERIMENTAL, NOT A DAILY DRIVER: PR #25051 is an unmerged DRAFT whose base is
# 2026-06-29, about six weeks behind the pinned 030ebb558, so it lacks later
# fixes. Re-measure when it lands in master. Quantized KV (-ctk q8_0) is
# documented as unsupported under -sm tensor but this build ACCEPTS it and runs;
# untested for quality, left at f16 here.
# ============================================================================

#!/usr/bin/env bash
# Dual-GPU Vulkan fallback: Qwen3.8 27B Q8_0, MTP, q8_0 KV,
# native 256k context, and BF16 vision. Measured at 786.6 pp / 40.3 tg.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700. --main-gpu does NOT cover the vision tower; clip.cpp only
# honours this env var and otherwise grabs the FIRST GPU, the constrained XTX.
# The sibling ROCm profile measured this as 1.11 GiB of permanent XTX headroom
# plus the whole ~0.9 GiB vision transient, at zero text pp/tg cost.
export MTMD_BACKEND_DEVICE=Vulkan1

# This build's binaries were linked in /tmp/llamacpp-vkpr and carry a baked-in
# RUNPATH pointing there; that directory is gone. The libs (libllama-server-impl.so,
# libggml-vulkan.so, ...) live next to the binary, so point the loader at them.
# Without this the server dies instantly with:
#   error while loading shared libraries: libllama-server-impl.so
# The stale RUNPATH also ends in an empty entry (= current directory), which is why
# it appears to "work" when launched from inside build/bin and fails everywhere else.
export LD_LIBRARY_PATH="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PROFILE_NAME="preset-38-27b-q8-mtp-vulkan-TENSOR-f16-200k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-27B-Q8_0.gguf"
LLAMA_BACKEND="vulkan"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=216064
LLAMA_UBATCH=1024
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

# f16 KV, 2026-08-16. Measured on Vulkan (6226-token prefill):
#     ts 28,37  825 pp  35.03 tg  acc 53.8%   XTX 22180 / R9700 29429  <- shipped
#     ts 26,39  827 pp  34.84 tg  acc 53.6%   XTX 20377 / R9700 31231
# Equal on speed; 28,37 leaves ~2.9 GiB on the R9700 instead of ~1.4 GiB. The
# R9700 is not unlimited either -- protecting the XTX by filling the other card
# is not a safe trade.
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
LLAMA_EXTRA_ARGS=(
    # Each cached prompt state carries --ctx-checkpoints copies of the
    # recurrent state (default 32). On these hybrid models that dominates
    # the entry size and pushes it past --cache-ram, at which point
    # server_prompt_cache::alloc() silently refuses to cache the context
    # at all and returning to it costs a full reprocess.
    --ctx-checkpoints 8
    --split-mode tensor
    # Parks compute buffers + projector on the roomier R9700; speed-neutral
    # but keeps ~2.5 GiB off the XTX, which also drives the desktop.
    --main-gpu 1
    --tensor-split 50,50
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
    # --reasoning-preserve REMOVED: the PR #25051 base (2026-06-29) predates
    # this flag and rejects it with "invalid argument".
)
