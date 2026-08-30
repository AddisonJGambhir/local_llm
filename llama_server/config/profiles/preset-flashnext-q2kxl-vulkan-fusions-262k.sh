# ============================================================================
# Qwen3.8-Flash-Next UD-Q2_K_XL -- Vulkan LAYER split, 262144 ctx -- 2026-08-30
# MAX CONTEXT preset. 262,144 is the model's trained maximum
# (qwen4exp.context_length); there is nothing above it without RoPE extension.
# ============================================================================
# Measured (5-pass median shallow; full-depth probe at 199,379 tokens):
#   shallow tg 50.0285  (min 48.1162, max 50.0814) -- 5-pass median
#   deep pp 585.63, deep tg 41.0227 at 8,948 tokens
#   R9700 31,069 MiB (1,131 spare under 32,200)
#   XTX   23,772 MiB (428 spare under 24,200)
#   NO CPU EXPERT OFFLOAD. Every expert is in VRAM.
#   FULL-DEPTH VALIDATED: a 199,379-token prefill ran clean, zero guard
#   strikes, peak R9700 31,075 / XTX 22,910. pp 319.87 and tg 12.5397 at
#   that depth. This is the only 262,144 profile proven at full depth.
#
# BEATS the 256K variant documented in preset-...-layer-128k.sh at the SAME
# context, inside a STRICTER R9700 guard:
#   shipped 256K variant: pp 440, tg 42.3, tg@deep 29.4, R9700 32,525
#   this preset:          pp 542, tg 48.4, tg@deep 38.8, R9700 31,751
#   -> tg +14.4%, pp +23.2%, deep tg +32.1%, and 774 MiB LESS on the R9700.
#
# BUILD IS NOT THE SHIPPED ONE. This preset requires the patched experimental
# worktree llama.cpp-qwen4exp-vulkan-fusions (detached at aa1b06d0c, dirty:
# Vulkan mat-vec epilog fusion, fused REPEAT+MUL+ADD kernel, skinny inject
# mat-vec prefill path, Vulkan-side radix TOP_K). Those changes are NOT
# committed inside that clone -- its AGENTS.md forbids commits without
# explicit approval. Do not reset, clean or checkout that worktree.
# Rebuild with:
#   CCACHE_DIR=/tmp/flashnext-ccache cmake --build \
#     llama.cpp-qwen4exp-vulkan-fusions/build-vulkan --target llama-server -j 16
#
# FASTER VARIANT, SHALLOW PROMPTS ONLY (-ub 256, same -ot):
#   shallow tg 48.3925 (spread 0.3459), deep pp 542.14, deep tg 38.8292 at
#   8,948 tokens; R9700 31,751 at rest. DO NOT use it for near-full-context
#   prompts: a 199,382-token prefill drove the R9700 to 32,227 MiB, over the
#   32,200 guard. It completed, but the margin is 27 MiB.
#
# PREFILL-FIRST VARIANT (8,948-token prefills only -- see warning):
#   LLAMA_UBATCH=2048
#   -ot '...,blk\.(9|14|28)\.ffn_(gate|up|down)_exps\.weight=CPU'
#   deep pp 807.4087, deep tg 37.7755, shallow tg 40.9395
#   R9700 31,915 (285 spare), XTX 23,641 (559 spare) AT REST
#   WARNING: a ~200,000-token prefill drove the R9700 to 32,359 MiB and the
#   guard killed the server. Validated for prefills up to 8,948 tokens only.
#
# 196,608 VARIANT (best all-round, ZERO expert offload):
#   LLAMA_CONTEXT=196608 LLAMA_UBATCH=2048
#   -ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'
#   shallow tg 47.9324 (3-pass median), deep pp 976.6169, deep tg 40.8701
#   R9700 31,627 (573 spare), XTX 23,535 (665 spare)
#
# WHY NO CPU EXPERT OFFLOAD: an earlier revision of this preset put blk.11's
# experts on CPU and measured 48.3925. Moving that layer back onto the GPUs --
# possible once token_embd.weight also goes to CPU, which frees ~341 MiB and
# costs nothing -- measures 50.0285. The CPU expert layer was costing ~1.6 t/s
# and buying headroom that token_embd already provides. If you ever DO need to
# shed a layer: send it to CPU, never to the other card (cross-card expert
# moves measured 34.75 to 38.80 t/s against 49.48 unsplit), and pick a middle
# layer -- blk.11 48.3925, blk.13 47.7699, blk.9 46.2938, blk.0 42.30.

# BALANCE TABLE (same split, same -ot, all-GPU, q8_0 KV, all verified clean):
#   ctx      ub    tg      pp     deep tg   R9700 spare   XTX spare
#   262,144  128   50.03   586    41.02     1,131         428
#   229,376  128   50.13   585    41.00     1,539         710
#   163,840  512   49.86   767    38.81     1,280         542
#   131,072  512   48.22   816    39.42     1,740         738
# ub 512 is the practical prefill ceiling. ub 1024 and 2048 breach at every
# context tried (131,072 with ub 2048 reached R9700 32,601 against the 32,200
# guard). Pick 163,840/ub512 for balance, 262,144/ub128 for maximum context.
#
# WHY -ub 128 AND NOT 256: both fit at load, and 256 is 0.4 t/s faster, but
# the prefill-time KQ mask is ctx x ubatch x 4 bytes per attention-hosting
# card -- 128 MiB at ub 128, 256 MiB at ub 256, 2 GiB at ub 2048. Only ub 128
# survives a full-context prefill inside the guard. A load-time fit is NOT a
# safety proof on this model; measure at the depth you actually intend to use.
#
# WHY NOT ub 512/1024: Vulkan buffer sizing is NOT monotonic in ubatch on this model.
# ub 512 needs MORE R9700 VRAM than ub 1024 (32,539 vs 31,688 measured) and
# breaches. ub 2048 fits only with three CPU expert layers. Measure, never
# assume a smaller ubatch is safer. --batch-size is not a VRAM lever at all
# (2048 vs 512 gave byte-identical card totals).
#
# --device IS MANDATORY (Vulkan2 is the CPU/iGPU device). --load-mode mmap is
# MANDATORY. --tensor-read-lazy was part of every validated run.
#
# HEADROOM NOTE: the XTX also drives the desktop (gnome-shell, Xwayland,
# mutter, Ghostty, Firefox, Docker Desktop, surfshark, corectrl) and holds
# 1,499-2,100 MiB before llama-server starts. The R9700 is headless at 57 MiB.
# Freeing that ~2 GiB -- e.g. driving the display from the 9950X3D iGPU
# (card2, PCI 7d:00.0) -- is what would allow the all-GPU 262,144 config and
# is the single highest-value remaining change for decode speed and pp.
#
# MTP: still not usable for THIS preset. The draft costs 2.43 GiB resident;
# the two cards have 449 and 672 MiB spare at 262,144. It cannot fit without
# giving back roughly 90,000 tokens of context.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-Q2KXL-vulkan-fusions-262k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_UBATCH=128

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
