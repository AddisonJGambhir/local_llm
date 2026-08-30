# ============================================================================
# Qwen3.8-Flash-Next UD-IQ3_XXS -- Vulkan LAYER split, 81920 ctx -- 2026-08-29
# ============================================================================
# pp 687.9 t/s, tg 42.5 @d0 / 29.2 @d9060. Everything is in VRAM except the
# PLE n-gram table.
#
# MEASURE PREFILL WITH A REAL PROMPT. A 20-token prompt reports pp ~50 t/s on
# this config, which is startup overhead, not throughput. The figures here use a
# 9,060-token prompt (src/benchmarks/prompts/ concatenated).
#
# BUILD: llama.cpp-qwen4exp/build-vulkan @ aa1b06d0c = PR #27880 (contains the
# qwen4exp arch PR #27742). Mainline llama.cpp CANNOT load this model.
#
# WHY IQ3_XXS AND NOT IQ4_XS: the PLE n-gram table is a FIXED 27,466 MiB
# (26.8 GiB) at IQ4_NL in EVERY unsloth quant -- measured identical in IQ4_XS
# and IQ3_XXS. It is a constant tax, so the only thing that varies is the body:
#   quant        total      - PLE   = main model   fits ~54 GiB VRAM?
#   UD-IQ4_XS    87.3 GiB    26.8     60.5 GiB     no, needs ~7 layers offloaded
#   UD-Q3_K_XL   83.8 GiB    26.8     57.0 GiB     no, ~3 GiB over
#   UD-IQ3_XXS   76.4 GiB    26.8     49.6 GiB     YES, zero expert offload
# IQ3_XXS is the first quant needing NO expert offload, and that is worth far
# more than the extra bits: IQ4_XS with 7 layers on CPU measured tg 9.14.
#
# WHY ts 38,62 AND ctx 81920 (measured, settled VRAM after real generation):
#   ctx      ts      XTX      R9700    tg
#   8192     39,61   23244    29955    43.20
#   65536    39,61   23354    30957    42.87
#   81920    38,62   22614    32393    42.35   <-- shipped
#   90112    37,63   22698    32558    42.42   (R9700 has 9 MiB left -- too thin)
#   94208    37,63   22786    31796    11.77   <-- CLIFF
#   98304    38,62   22839    31843    11.78   <-- CLIFF
# The cliff between 90112 and 94208 is abrupt and total. 81920 keeps 1,586 MiB
# of XTX margin, which covers the desktop baseline swings; 90112 does not.
#
# WHY q8_0 KV: f16 is FASTER where it fits (43.50 vs 42.77 at ctx 65536) but
# pushes past the cliff at 81920 (measured tg 11.82). q8_0 is what makes 81920
# reachable at full speed.
#
# WHY -ub 1024 (9,060-token prompt, ctx 81920):
#   ub 512   pp 576.6      ub 2048/b4096   pp 69.4   <- CLIFF
#   ub 1024  pp 694.8      ub 2048/b16384  pp 69.2   <- CLIFF
#                          ub 4096/b8192   pp 70.6   <- CLIFF
# ub 1024 is +20% prefill over the default and costs nothing on decode. Anything
# above it needs compute buffers that do not fit and spills.
#
# WHY ts 385,615: --tensor-split takes PROPORTIONS, not percentages, so finer
# steps than 1%% are available -- but layer assignment is discrete over 48 layers
# so most fine steps quantise to the same placement (370,630 and 375,625 gave
# byte-identical VRAM). 385,615 is the step that lands one layer differently:
#   ts        XTX     margin   R9700   pp
#   38,62     23429      771   32491   699.1
#   370,630   23376      824   32490   691.1
#   385,615   23140     1060   32490   687.9   <-- shipped
#   36,64     22006     2194   32446    67.0   <- CLIFF
# THE R9700 IS THE BINDING CONSTRAINT and sits at 32,490 (77 MiB spare) in every
# working config. Moving weight OFF the XTX does not help -- it pushes the R9700
# past its limit and falls off the cliff. 385,615 buys 289 MiB of XTX headroom
# for 1.6%% of prefill, which is worth it because the XTX drives the desktop and
# its baseline was observed between 1,479 and 3,525 MiB in a single day.
#
# WHY --flash-attn on: worthless alone (6.49 -> 6.47 on IQ4_XS) but +39% when
# paired with quantised KV. Do not ship one without the other.
#
# MTP: DO NOT USE. It loads only on LaurentZuijdwijk's qwen4exp/mtp-fix fork
# (mainline aborts: "tensor 'blk.0.hc_attn_norm.weight' not found"), and on that
# fork it is a large NET LOSS on this model -- measured tg 19.73 (n1), 19.19
# (n2), 21.84 (n3), 19.02 (adaptive) against 40.98 with MTP off on the same
# build. The draft forward costs more than the accepted tokens save because the
# target already decodes at ~43 t/s. MTP wins when the target is slow; this is
# not that case. Acceptance was high (99% at n3) and it still lost.
#
# ALSO TRIED, ALL SLOWER: qwen4exp/ple-sidecar fork (pp 47.5 / tg 34.62, WIP),
# the mtp-fix fork without MTP (pp 22.8 / tg 40.98 -- much worse prefill),
# -ub 2048/-b 4096 (no change), -ub 256 (worse pp).
# vulkan/iq3s-mmv-register-spill does not contain qwen4exp and cannot load this
# model; its commit would have to be cherry-picked onto the qwen4exp tree.
#
# -sm tensor IS UNAVAILABLE for this model: it aborts with "multi buffers are
# not supported by the meta backend" whenever any tensor is CPU-resident, and
# this model cannot fit without the PLE on CPU. -sm row is unimplemented on RADV.
#
# --device IS MANDATORY. This box enumerates THREE Vulkan devices; Vulkan2 is
# the CPU (RADV RAPHAEL_MENDOCINO) advertising 32 GB free, and auto-selection
# will silently include it and destroy throughput.
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

PROFILE_NAME="preset-flashnext-IQ3XXS-vulkan-LAYER-82k"
LLAMA_BINARY="$LLM_ROOT/llama.cpp-qwen4exp/build-vulkan/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Qwen3.8-Flash-Next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
LLAMA_BACKEND="vulkan"

LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=81920
LLAMA_UBATCH=1024

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
    --tensor-split 385,615
    --batch-size 2048
    --flash-attn on
    --load-mode none
    -ot 'per_layer_token_embd\.weight=CPU'
)
