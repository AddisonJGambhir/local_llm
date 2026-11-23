# ============================================================================
# UD-Q6_K_XL + Vulkan TENSOR SPLIT -- DECODE-TUNED. ctx 262144 unchanged.
# q8_0 KV + ts 62,38. Chosen on RAW (--spec-type none) measurements.
#
# WHY RAW: speculative decoding is a gamble -- draft acceptance varies with seed
# and content, so MTP numbers cannot isolate a placement effect like the tensor
# split ratio. Measured in this session:
#     no-spec range across seeds  = 0.02-0.07 tg
#     MTP range across seeds      = 5-24 tg
# An earlier MTP-based sweep here produced a NON-MONOTONIC ratio curve and picked
# the wrong config. The raw sweep below is monotonic and reproducible.
#
# THE IDEA: the cards are NOT symmetric. 7900 XTX ~960 GB/s vs R9700 ~640 GB/s.
# Decode is bandwidth-bound and tensor split makes both cards work every layer
# with a sync between, so the SLOWER card gates every step. A 50,50 split leaves
# the XTX idling; the share should track bandwidth.
# (First --tensor-split number is Vulkan0 = the XTX.)
#
# RAW DECODE, --spec-type none, ctx 262144, 3 seeds, aggregate tok/s.
# Seed-to-seed range was 0.02-0.07 tg, i.e. these are real differences.
#
#   ts        code10k  code40k   XTX raw   XTX w/ MTP   margin to 24,200 guard
#   f16 50,50  22.40    21.56     21,616     23,056        1,144  <- preset 10
#   60,40      24.79    24.01     21,080     22,532        1,668
#   61,39      25.00    24.23     21,278       -             -
#   THIS 62,38 25.35    24.48     21,514     22,992        1,208
#   63,37      25.90    25.08     21,981     23,482          718
#   64,36      26.24    25.24     22,232     23,717          483
#   65,35      26.39    25.48     22,430     23,875          325
#
#   -> THIS is +13.2% / +13.5% raw decode over preset 10, and its XTX peak with
#      MTP loaded (22,992) is slightly LOWER than preset 10's (23,056), so it
#      does not increase risk on the card that drives the display.
#
# WHY STOP AT 62,38 when 65,35 is faster: the ratio curve keeps climbing, but so
# does XTX occupancy, and the XTX DRIVES THE DISPLAY. 62,38 was chosen to keep at
# least as much headroom as the preset it replaces. 63,37 (+2.2% more) and 64,36
# (+3.5% more) both load and pass correctness -- use them only if your desktop
# baseline is low and stable. This preset's own notes record the desktop baseline
# moving between 1,551 and 3,525 MiB historically, which is why margin matters.
#
# q8_0 KV COSTS ALMOST NOTHING RAW: at the SAME ratio, f16K/q8V ts50,50 gives
# 22.37 vs f16/f16's 22.40 -- a 0.13% difference, while freeing ~1,900 MiB.
# The freed VRAM is the whole point: it is what allows the rebalance.
#
# f16 CANNOT BE REBALANCED at full ctx. Measured: f16/f16 ts52,48 sits at the
# 24,560 physical limit for +1.3%, and ts53,47 SPILLS (tg collapses 44 -> 28).
#
# ALTERNATIVES, both one edit away:
#   ts 65,35 (same KV): +17.8% raw, but XTX 23,875 with MTP -- only ~325 MiB
#     under the guard, on the card that DRIVES THE DISPLAY.
#   K exact (LLAMA_CACHE_TYPE_K=f16) at ts 57,43: +7.1% raw and keeps the K cache
#     at full precision, but XTX ~24,426 with MTP, over the 24,200 guard.
#
# CORRECTNESS CHECKED (llama.cpp's multi-gpu docs say --split-mode tensor wants a
# non-quantized KV cache; it runs anyway on this build, so it was verified):
#   - greedy factual answer coherent ("Paris ... the Seine")
#   - needle planted at 3k inside a 40k context retrieved exactly (74123-ZEBRA-9)
#
# Build: llama.cpp-tp-adaptive. Master CANNOT replace it -- master's Vulkan
# rejects --split-mode tensor for this model:
#   ggml-backend-meta.cpp:1205 multi buffers are not supported by the meta backend
# See records/2026-09-03-27b-vulkan-decode/RESULTS.md
# ============================================================================

#!/usr/bin/env bash
source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

# Vulkan1 = R9700 (Vulkan0 is the 7900 XTX).
export MTMD_BACKEND_DEVICE=Vulkan1

PROFILE_NAME="preset-38-27b-UDQ6KXL-vulkan-TENSOR-q8kv-6238-256k"
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

LLAMA_CACHE_TYPE_K="q8_0"
LLAMA_CACHE_TYPE_V="q8_0"

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
    --tensor-split 62,38
    --batch-size 2048
    --no-mmap
    --mmproj "$LLAMA_SERVER_ROOT/models/mmproj-BF16.gguf"
)
