#!/usr/bin/env bash
# Muse Glimmer 30B Q8_0 — ROCm dual-GPU, 256k via YaRN. 128k is the trained limit.
#
# VALIDATED BY RETRIEVAL, not merely by starting. A server will happily load a
# 256k context and then fail to attend past its trained window, so the acceptance
# criterion was needle-in-a-haystack at three depths:
#
#   prompt 149,237 tokens (well beyond the native 131,072)
#     depth 10%  FOUND   34.1 tg
#     depth 50%  FOUND   36.1 tg
#     depth 90%  FOUND   31.2 tg     -> 3/3
#   prefill 890.8 t/s
#   XTX 19.46 GiB (4.53 free) / R9700 17.86 GiB — only +1.27 GiB over the 128k
#   profile, because sliding-window attention keeps the KV cache tiny.
#
# This model extends unusually well: most attention is sliding-window (2048,
# pattern 4) and the global layers are NoPE, so there is far less RoPE geometry
# to distort than in a full-RoPE model.
#
# Flags verified present in this build (b10358) before use:
#   --rope-scaling {none,linear,yarn} · --rope-scale N · --yarn-orig-ctx N
#   --override-kv KEY=TYPE:VALUE  (types: int, float, bool, str)
# --override-kv is NOT optional: without it the model still reports 131072.
#
# DFlash is retained — it did not destabilise long context (retrieval 3/3 with it
# enabled). Use the 128k profile when you do not need the extra context; it has
# more headroom and does not pay YaRN's interpolation cost.
#
# 1M (rope-scale 8) is NOT shipped: it should only be attempted after 256k has
# been exercised in real use, and needs its own staged retrieval validation.

source "$LLAMA_SERVER_ROOT/config/profiles/default.sh"

export HIP_VISIBLE_DEVICES=0,1

PROFILE_NAME="preset-muse-glimmer-30b-q8-rocm-dual-256k-yarn"
LLAMA_BINARY="$LLM_ROOT/llama.cpp/build/bin/llama-server"
LLAMA_MODEL="$LLAMA_SERVER_ROOT/models/Muse-Glimmer-30B-Q8_0.gguf"
LLAMA_BACKEND="rocm"
LLAMA_DEVICE="ROCm0,ROCm1"
LLAMA_CONTEXT=262144
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
    # YaRN context extension 131072 -> 262144. Runtime override only; the GGUF is
    # NOT patched. --override-kv is required as well as the rope flags, otherwise
    # the model reports its trained 131072 and the extension is not applied.
    --rope-scaling yarn
    --rope-scale 2
    --yarn-orig-ctx 131072
    --override-kv muse-glimmer.context_length=int:262144
)
