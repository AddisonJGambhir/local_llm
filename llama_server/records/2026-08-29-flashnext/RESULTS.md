# Qwen3.8-Flash-Next (qwen4exp) on 7900 XTX + R9700 -- 2026-08-29

Model: unsloth UD-IQ4_XS, 93.7 GB, 3 shards. Also downloaded: UD-Q3_K_XL (90 GB).
Build: llama.cpp-qwen4exp @ aa1b06d0c = PR #27880 (which contains PR #27742).
  build-vulkan  WORKS
  build-rocm    builds clean, cannot fit the model (see below)

## WORKING CONFIG (Vulkan)

    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 llama-server -m <IQ4_XS shard 1> \
      -c 8192 -ngl 99 --device Vulkan0,Vulkan1 \
      --split-mode layer --tensor-split 40,60 -lm none \
      -ot 'per_layer_token_embd\.weight=CPU' \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0

  measured  pp 23.0 t/s   tg 9.14 t/s   (c=8192, 128-token generation)
  VRAM      XTX 24,148 MiB   R9700 32,486 MiB

## MEASURED MODEL FACTS (from -v tensor dump, not arithmetic)

  per_layer_token_embd.weight   27,465 MiB (26.8 GiB)  iq4_nl  SINGLE TENSOR
  per-layer experts             1,136 MiB  (down 450 iq4_nl + gate 343 + up 343 iq3_s)
  blocks 48, full_attention_interval 4 -> 12 attention layers, 36 GDN
  head_count_kv 2, k/v length 256 -> 24 KiB/token f16 -> 6.0 GiB at 262144

## SSD OFFLOAD IS NOT NEEDED

Measured disk throughput during sustained generation: 0.0-0.2 MB/s. The n-gram
table is 26.8 GiB and lives entirely in the 59 GiB of RAM. There is no paging.
Do not build a preset around SSD offload; it is not what is happening.

## WHAT MOVED THE NUMBER

  baseline (layer, mmap)                    tg 5.96
  -lm none instead of mmap                  tg 6.49   (+9%)
  + --flash-attn on                         tg 6.47   (flat alone)
  + --flash-attn on + q8_0 KV               tg 9.03   (+39% over -lm none)
  + PR #27880                               pp 14.6 -> 23.0 (+58%), tg flat

Flash attention alone does nothing; flash attention WITH q8_0 KV is the win.
PR #27880 ("reduce number of graph splits") fixes prefill only.

## DEAD ENDS -- DO NOT RETEST

  -sm tensor    ABORTS: "multi buffers are not supported by the meta backend".
                Tensor split cannot coexist with ANY CPU offload, and this model
                cannot fit without offloading the n-gram table. Structurally
                unavailable for this model.
  -sm row       Vulkan: "device Vulkan0 does not support split buffers"
                ROCm:   "device ROCm0 does not support split buffers"
  ROCm backend  Always tries to allocate 34,621.23 MiB on device 1 (32,624 MiB
                card) REGARDLESS of --tensor-split or how many expert layers are
                offloaded. The number never changes. ROCm wants ~58,680 MiB
                total vs 55,267 available. Not viable.

## llama.cpp FLAG GOTCHAS FOUND

  -ncmoe is SILENTLY IGNORED when -ot is also passed (same override list).
  Repeated -ot flags DO NOT ACCUMULATE -- only the first applies.
  Use ONE -ot with comma-separated pattern=buffer pairs.
  Expert tensors are blk.N.ffn_{down,gate,up}_exps.weight.
  Overrides land in ROCm_Host / CPU buffer types.

## OPEN: decode is ~3% of hardware

6B active at ~4.15 bpw is ~3.1 GB/token; 9.14 t/s is ~28 GB/s against ~800 GB/s
of aggregate bandwidth. Profiling during generation shows the two GPUs
ALTERNATING, never co-saturating (XTX 32% while R9700 73%, then inverted) --
layer split serialises them, and tensor split is unavailable. CPU sits at ~38%
with no disk I/O. Community reports 16-17 t/s on 2x RTX 3080 (20 GB total) and
44.7 t/s on 2x 4090D, so this is not a hardware ceiling.
Prime suspect: Vulkan GDN/QSA kernels (36 of 48 layers are Gated DeltaNet) are
new and unoptimised. Not yet isolated.

---

# ROUND 2 (same day): 9.14 -> 42.99 t/s by switching quant

The IQ4_XS result above was bottlenecked by expert offload, not by kernels.
UD-IQ3_XXS is the first quant whose main model fits VRAM with ZERO expert
offload, and that single change is worth 4.7x.

## Shipped config

    preset-flashnext-iq3xxs-vulkan-layer-82k.sh    (Vulkan menu 12)
    build llama.cpp-qwen4exp/build-vulkan @ aa1b06d0c (PR #27880)
    ctx 81920, layer split 38,62, q8_0 KV, --flash-attn on, --load-mode none,
    -ot 'per_layer_token_embd\.weight=CPU', --device Vulkan0,Vulkan1
    ==> pp 50.9 t/s   tg 42.99 t/s   XTX 22,614 MiB   R9700 32,393 MiB

## The PLE table is a FIXED tax across quants (measured, both files)

    UD-IQ4_XS   per_layer_token_embd.weight  IQ4_NL  27,465 MiB
    UD-IQ3_XXS  per_layer_token_embd.weight  IQ4_NL  27,466 MiB

Unsloth keeps it at IQ4_NL regardless of body quant. Only the body varies:

    quant        total     - PLE  = main    zero-offload fit in ~54 GiB?
    UD-IQ4_XS    87.3 GiB   26.8    60.5    no  -> 7 layers to CPU -> tg 9.14
    UD-Q3_K_XL   83.8 GiB   26.8    57.0    no  -> ~3 GiB over
    UD-IQ3_XXS   76.4 GiB   26.8    49.6    YES -> tg 42.99

## Context cliff is abrupt

    ctx     ts      XTX     R9700   tg
    8192    39,61   23244   29955   43.20
    32768   39,61   23539   30389   42.58
    49152   39,61   23715   30673   42.89
    65536   39,61   23354   30957   42.87
    81920   38,62   22614   32393   42.35   <- shipped
    90112   37,63   22698   32558   42.42   (R9700 9 MiB spare)
    94208   37,63   22786   31796   11.77   <- cliff
    98304   38,62   22839   31843   11.78   <- cliff

f16 KV is faster where it fits (43.50 vs 42.77 at 65536) but falls off the
cliff at 81920 (11.82). q8_0 is what buys the context.

## MTP: LOADS ON THE FORK, BUT IS A NET LOSS. DO NOT SHIP IT.

Mainline aborts with "tensor 'blk.0.hc_attn_norm.weight' not found" -- the MTP
graph is only built by LaurentZuijdwijk/llama.cpp branch qwen4exp/mtp-fix
(337c8bb58), which also needs LLAMA_ATTN_ROT_DISABLE=1.

On that fork, at ctx 32768 / ts 42,58 / draft on Vulkan1:

    MTP off      tg 40.98
    n-max 1      tg 19.73
    n-max 2      tg 19.19
    n-max 3      tg 21.84   (acceptance 99%)
    adaptive     tg 19.02
    n-max 5      tg  5.94

High acceptance and still a 2x loss: the draft forward costs more than the
accepted tokens save when the target already decodes at ~43 t/s. Draft
PLACEMENT matters hugely (Vulkan1 21.84 vs Vulkan0 8.14) but never enough.

## Branch comparison, identical config (ctx 81920, ts 38,62)

    llama.cpp-qwen4exp  PR #27880       pp 50.9  tg 42.99   <- BEST
    fork qwen4exp/mtp-fix, MTP off      pp 22.8  tg 40.98   (much worse prefill)
    fork qwen4exp/ple-sidecar (WIP)     pp 47.5  tg 34.62
    fork vulkan/iq3s-mmv-register-spill  cannot load: no qwen4exp in that branch

The iq3s-mmv-register-spill commit (d19f92093) targets exactly the IQ3_S mat-vec
path this quant uses and was never evaluated -- it would need cherry-picking
onto the qwen4exp tree. That is the most promising untried lead.
