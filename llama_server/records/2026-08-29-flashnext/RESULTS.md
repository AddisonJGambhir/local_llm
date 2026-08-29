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
