# Vulkan decode on Qwen3.8-27B UD-Q6_K_XL — 2026-09-03

## READ THIS FIRST: measure decode RAW, not through speculation

User, correcting this work mid-session: *"speculative decoding is gambling.
thats why we dont trust MTP numbers for decode. u should have done it raw."*

Correct, and it changed the conclusion. Draft acceptance varies with seed and
content, so MTP tg cannot isolate a hardware/placement effect. Measured here:

| | spread across seeds |
|---|---|
| `--spec-type none` | **0.02 - 0.07 tg** |
| MTP n3 | **5 - 24 tg** |

The MTP-based ratio sweep came out **non-monotonic** (ts 53,47 -> 46.90,
55,45 -> 45.99, 56,44 -> 48.21) and led to shipping the wrong config. The raw
sweep below is monotonic, reproducible, and picks a different winner.

**Always sweep placement/ratio/VRAM knobs with `--spec-type none`.** Layer MTP on
afterwards; it multiplies whatever raw throughput you have.

## RAW result (--spec-type none, ctx 262144, 3 seeds, aggregate tok/s)

Raw decode is content-independent (measured: code 29.64 vs prose 29.70 elsewhere
in this session), so one prompt suffices; depth still matters. Seed-to-seed range
was **0.02-0.07 tg**, so these differences are real.

| ts / KV | code10k | code40k | XTX raw | XTX w/ MTP | margin to guard |
|---|---|---|---|---|---|
| f16/f16 50,50 (preset 10) | 22.40 | 21.56 | 21,616 | 23,056 | 1,144 |
| f16K/q8V 50,50 | 22.37 | 21.63 | 19,726 | — | — |
| f16K/q8V 55,45 | 23.37 | 22.48 | 22,497 | 23,921 | 279 |
| f16K/q8V 57,43 | 24.00 | 23.13 | 22,994 | 24,426 | over |
| q8/q8 60,40 | 24.79 | 24.01 | 21,080 | 22,532 | 1,668 |
| q8/q8 61,39 | 25.00 | 24.23 | 21,278 | — | — |
| **q8/q8 62,38 (SHIPPED 17)** | **25.35** | **24.48** | 21,514 | **22,992** | **1,208** |
| q8/q8 63,37 | 25.90 | 25.08 | 21,981 | 23,482 | 718 |
| q8/q8 64,36 | 26.24 | 25.24 | 22,232 | 23,717 | 483 |
| q8/q8 65,35 | 26.39 | 25.48 | 22,430 | 23,875 | 325 |

**Shipped q8/q8 ts 62,38: +13.2% / +13.5% raw decode over preset 10, at an XTX
peak (22,992) slightly BELOW preset 10's (23,056)** — so it is faster without
increasing risk on the card that drives the display.

The curve is monotonic all the way to 65,35, and 63,37 / 64,36 both load and pass
correctness. They are documented rather than shipped because they cut the display
GPU's headroom to 718 / 483 MiB, and this machine's desktop baseline has
historically moved between 1,551 and 3,525 MiB.

Three facts only the raw sweep could show:

1. **q8_0 KV costs almost nothing.** At the *same* ratio, f16K/q8V gives 22.37 vs
   f16/f16's 22.40 — 0.13% — while freeing ~1,900 MiB. The freed VRAM is the
   entire point: it is what pays for the rebalance. So the KV choice is not a
   speed-vs-quality trade, it is a VRAM-vs-quality trade.
2. **The ratio effect is real and monotonic**, exactly as the bandwidth argument
   predicts. MTP noise had hidden it and inverted parts of the ordering.
3. **f16K/q8V is a poor middle ground.** At equal VRAM, q8/q8 at a higher ratio
   beats it: q8/q8 60,40 does 24.79 at XTX 21,080, while f16K/q8V 55,45 does only
   23.37 at 22,497 — slower *and* hungrier.

An earlier version of this document claimed **+17.7%** from MTP measurements and
shipped f16K/q8V ts 55,45. Both were wrong.

## First: the old headline number was not what it looked like

The preset advertised **25.4 tg**. That is a `llama-bench` figure with **no
speculation**. With MTP n3 and a real prompt the shipped preset does **44.34 tg**
on code at 10k. The Vulkan-vs-ROCm decode gap was 12%, not 16%.

## The win: the two GPUs are not symmetric

7900 XTX ~**960 GB/s**, R9700 ~**640 GB/s**. Decode is bandwidth-bound, and
`--split-mode tensor` makes both cards work on every layer with a sync between,
so **the slower card gates every step**. A 50,50 split leaves the XTX idling.
The share should track bandwidth, ~60/40. (First `--tensor-split` number is
Vulkan0 = the XTX.)

f16 KV leaves no room to shift: the XTX is already at 22,944 of a 24,200 guard,
and **ts 55,45 at f16 spilled** — xtx 24,285, tg collapsed 44.34 -> 28.42.
q8_0 KV frees ~3.9 GB and buys exactly the headroom the rebalance needs.

| config | code10k | prose10k | code40k | prose40k | XTX@40k |
|---|---|---|---|---|---|
| f16 ts50,50 (shipped) | 44.34 | 28.23 | 41.91 | 25.28 | 23,056 |
| q8_0 ts50,50 | 43.24 | 27.67 | — | — | — |
| q8_0 ts55,45 | 45.70 | 29.60 | — | — | — |
| **q8_0 ts60,40 (SHIPPED as 17)** | **48.99** | **30.75** | **47.28** | **26.28** | **22,532** |
| q8_0 ts65,35 | 52.08 | 34.21 | 48.73 | 27.57 | 23,875 |
| f16 ts55,45 | 28.42 | 18.52 | — | — | 24,285 spill |

**ts 60,40 is strictly better than the shipped preset**: faster on every cell AND
lower XTX VRAM (22,532 vs 23,056), because q8_0 KV saves more than the 10% weight
shift costs. q8_0 KV *alone* costs 2.5% (43.24 vs 44.34) — the rebalance repays
that several times over.

ts 65,35 is faster still but leaves only ~325 MiB under the guard on the card
that drives the display. Offered as an aggressive option, not the default.
ts 70,30 does not fit.

For reference the ROCm tensor preset does 50.46 code / 33.15 prose at 10k, so
this closes most of the Vulkan-vs-ROCm decode gap.

## What f16 can and cannot do — the user's objection, measured

The first version of this work shipped q8_0 on BOTH K and V. The user pushed
back: *"you are sacrificing context size and quality"*, then *"really we need
200k context minimum. Q8 is okay but FP16 is much better"*.

- **Context was never sacrificed.** Every preset here is ctx **262144**,
  identical to the shipped one. (User: *"oh okay, full context is cool"*.)
- **Quality was.** That objection was correct, and "strictly better" was the
  wrong phrase for a config that changes the KV cache.

So: how much decode is available *without* giving up f16? Measured at full
ctx 262144:

| config | code10k | XTX peak | verdict |
|---|---|---|---|
| f16/f16 ts50,50 (shipped) | 44.34 | 23,056 | the baseline |
| f16/f16 ts52,48 | 44.92 | **24,536** | +1.3%, sits at the 24,560 physical limit |
| f16/f16 ts53,47 | **27.69** | 24,168 | **SPILLED** — tg collapsed 38% |

**Preset 10 is already at the practical f16 ceiling.** The rebalance genuinely
requires freeing KV memory; there is no free f16 win. Note also that VRAM is
**not linear in the ratio** — 50->52 cost 1,401 MiB, far more than the ~270
MiB/point the q8 data suggested. Layer-boundary rounding again.

### The middle ground: quantize V only, keep K exact

| config | code10k | prose10k | code40k | prose40k | XTX peak |
|---|---|---|---|---|---|
| f16/f16 ts50,50 (preset 10) | 44.34 | 28.23 | 41.91 | 25.28 | 23,056 |
| **f16K/q8V ts55,45 (SHIPPED 17)** | **45.78** | **30.13** | **49.31** | **25.40** | **23,885** |
| f16K/q8V ts57,43 | 48.97 | — | 51.83 | — | 24,426 |
| f16K/q8V ts60,40 | 50.54 | — | 50.46 | — | 24,275 |
| q8/q8 ts60,40 | 48.99 | 30.75 | 47.28 | 26.28 | 22,532 |
| q8/q8 ts65,35 | 52.08 | 34.21 | 48.73 | 27.57 | 23,875 |

f16 K + q8_0 V is *faster than q8/q8 at the same ratio* (50.54 vs 48.99 code10k
at ts 60,40) while keeping K exact — plausibly because f16 K avoids dequantising
on the hot attention path while q8 V still halves that traffic.

ts 55,45 was measured twice independently and reproduced closely
(45.78/30.13/49.31/25.40 then 45.99/30.15/49.36/25.43). ts 56,44 was also tested:
48.21/30.19/48.82/26.91 at XTX peak 24,127 — faster at 10k but SLOWER on code at
40k, and only 73 MiB under the guard versus 279 for 55,45, so it is documented
rather than shipped.

**Shipped: f16K/q8V at ts 55,45** — +17.7% on code at 40k (the depth a coding
agent actually runs at), +3.2% code / +6.7% prose at 10k, XTX peak 23,885 which
stays under the 24,200 guard. The faster ts 57,43 / 60,40 variants exceed the
guard on the display GPU and are documented in the preset header rather than
shipped.

## A measurement bug worth remembering

Three arms "failed" at an identical `xtx=24515`/`24530` within seconds of each
other. That was **not** three capacity failures: `kill -9` does not release VRAM
immediately, so the next arm's guard read stale VRAM, and separately the guard
tripped on a **load-time transient** (f16 ts52,48 peaks ~24,530 then settles at
24,345 and runs fine). Fixes: wait for VRAM to actually drop below a threshold
between arms, and judge a config by whether it reaches READY and by its steady
peak, not by a transient.

## Correctness checked

llama.cpp's multi-gpu docs say `--split-mode tensor` requires a **non-quantized**
KV cache. It runs anyway on the `tp-adaptive` build, so it was verified rather
than assumed. Both ts 60,40 and ts 65,35 passed:

- greedy factual answer coherent — *"The capital of France is Paris, and the river
  that runs through it is the Seine."*
- a needle planted at 3k inside a **40k** context retrieved exactly:
  `74123-ZEBRA-9`

This is still q8_0 KV rather than f16 — a real if small quality choice. The f16
50,50 preset is retained unchanged (entry 10) for anyone who wants it.

## Measured and rejected

| lever | result |
|---|---|
| **master Vulkan build** (67a17c17c) | **cannot replace `tp-adaptive`.** Master's Vulkan rejects `--split-mode tensor` for this model: `ggml-backend-meta.cpp:1205 multi buffers are not supported by the meta backend`. Its layer split is also slightly SLOWER (37.15 vs 38.11 code10k). The custom #25051 port stays. |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=0` | 44.55 vs 44.34 code, 28.22 vs 28.23 prose — noise. Upstream's "-8% on dense" does not reproduce here either way. |
| `--split-mode layer` | 38.11 / 25.12 — 14% worse decode. Tensor split is correct for decode. |
| q8_0 KV at ts 50,50 | −2.5%. Only worth it as the enabler for the rebalance. |

## Not tested — needs root

`power_dpm_force_performance_level=high` is reported at **+14% TG** on RDNA4
(llama.cpp discussion #21043) and is `root`-owned (`Permission denied`).
`pcie_aspm.policy=performance` (+10.8% decode reported) likewise. Both are
one-line writes if the user wants them:

```
echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
```

Note the dpm one is reported as +14% TG but **−9% PP**, so it is a decode-only
trade. Both revert on reboot (or by writing `auto` / `default` back).
