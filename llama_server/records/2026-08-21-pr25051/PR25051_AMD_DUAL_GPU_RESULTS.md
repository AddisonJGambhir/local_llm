> **Historical record - not current operating guidance.**  
> Current instructions: [How to optimize](../../docs/HOW_TO_OPTIMIZE.md) | [Setup](../../docs/SETUP.md)

# PR #25051 AMD dual-GPU results: Qwen3.8 27B, full Q6_K / Q8_0 x tensor / layer x Vulkan / ROCm matrix

Complete measured data from an AMD-only box for [ggml-org/llama.cpp PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051)
(vulkan: allreduce + Tensor Parallel). Every number below was measured on this machine. Every prefill number is
annotated with the context depth it was measured at, because depth is the dominant variable: the same config runs
2-3x faster at ctx 0 than at 200k depth. Companion to the short comment posted on the PR.

## TLDR

RX 7900 XTX (24 GB, drives the desktop) + AI PRO R9700 (32 GB), PCIe only (no P2P, no xGMI between the cards).
f16 KV throughout.

- At matched 65k depth, Q8_0: this PR's Vulkan tensor does 561.7 t/s prefill, 2.04x ROCm tensor (275.2) and +10%
  vs ROCm layer (509.4). It is the only config that beats both ROCm regimes on prefill at depth.
- Decode splits by quant. At Q8_0, ROCm tensor keeps the MTP decode lead (44.8 vs 40.3 t/s). At Q6_K, this PR
  leads both axes: raw tg 29.2 vs 27.3, and MTP 54.5 at 128k vs 43.6 at 245k.
- Q6_K is what unlocks full 262144 context (Q8_0 Vulkan tensor tops out at 200704; Q6_K ROCm tops out at 245760
  for VRAM reasons).
- 7 of 8 cells in the quant x backend x split matrix are measured. The missing cell (Q6_K Vulkan layer) is
  called out where it matters.

## Depth protocols (read before comparing rows)

All pp/tg are `llama-bench` unless labeled live. The depth tag says what the measured chunk was prefilled on top of:

| Tag | Protocol | Meaning |
| --- | --- | --- |
| ctx 0 | `-p 8192 -n 256`, no `-d`, no `-c` | Empty context. Most optimistic number. |
| d65k | `-d 65536 -p 10240 -n 256` | Fill 65,536 tokens of context first, then measure the marginal rate on top of it. |
| d131k | `-d 131072 -p 2048 -n 32` | Same, at 131k depth. |
| d200k | `-d 200704 -p 2048 -n 128` | Same, at 200k depth. |
| 6.2k | 6.2k-token uncached prompt, 256-token decode at ~10k depth | The shallow serving harness from the 2026-08-16 audit. |
| live | llama-server, MTP n3, 4 fixed natural prompts x 400 tokens x 2 passes | Real serving at the profile's shipping context. |

"pp8192 at c N" means `llama-bench` pp8192 with an N-token context allocated but KV empty. It tests the
allocation, not N tokens of depth. Do not read it as a 262k-depth prefill. The "MEASURED HERE" pp numbers in
the Q6 Vulkan tensor profiles (882.46 at c 262144, 709.11 at c 131072) are such llama-bench runs, not live
server prefills; the same probe blocks carry the decode/MTP numbers, which ARE live.

## The matrix

f16 KV. pp = llama-bench prefill t/s. tg raw = unspeculated generation t/s (llama-bench cannot run MTP).
tg MTP = live server, n_max 3.

| Quant | Config (split) | Ship ctx | pp (depth) | tg raw (depth) | tg MTP (live) |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | Vulkan layer, 28/37 | 262144 | 825 (6.2k) | not measured | 35.0 (53.8% acc, ~10k depth) |
| Q8_0 | **Vulkan tensor, 50/50 (this PR)** | 200704 | 946.7 (ctx 0); 561.7 (d65k); 324.1 (d200k) | 21.4 (ctx 0) | 40.3 (54.5% acc) |
| Q8_0 | ROCm layer, 1/1 | 245760 | 982 (6.2k, 5 seeds); 1099.1 (d0); 509.4 (d65k) | 18.4 (d65k) | 34.4 (53.5% acc) |
| Q8_0 | ROCm tensor, 45/55 | 245760 | 275.2 (d65k) | 22.6 (d65k) | 44.8 (54.0% acc) |
| Q6_K | Vulkan layer | - | not measured (no profile exists) | - | - |
| Q6_K | **Vulkan tensor, 50/50 (this PR)** | 262144 | 889.6 (ctx 0); 882.5 (pp8192 @ c 262k) | 23.2 (ctx 0) | 43.4 |
| Q6_K | **Vulkan tensor, 70/30 (this PR)** | 131072 | 729.8 (ctx 0); 709.1 (pp8192 @ c 131k) | 29.2 (ctx 0) | 54.5 (59.7% acc) |
| Q6_K | ROCm layer, 38/27 | 245760 | 565.9 (d65k); 353.3 (d131k) | 20.0 (d65k); 18.1 (d131k) | 32.2 (51.6% acc) |
| Q6_K | ROCm tensor, 1/1 | 245760 | 294.5 (d65k) | 27.3 (d65k) | 43.6 (51.0% acc) |

## Matched-depth head-to-heads

**Q8_0 at d65536** (same quant, same protocol; the cleanest comparison in this document):

| Config | pp10240 @ d65k | tg2048 @ d65k (raw) |
| --- | --- | --- |
| Vulkan tensor 50/50 (this PR) | 561.7 | not measured at depth |
| ROCm tensor 45/55 | 275.2 | 22.6 |
| ROCm layer 1/1 | 509.4 | 18.4 |

**Q6_K at d65536:**

| Config | pp10240 @ d65k | tg2048 @ d65k (raw) |
| --- | --- | --- |
| ROCm layer 38/27 | 565.9 | 20.0 |
| ROCm tensor 1/1 | 294.5 | 27.3 |
| Vulkan tensor (this PR) | not measured at depth (ctx 0: 889.6 / 729.8) | not measured at depth (ctx 0: 23.2 / 29.2) |

Vulkan tensor's real-depth pp is measured only at Q8_0 (d65k, d200k). At Q6_K its measured pp (ctx 0 and
pp8192 at allocated context) sits 1.5-3x above the ROCm deep numbers, but that comparison crosses protocols.
The Q6_K Vulkan tensor d65k run is the honest missing cell; happy to add it.

**Live MTP ranking** (serving, at each config's shipping context):

54.5 (Q6 Vulkan tensor @128k) > 44.8 (Q8 ROCm tensor @245k) > 43.6 (Q6 ROCm tensor @245k) >
43.4 (Q6 Vulkan tensor @262k) > 40.3 (Q8 Vulkan tensor @200k) > 35.0 (Q8 Vulkan layer @262k) >
34.4 (Q8 ROCm layer @245k) > 32.2 (Q6 ROCm layer @245k)

## Hardware

| Component | Detail |
| --- | --- |
| GPU 0 (Vulkan0 / ROCm0) | AMD Radeon RX 7900 XTX (NAVI31), 24560 MiB. **Drives the desktop.** PCIe 5.0 x16. |
| GPU 1 (Vulkan1 / ROCm1) | AMD Radeon AI PRO R9700 (GFX1201), 32624 MiB. PCIe 4.0 x4 via chipset link (behind PCH, not a direct x16). |
| GPU 2 | Ryzen iGPU (Granite Ridge), enumerated but unused by the server. |
| CPU | AMD Ryzen 9 9950X3D (16C/32T). |
| RAM | 64 GB. |
| Interconnect | PCIe only. **No P2P, no xGMI between the two dGPUs.** All tensor-mode allreduce traffic round-trips through system memory. |
| VRAM constraint | XTX must stay under ~24300 MiB; at 24388 MiB the compositor evicts to GTT and the desktop falls to ~3 fps. The XTX is the bottleneck card; the R9700 has headroom. |

## Software

| Component | Value |
| --- | --- |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-29-generic |
| Vulkan driver | RADV (AMD open source; not conformant, prints a warning at startup) |
| ROCm (ROCm-side measurements) | 7.1.52801, HIP/clang 21, `GGML_HIP_ARCHITECTURES=gfx1100;gfx1201`, `GGML_HIP_RCCL` OFF (default); RCCL ON tested separately, see Observations |
| Model | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q6_K.gguf` (21.30 GiB) and `Qwen3.8-27B-Q8_0.gguf` (27.05 GiB), imatrix. Dense + SWA hybrid + recurrent state + built-in MTP. |
| Vulkan tensor build (this PR) | `llama.cpp-vulkan-tp`, git worktree of branch `pr25051` (PR #25051, pwilkin), HEAD `13ac90156`. CMake: `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`. Used for every "Vulkan tensor" row. |
| Stock Vulkan build | `llama.cpp/build-vulkan`, same base as the ROCm build. Used for the Q8_0 "Vulkan layer" row only. Stock Vulkan cannot tensor-split: it silently ignores `--tensor-split` and puts every weight on device 0 (measured: 23,865 MiB on the XTX, 57 MiB on the R9700, then death). |
| ROCm build (baselines) | llama.cpp b10358 / `030ebb558`, CMake `-DGGML_HIP=ON`. Used for every "ROCm" row. |
| KV cache | f16 everywhere (`--cache-type-k f16 --cache-type-v f16`), flash attention ON (required by tensor mode) |

Rebuild recipe for the PR worktree:

```bash
cd llama.cpp
git fetch origin pull/25051/head:pr25051
git worktree add ../llama.cpp-vulkan-tp pr25051
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Exact server command (Q6_K 256k profile, as run)

```bash
llama.cpp-vulkan-tp/build/bin/llama-server \
  -m models/Qwen3.8-27B-Q6_K.gguf \
  --device Vulkan0,Vulkan1 -ngl 99 -fit off \
  -c 262144 -t 16 -tb 32 --parallel 1 \
  --cache-type-k f16 --cache-type-v f16 --flash-attn on \
  --ubatch-size 1024 --cache-ram 24576 --no-context-shift \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --metrics --port 1234 --host 127.0.0.1 --timeout 3600 --log-colors off --jinja \
  --ctx-checkpoints 8 --split-mode tensor --main-gpu 1 --tensor-split 50,50 \
  --batch-size 2048 --no-mmap --mmproj models/mmproj-BF16.gguf
```

128k decode profile: same, with `-c 131072 --ubatch-size 512 --tensor-split 70,30`.
Profiles for every row in the matrix live in this repo under `config/profiles/`
(`preset-38-27b-{q6k,q8}-...sh`).

## Per-cell notes

### Q8_0 Vulkan tensor (this PR): ts 50/50, ub 1024, c 200704

Split sweep (llama-bench, `-sm tensor`, ub 1024, f16 KV, pp8192/tg256, ctx 0):

| tensor-split | pp t/s | tg t/s (raw) |
| --- | ---: | ---: |
| 30/70 | 838.92 | 17.58 |
| 40/60 | 902.23 | 19.45 |
| 45/55 | 936.86 | 20.31 |
| 50/50 | **946.74** | 21.37 (pp peak, shipped) |
| 60/40 | 873.63 | 24.01 |
| 65/35 | 834.37 | 25.80 |
| 70/30 | 801.09 | 25.90 (tg peak, but see VRAM) |

Decode peaks toward the XTX (the R9700 sits behind the Gen4 x4 chipset link), but those splits cannot hold
context: 65/35 exceeds the XTX ceiling by c=98304. 50/50 is the only point that is both fast and able to hold 200k.

Deep-context pp with the same settings: 946.7 (ctx 0) -> 561.7 (d65k) -> 324.1 (d200k). The drop from ctx 0 to
d65k is real depth degradation, same as on ROCm; the point of this table is that at matched d65k depth Vulkan
tensor is still 2.04x ROCm tensor.

Context ceiling (measured with real work, `-p 2048 -n 128`): 200704 -> 23,538 MiB exit 0 (shipped, 762 MiB
under ceiling); 225280 -> 24,122 MiB exit 0 but only 178 MiB spare, rejected. A trivial `-p 64 -n 1` probe
would have said 196,608 was the max; real work at 200k needs ~1 GB more than that probe implied.

MTP: 40.29 t/s at 54.5% acceptance (live, c 200704). The profile header cross-references an earlier looser
ROCm tensor MTP figure of 48.83 t/s at 55.1%; the fixed-protocol A/B in the ROCm tensor profile (4 prompts x
400 tokens x 2 passes) gives 44.81 at 53.98%. This document uses 44.81 everywhere, and does not compare it
to the Q6 rows because the MTP probes differ between the two quant profiles.

### Q8_0 Vulkan layer (stock Vulkan, pre-PR): ts 28/37, ub 512, c 262144

825 pp / 35.0 tg (MTP, 53.8% acceptance), shallow protocol (6.2k-token uncached prompt, 256-token decode at
~10k depth). This is the pre-PR Vulkan path and it loses to ROCm layer at shallow depth (982 pp / 38.5 tg
median across 5 seeds), which is the baseline this PR has to beat on the Vulkan side. It is also the fallback
if the PR build is not available: same stock build, `--split-mode layer`, `--main-gpu 1`.

### Q8_0 ROCm (b10358)

Layer 1/1, ub 256, c 245760: 1099.1 pp (d0) / 509.4 pp (d65k); 18.4 tg raw (d65k); 34.4 MTP live.
The former c 262144 shipped setting is no longer safe: it allocated 23,927 MiB at load and peaked 24,150 under
load, 50 MiB from the 24,200 hard ceiling. 245760 was validated over 8 x 400-token MTP samples (peak 23,225).

Tensor 45/55, ub 512, c 245760: 275.2 pp (d65k); 22.6 tg raw (d65k); 44.8 MTP live (54.0% acceptance).
Tensor split sweep at d65k: 35/65 = 273.89/20.47, 40/60 = 265.99/21.56, 45/55 = 263.93/22.48 (pp/tg, ub 256).
ts 50/50 was faster (300.16 pp / 24.29 tg; live MTP 48.0) but OOMs under real serving load: the load-time
plateau said 22,670 MiB, production serving hit 24,487 of 24,560 with the svm eviction worker already firing.
The lesson: size tensor-mode VRAM while serving, not at the load-time plateau.

Known ROCm quirk on this host (unrelated to the PR): one image request permanently drops text prefill ~40%
until restart (2026-08-15, ruled out batching/thermals/clocks/prompt cache/GTT spill; matches upstream
#26873, unverified).

### Q6_K ROCm (b10358)

Layer 38/27, ub 256, c 245760: 565.9 pp (d65k) / 353.3 pp (d131k); 20.0 / 18.1 tg raw at the same depths;
32.2 MTP live (51.6% acceptance, from the tensor A/B baseline). The d131k marginal rate is the number real
long sessions feel. The XTX-weighted split (38/27 over the inherited 1/1) was unlocked by the quant: Q6_K
frees 5.74 GiB of weights, ~2.9 GiB of it on the XTX, which made XTX-loading splits fit for the first time.
Mechanism: the R9700 sits pinned at its 300 W cap during prefill, and in layer mode the GPUs run sequentially
per token, so its downclock adds to every token.

Tensor 1/1, ub 256, c 245760: 294.5 pp (d65k); 27.3 tg raw (d65k); 43.6 MTP live (51.0% acceptance), +35.2%
over its layer sibling (32.2), paired 95% CI +9.35 to +13.34 t/s. Even split because Q6_K's 21.30 GiB fits
roughly half on each card; the 34/31 skew that won on the Q8 tensor profile hit the XTX ceiling (24,226 MiB)
and was rejected.

### Q6_K Vulkan tensor (this PR)

Why Q6_K here: 5.74 GiB lighter than Q8_0, and that saving is what buys full 262144 context on Vulkan tensor
(Q8_0 tops out at 200704).

Split sweep (llama-bench, `-sm tensor`, ub 1024, f16 KV, pp8192/tg256, ctx 0, r=2):

| tensor-split | pp t/s | tg t/s (raw) |
| --- | ---: | ---: |
| 50/50 | **889.60** | 23.17 (pp peak, shipped for 256k) |
| 60/40 | 811.35 | 25.63 |
| 65/35 | 770.92 | 27.47 |
| 70/30 | 729.82 | **29.15** (tg peak, shipped for 128k decode) |
| 75/25 | 701.73 | 29.14 |
| 80/20 | 670.47 | 28.09 |
| 85/15 | 642.86 | 26.23 |

Prefill falls monotonically as work moves to the XTX; decode has a real interior maximum at 70/30 and declines
past it.

**The GTT cliff.** At c 262144, ts 50/50 leaves 1,518 MiB of XTX headroom, which looks like room to buy decode
by shifting weights to the XTX. It is not:

| c 262144 | peak XTX | pp | tg (MTP) |
| --- | --- | --- | ---: |
| ts 50/50 | 22,782 MiB | 671 (live probe) | 43.41 (shipped) |
| ts 55/45 | 24,180 MiB | 140 (live probe) | 27.00 (collapse) |

Both rows are live-server measurements under load at c 262144 (not llama-bench), which is why they read
well below the pp8192 numbers: the 50/50 row is the same live probe as the shipped-decode measurement
below. The pp/tg here exist to show the collapse, not to bench the split.

Five split points cost 1,398 MiB and push the XTX into GTT spillover. The failure mode is not an OOM and not a
crash: the server starts, serves, and quietly runs at 21% of prefill and 62% of decode. Tuning by speed alone
reads this as "tensor split got slower" and never finds the cause. Anything peaking above ~23,900 MiB on this
card is in that zone.

ubatch at ts 50/50, c 262144 (pp8192): ub 512 = 823.91, ub 1024 = 882.46 (shipped), ub 2048 = 910.88.
ub 2048 was rejected on memory: 23,953 MiB peak, inside the cliff zone above.

**Shipped decode (live probe: 3,242-token prompt, 512-token generation, MTP n3, c 262144):** 43.41 tg (MTP);
peak XTX 22,782 MiB, R9700 20,792 MiB. The 256k profile does not record an acceptance figure for this probe;
the closest acceptance on this build is the 128k decode profile's 59.7% (a different split and context).

MTP depth at ts 70/30, c 131072: n_max 3 = 54.52 tg, 59.7% acceptance, mean accepted 2.79; n_max 4 = 51.39 tg,
51.0% acceptance. n4 drafts a longer chain and lands less of it.

Context at ts 70/30 ub 512: c 131072 = 54.52 tg, peak 22,437 MiB (margin 1,863); c 147456 = 53.39 tg, peak
23,092; c 163840 rejected on margin (peak 23,815).

## Observations for the maintainers

1. **The CPU-proxy F16 allreduce is far less penalized by a PCIe-only topology than the ROCm tensor path.**
   At matched d65k depth and Q8_0, Vulkan tensor prefill (561.7) is 2.04x ROCm tensor (275.2) even though
   both round-trip allreduce through system memory on this box (no P2P). The gap is in the path, not the
   collective.
2. **RCCL makes zero measurable difference on the ROCm tensor side here.** `GGML_HIP_RCCL=OFF` vs `ON` gives
   the same prefill and decode within noise, and RCCL=ON segfaults intermittently (no fixed shape
   reproduced). Either it silently falls back to the generic path, or it is active and the bottleneck is the
   interconnect rather than the collective. Worth a code-path check if anyone else sees flat RCCL numbers.
3. **Topology matters for split choice.** The R9700 is behind a Gen4 x4 chipset link. Allreduce traffic routed
   to it crosses that bottleneck; the decode-interior-max at 70/30 (toward the XTX) is consistent with that.
   Weighting toward the R9700 measured best in speed on the ROCm tensor side (ts 55/45) but is not shippable
   for VRAM reasons.
4. **Hybrid model.** Qwen3.8 27B is SWA + recurrent hybrid with built-in MTP, and this PR handled it,
   including the MTP draft context. Worth confirming at merge time that it stays clean: the docs'
   tensor-mode not-implemented list excludes Mamba/Mamba2 hybrids, and the meta backend has had multi-buffer
   issues (#22197).
5. **Vulkan-only gotcha.** `--main-gpu` does not cover the vision tower; `clip.cpp` only honours
   `MTMD_BACKEND_DEVICE` and otherwise grabs the first GPU (the constrained XTX). With `--mmproj`, set
   `MTMD_BACKEND_DEVICE=Vulkan1`.

## Not measured (stated plainly)

- Q6_K Vulkan layer: no profile exists. Layer mode on stock Vulkan is the pre-PR path, and the only Vulkan
  layer numbers here are the Q8_0 shallow ones.
- Q6_K Vulkan tensor pp/tg at real depth (d65k/d131k): only ctx 0 and pp8192 at allocated context (KV empty).
  The Q6 numbers in the matrix are therefore not directly comparable to the Q6 ROCm deep rows; flagged in the
  head-to-head.
- Q8_0 Vulkan tensor tg at depth: only ctx 0 raw and live MTP.
- Q8_0 Vulkan layer deep pp: only the shallow 6.2k harness.
- Concurrent requests (`--parallel` stays 1; this is a hybrid model and the recurrent state does not like
  it), vision-request throughput, quantized KV under tensor mode (documented unsupported, this build accepts
  and runs it, untested for quality).

## Known caveats

- The PR build is based on 2026-06-29 master and rejects flags added since (e.g. `--reasoning-preserve`).
  All Vulkan tensor numbers are for that base; re-measure when it lands on a newer master.
- The XTX drives the desktop, so its VRAM budget is ~24,300 MiB, not 24,560. Any config peaking above
  ~23,900 MiB is in the danger zone described under the GTT cliff.
- MTP numbers are live-server numbers at each config's shipping context; llama-bench cannot run speculative
  decode, so all "tg raw" values are unspeculated.
