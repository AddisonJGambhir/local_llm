> **Historical record - not current operating guidance.**  
> Current instructions: [How to optimize](../../docs/HOW_TO_OPTIMIZE.md) | [Setup](../../docs/SETUP.md)

# Qwen3.8 27B Q6_K on Vulkan tensor split (PR #25051): full reproducible results

Dual-GPU AMD data for [ggml-org/llama.cpp PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051)
(vulkan: allreduce + Tensor Parallel). Every number below is measured on this machine; every command
is reproducible. Companion to the short comment posted on the PR.

## TLDR

On an RX 7900 XTX (24 GB, desktop-driving) + AI PRO R9700 (32 GB) PCIe-only topology, this PR's Vulkan
tensor split is the best configuration found for Qwen3.8 27B to date. It is the only setup that beats
ROCm layer-split prefill **and** ROCm tensor decode at the same time, and it unlocks full 262144 context
at Q6_K. Best measured: **889.6 pp (pp8192) / 43.4 tg (MTP n3)** at 262k, and
**709.1 pp / 54.5 tg (MTP n3)** at 128k.

## Hardware

| Component | Detail |
| --- | --- |
| GPU 0 (Vulkan0) | AMD Radeon RX 7900 XTX (NAV131), 24560 MiB. **Drives the desktop.** PCIe 5.0 x16, slot 1. |
| GPU 1 (Vulkan1) | AMD Radeon AI PRO R9700 (GFX1201), 32624 MiB. PCIe 4.0 x4 via chipset link (behind PCH, not a direct x16). |
| GPU 2 (Vulkan2) | Ryzen iGPU (Granite Ridge) - enumerated but not used by the server. |
| CPU | AMD Ryzen 9 9950X3D (16C/32T), socket `07:00.0` Navi31, `14:00.0` R9700 |
| RAM | 64 GB |
| Interconnect | PCIe only. **No P2P, no xGMI between the two dGPUs.** All tensor-mode allreduce traffic round-trips through system memory. |
| VRAM constraint | XTX must stay under ~24300 MiB; at 24388 MiB the compositor evicts to GTT and the desktop falls to ~3 fps. The XTX is the bottleneck card, the R9700 has headroom. |

## Software

| Component | Value |
| --- | --- |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-29-generic |
| Vulkan driver | RADV (AMD open source; not conformant, prints a warning at startup) |
| ROCm (for the ROCm-side measurements) | 7.1.52801, HIP/clang 21, `GGML_HIP_ARCHITECTURES=gfx1100;gfx1201`, `GGML_HIP_RCCL` OFF (default); RCCL ON tested separately, see Observations |
| Model | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q6_K.gguf`, 21.30 GiB, imatrix. Dense + SWA hybrid + recurrent state + built-in MTP. |
| Vulkan build (the PR) | `llama.cpp-vulkan-tp`, git worktree of branch `pr25051` (PR #25051, pwilkin), HEAD `13ac90156` ("vulkan: fix -sm tensor decode crawl, comm init robustness, add opt-in tree all-reduce"). CMake: `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release`. |
| ROCm build (baseline) | llama.cpp b10358 / `030ebb558`, CMake `-DGGML_HIP=ON` |
| KV cache | f16 everywhere (`--cache-type-k f16 --cache-type-v f16`), flash attention ON (required by tensor mode) |

Rebuild recipe for the PR worktree (stock llama.cpp build **cannot** tensor-split on Vulkan; it
silently ignores `--tensor-split` and puts everything on device 0):

```bash
cd llama.cpp
git fetch origin pull/25051/head:pr25051
git worktree add ../llama.cpp-vulkan-tp pr25051
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Exact server command (256k profile, as run)

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
(Profiles live in this repo: `config/profiles/preset-38-27b-q6k-vulkan-tensor-256k.sh` and
`preset-38-27b-q6k-vulkan-tensor-decode-128k.sh`.)

## Benchmark protocol

- `llama-bench` splits sweeps: `-sm tensor`, ub 1024, f16 KV, `-p 8192 -n 256`, r=2.
- Long-context pp/tg (ROCm baselines): fixed protocol `-d 65536 -p 10240 -n 256` (deep context,
  repeated 2x), plus a live-server probe (3,242-token prompt, 512-token generation) for MTP numbers.
- MTP: `draft-mtp`, n_max 3, draft KV q8_0, target KV f16.
- Multiple seeds where noted (a fixed-seed repeat only reproduces the same number).

## Measured numbers

### Vulkan tensor split sweep (PR #25051 build, ub 1024, f16 KV, pp8192 / tg256)

| tensor-split | pp t/s | tg t/s (no MTP) |
| --- | ---: | ---: |
| 50/50 | **889.60** | 23.17 |
| 60/40 | 811.35 | 25.63 |
| 65/35 | 770.92 | 27.47 |
| 70/30 | 729.82 | **29.15** (tg peak) |
| 75/25 | 701.73 | 29.14 |
| 80/20 | 670.47 | 28.09 |
| 85/15 | 642.86 | 26.23 |

Prefill falls monotonically as work moves to the XTX; decode has a real interior maximum at 70/30
and declines past it.

### Shipped Vulkan profiles (PR #25051 build, probe: 3,242-token prompt, 512-token generation, MTP n3)

| Profile | split | ctx | ubatch | pp (pp8192) | tg (MTP) | MTP acceptance | peak XTX / R9700 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 256k | 50/50 | 262144 | 1024 | 882.46 | **43.41** | ~53% | 22782 / 20792 MiB |
| 128k decode | 70/30 | 131072 | 512 | 709.11 | **54.52** | 59.7% (mean len 2.79) | 22437 / 10313 MiB |

MTP depth on this quant (ts 70/30, c 131072): n_max 3 = 54.52 tg / 59.7% acceptance; n_max 4 =
51.39 tg / 51.0%. n4 drafts longer, lands less.

ubatch at ts 70/30, c 131072: ub 512 = 54.52 tg, peak XTX 22437 MiB; ub 1024 = 53.55 tg, peak
23385 MiB (worth +2.7% prefill, costs ~950 MiB and a little decode).

Context at ts 70/30 ub 512: c 131072 = 54.52 tg, peak 22437 (margin 1863); c 147456 = 53.39 tg,
peak 23092; c 163840 rejected on margin (peak 23815, margin 485).

### ROCm baseline (master b10358, Q8_0, f16 KV, fixed long-ctx protocol)

| config | pp (pp10240 @ d65536) | tg (no spec) | tg (MTP n3, live) |
| --- | ---: | ---: | ---: |
| `-sm tensor` ts 45/55, ub 512, ~245k | 275.20 | 22.63 | 44.81 (53.98%) |
| `-sm layer` ts 1/1, ub 256, ~245k | 509.40 | 18.44 | 34.35 (53.46%) |

ROCm tensor: +25.5% decode vs layer, **-45.9% long-context prefill**. Split sweep at tensor
(35/65 = 273.89/20.47, 40/60 = 265.99/21.56, 45/55 = 263.93/22.48 pp/tg). ts 50/50 was faster
(300.16 pp / 24.29 tg; live MTP 48.0) but OOMs under real serving load (XTX 24487 of 24560).

Note: ROCm tensor baselines are Q8_0 (the Q6_K ROCm tensor number has not been measured yet;
Q8_0 does not hold 262k at a useful split, which is the whole reason Q6_K exists for this workload).

## Observations for the maintainers

1. **Vulkan tensor mode on AMD is now the best of three regimes** (Vulkan tensor, ROCm tensor,
   ROCm layer) for this model at both 128k and 262k. The CPU-proxy F16 allreduce appears far less
   penalized by this PCIe-only topology than the ROCm tensor path.
2. **RCCL makes zero measurable difference on ROCm tensor here**: `GGML_HIP_RCCL=OFF` vs `ON` gives
   the same prefill and decode within noise, and RCCL=ON segfaults occasionally (intermittent, no
   fixed shape reproduced). If RCCL can silently fall back to the generic path, this may be that;
   if it is active and doing nothing, the bottleneck is the interconnect (no P2P), not the collective.
3. **Topology matters**: the R9700 sits behind a Gen4 x4 chipset link. Allreduce traffic routed to
   it crosses that bottleneck; weighting toward it (ROCm ts 55/45) measured best in speed but is not
   shippable for VRAM reasons.
4. **Hybrid model**: Qwen3.8 27B is SWA + recurrent hybrid with built-in MTP. This PR handled it,
   including the MTP draft context. Worth confirming that stays clean at merge time: the docs'
   tensor-mode not-implemented list excludes Mamba/Mamba2 hybrids, and the meta backend has had
   multi-buffer issues (#22197).
5. **Vulkan-only gotcha**: `--main-gpu` does not cover the vision tower; `clip.cpp` only honours
   `MTMD_BACKEND_DEVICE` and otherwise grabs the first GPU (the constrained XTX). With `--mmproj`,
   set `MTMD_BACKEND_DEVICE=Vulkan1`.

## Known caveats

- The PR build is based on 2026-06-29 master and rejects flags added since (e.g. `--reasoning-preserve`).
  All numbers are for that base; re-measure when it lands on a newer master.
- The XTX drives the desktop, so its VRAM budget is ~24300 MiB, not 24560. Any config peaking above
  ~23900 MiB is in the danger zone: at 24180 MiB (ROCm tensor ts 55/45 at 262k) the server silently
  dropped to ~21% of prefill / ~62% of decode with no OOM and no crash (GTT spillover).
