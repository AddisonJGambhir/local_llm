# Tuning llama.cpp on dual AMD GPUs: what worked, what didn't

Measured 2026-08-20/21 on one machine. Every number here came off this box, with
`llama-bench -r 3` or `-r 5` unless stated. Nulls are listed as prominently as
wins, because the nulls are what save you a weekend.

**Status:** MTP and the prefill decay curve are complete. The 100k wall-clock
encode is still running.

---

## The box

| | |
|---|---|
| GPU 0 | RX 7900 XTX, gfx1100, RDNA3, 24,560 MiB, **960 GB/s**. Drives the desktop. |
| GPU 1 | AI PRO R9700, gfx1201, RDNA4, 32,624 MiB, **~640 GB/s** |
| Link | XTX: Gen4 x16 CPU-direct (~31.5 GB/s). R9700: **Gen4 x4 through 4 chipset bridges** (~7.9 GB/s). **No P2P.** |
| CPU / RAM | Ryzen 9 9950X3D, 64 GB |
| OS | Ubuntu 26.04, kernel 7.0.0-29, in-tree amdgpu, Mesa RADV |
| ROCm | 7.1.0-0ubuntu2 (Ubuntu packaging, NOT AMD's repo) |
| Model | Qwen3.8-27B dense, arch `qwen35`, 65 blocks, hybrid (has `ssm.*` params) |

The asymmetry matters. Most multi-GPU advice assumes matched cards on matched
links. Here one card has 50% more bandwidth and 4x the PCIe width.

---

## WHAT WORKED

### 1. Tune the tensor-split ratio. +8.2% decode.

The single biggest configuration win. Decode has a real interior maximum and it
is **not** 50/50 and **not** "as much as fits on the fast card".

ROCm, `-sm tensor`, Q6_K, `-p 0 -n 512 -r 3`:

| `-ts` | tg512 |
|---|---|
| 50/50 | 30.19 |
| 54/46 | 31.75 |
| 56/44 | 32.35 |
| **58/42** | **32.67** |
| 60/40 | 32.47 |
| 62/38 | 32.01 |
| 65/35 | 31.39 |
| 70/30 | 29.65 |

Confirmed on Q8_0 independently: same 58/42 optimum (26.36 → **28.82** → 27.61
across 50/50 → 58/42 → 66/34).

**The trap:** an earlier sweep tested only 50/50 and 70/30. 70/30 came out WORSE
than 50/50, which made 50/50 look optimal. The peak was sitting between the two
probes. If you sweep a ratio, use at least 5 points.

**Do not port the ratio between backends.** Vulkan peaks at 75/25 (29.99) and
measures only 25.63 at 60/40. ROCm peaks at 58/42. Different backends, different
optima, same hardware.

### 2. `GGML_CUDA_DISABLE_GRAPHS=1`. +1.2% to +3.0% decode, free.

Undocumented for this purpose. Positive at every ratio and on both quants:

| config | graphs ON | graphs OFF |
|---|---|---|
| Q6 ts 54/46 | 31.86 | 32.75 |
| Q6 ts 58/42 | 32.78 | **33.24** |
| Q6 ts 62/38 | 31.87 | 32.25 |
| Q8 ts 58/42 | 28.49 | **29.35** |

Prefill is unaffected (428.46 vs 429.31), so there is no trade.

This is consistent with the tensor-parallelism author's own suspicion in
[PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378): *"issues with CUDA
graph overhead due to the large number of subgraphs that the meta backend is
creating"*. It was never quantified there. It costs **decode**, not prefill.

### 3. PR #25051 for Vulkan tensor mode. 2.91x prefill.

If you use `-sm tensor` on Vulkan, you want
[PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051) (pwilkin), which
adds a cross-device allreduce with a CPU proxy.

Matched settings (`-ub 8192 -p 8192`, tensor 50/50, Q6_K):

| build | pp8192 |
|---|---|
| stock Vulkan `936918514` | 312.68 |
| **PR #25051 `13ac90156`** | **910.13** |
| ROCm `030ebb558` | 438.19 |

**The control that makes this trustworthy:** in `-sm layer` the same two builds
are identical (838.31 vs 836.62, 0.2% apart). So the effect is the tensor path
specifically, not a build-version artifact.

Output verified byte-identical to a ROCm layer-split reference under greedy
decode, on a hybrid SSM model that upstream lists as unsupported for tensor mode.

---

## PREFILL: the ranking inverts with context depth

The most important prefill finding, and the one a standard benchmark hides.
Everyone reports `pp` at empty KV. That is the depth where tensor mode looks
worst and layer/row look best. Marginal prefill rate (t/s) at increasing depth:

| config | d0 | d32k | d65k | d98k | decay |
|---|---:|---:|---:|---:|---:|
| VK Q8 **tensor** 50/50 | 959.01 | 744.17 | 587.29 | **494.97** | −48% |
| VK Q6 **tensor** 50/50 | 900.53 | 708.65 | 562.65 | **477.30** | −47% |
| VK Q8 row | **1312.66** | 756.94 | 516.61 | 360.47 | −73% |
| VK Q6 layer 28/37 | 1134.56 | 699.51 | 488.55 | 345.27 | −70% |
| ROCm Q8 layer 50/50 | 962.22 | 607.23 | 430.49 | 334.27 | −65% |
| ROCm Q6 layer 50/50 | 995.26 | 593.30 | 423.64 | 314.91 | −68% |
| ROCm Q6 tensor 50/50 | 433.79 | 370.08 | 307.43 | **CRASH** | -- |

At d0, Vulkan row leads and tensor looks mediocre. **At 98k, Vulkan tensor wins
by +48%** over the best alternative, because tensor decays ~47% where layer and
row decay 65-73%. The crossover is around 40-50k.

If you serve long prompts, benchmark at YOUR depth. A d0 number will send you to
the wrong split mode.

Note the last row: ROCm tensor cannot reach 98k depth at `-ub 1024` at all --
`HSA exception: MemoryRegion::BlockAllocator::alloc failed` then
`ggml-backend-meta.cpp:1576`. Vulkan tensor completes the same test.

## MTP / speculative decoding

### Do not use MTP throughput to compare configurations

At `temp 1.0`, across 36 samples of 4 identical configs:
**correlation(acceptance, tg) = 0.918**, acceptance ranged **0.10 to 0.90**, and
decode ranged **26.02 to 67.62 t/s on the same config**. A 3-sample comparison
concluded `n_max 2` beat `n_max 3`; 9 samples reversed it; neither was
significant.

**The fix is `temp 0`, not more samples.** At temp 0 the continuation is
deterministic and acceptance stops being a random variable. Repeat spread
collapsed from sd 8.2 to ~0.1 t/s:

    n3 p0.0  code   59.31 / 59.30 / 59.51

Then measure each content type separately, because content dominates config:

| content | tg (same config) |
|---|---:|
| code | 59.4 |
| technical | 51.4 |
| prose | 46.8 |

### `--spec-draft-p-min` (discussion #25198): does NOT reproduce here

[Discussion #25198](https://github.com/ggml-org/llama.cpp/discussions/25198)
reports `--spec-draft-n-max 16 --spec-draft-p-min 0.8` giving +20% on an
RTX 5090 with Gemma-4-31B and Qwen3.6-27B. Tested here on Qwen3.8-27B Q6_K,
`-sm layer`, temp 0, 3 content types x 2 reps:

| config | prose | code | tech | acceptance |
|---|---:|---:|---:|---:|
| **n3 p0.0 (default)** | **31.01** | 38.19 | **33.57** | 0.596 |
| n8 p0.8 | 28.55 | **39.03** | 33.12 | **0.803** |
| n16 p0.8 (#25198) | 27.26 | 35.63 | 31.48 | 0.755 |
| n16 p0.6 | 27.43 | 30.80 | 28.45 | 0.537 |

n16 p0.8 is **-12.1% prose / -6.7% code / -6.2% tech**. No clamp warning was
emitted, so `n_max` genuinely ran at 16.

**Their mechanism works; the trade does not pay here.** `p_min` raises acceptance
exactly as described (0.596 -> 0.803 at n8 p0.8, the highest measured). But
drafting 8-16 tokens per step costs more than the longer accepted runs return on
this model. Different model and different hardware, opposite conclusion. If your
draft head is stronger, retest -- the knob is real, the result is not universal.

### BUG: `--spec-draft-n-max > 8` crashes under `-sm tensor`

Reproducible on this box, Qwen3.8-27B Q6_K, ROCm:

    -sm tensor --spec-draft-n-max 10  -> ggml-backend-meta.cpp:1576:
                                         GGML_ASSERT(meta_buf_ctx->bufs[i]) failed
    -sm tensor --spec-draft-n-max 12  -> same
    -sm tensor --spec-draft-n-max 16  -> same
    -sm tensor --spec-draft-n-max 8   -> works
    -sm layer  --spec-draft-n-max 16  -> works

Not memory pressure (XTX was at 12 GB of 24), not a model limit (layer split runs
the identical config). The meta backend mis-sizes its buffer array for larger
draft graphs. Same component as
[PR #24549](https://github.com/ggml-org/llama.cpp/pull/24549) and
[PR #24566](https://github.com/ggml-org/llama.cpp/pull/24566), which fix a
related MTP-plus-tensor crash. This means #25198's configuration cannot be
evaluated under tensor split at all on this build.

## WHAT DID NOT WORK

Roughly 45 knobs. All measured, all null or negative.

### Environment variables (ROCm/HIP)

| knob | result |
|---|---|
| `GGML_CUDA_PDL=1` | 32.80 vs 32.78 baseline. Nothing. |
| `GGML_CUDA_GRAPH_OPT=1` | 32.72. Nothing. |
| `GGML_CUDA_REGISTER_HOST=1` | 32.82. Nothing. |
| `GGML_CUDA_NO_PINNED=1` | 31.99. **−2.4%, harmful.** |
| `GGML_CUDA_DISABLE_FUSION=1` | 31.04. **−5.3%, harmful.** |
| `GGML_CUDA_ALLREDUCE=internal` vs `none` | 430.30 vs 429.57. Identical. |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` 512KB→16MB | all ~429 pp. Nothing across a 32x range. |
| `GGML_CUDA_AR_COPY_THRESHOLD` 1MB→64MB | 429.15. Nothing. |
| `GGML_CUDA_AR_BF16_THRESHOLD=0` | 428.85 vs 429.32. **Doubling wire bytes changes nothing.** |

### Environment variables (Vulkan)

| knob | result |
|---|---|
| `GGML_VK_COMM_TREE=1` | 29.93 vs 29.84. Nothing (this is the PR's own opt-in tree allreduce). |
| `GGML_VK_COMM_PROXY=0` or `=1` | 28.58. **Harmful either way.** |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 29.98. Nothing. |
| `GGML_VK_DISABLE_MMVQ` / `FORCE_MMVQ` | 29.87 / 28.97. Nothing / harmful. |
| `GGML_VK_DISABLE_COOPMAT2`, `_DECODE_VECTOR` | 29.88 / 29.96. Nothing. |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 29.97. Nothing. |
| `GGML_VK_MAX_NODES_PER_SUBMIT` 50 / 1000 | 29.76 / 29.79. Nothing. |
| `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` | 29.80. Nothing. |
| **`GGML_VK_DISABLE_ASYNC=1`** | **12.56. Catastrophic, −58%.** Never set this. |

### Flags

| knob | result |
|---|---|
| `-t` 1 / 2 / 4 / 8 / 16 / 32 | 32.68–32.81. **Nothing across a 32x range.** |
| `-ub` 64 / 128 / 256 / 512 / 1024 | 32.67–32.79. Nothing (ubatch is a prefill knob). |
| Quantized KV (`q8_0`) for speed | 31.74 vs 32.67 ROCm, 29.28 vs 29.84 Vulkan. **Slightly harmful.** Use it for context capacity, never for speed. |

### System

| knob | result |
|---|---|
| **PCIe ASPM → `performance`** | **Null.** 33.05 vs 33.24. A single-GPU control config (which cannot be affected by PCIe settings) drifted by the same −0.3%, identifying it as session drift. Reported as +10.8% in [discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043); did not reproduce here. |
| **RCCL** (`-DGGML_HIP_RCCL=ON`) | **Segfaults.** See below. |

---

## RCCL on mainline Linux: does not work

`librccl1` 7.1.0-0ubuntu2 (RCCL 2.72.5) is installed and links correctly. It
**cannot complete a two-GPU AllReduce on this machine**, and llama.cpp is not
implicated:

```
UT_MIN_GPUS=2 /usr/libexec/rocm/librccl1-tests/rccl-UnitTests --gtest_filter='AllReduce.*'
[ RUN      ] AllReduce.OutOfPlace
[ ERROR    ] Child 0 pipe closed unexpectedly
```

Single-GPU AllReduce passes. Two GPUs dies on the first collective. That is AMD's
own test suite.

- With P2P: RCCL misdetects it as available (`isAllDirectP2p 1`) even though the
  cards are on different root ports, connects, then segfaults.
- With `NCCL_P2P_DISABLE=1`: transport genuinely switches to `via SHM/direct`,
  and it **still segfaults**. So this is not a P2P-path bug.

This matches IMbackK in [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378):
RCCL *"doesn't really work on mainline linux - only on amd's fork"*. This box runs
Ubuntu's ROCm packaging plus the in-tree amdgpu driver, i.e. the mainline case.

**It would not have helped anyway.** With no P2P route, a working RCCL transports
over host memory -- the same path llama.cpp's internal allreduce already uses.
The two existing implementations already measure identically.

---

## Why tensor split is slow on AMD (prefill)

ROCm `-sm tensor` prefill is ~429 t/s. ROCm `-sm layer` is ~1195-1229 on the same
build and the same link. Tensor mode costs ~3x prefill.

**It is not interconnect bandwidth.** Five independent knobs say so:

- Doubling the wire bytes (`GGML_CUDA_AR_BF16_THRESHOLD=0`): no change.
- 32x change in transfer chunk size: no change.
- 16x change in ubatch (so 16x fewer allreduce calls): no change.
- Swapping the allreduce algorithm entirely: no change.
- Layer split moves data over the identical link and is 3x faster.

The internal allreduce is a **GPU kernel that spin-waits on pinned host memory**
(`ggml/src/ggml-cuda/allreduce.cu`: *"staging it through pinned host memory over
PCIe"*). Tensor mode needs ~2 allreduces per layer x 65 layers per forward pass,
each a hard barrier where both GPUs wait. The cost is synchronization count, not
payload size.

Decode is unaffected because its activations are ~10 KB instead of ~10.5 MB,
which is why ROCm tensor **decode** is excellent while its prefill is not.

Upstream acknowledges the shape without diagnosing it. From #19378, the author:
*"The 'ROCm' backend works since it is just the CUDA code translated via HIP. On
the hardware combinations that I have (RX 6800 + MI50 or RX 9060 XT + MI100) the
performance is bad vs. the `-sm layer` baseline though."*
And: *"only the CUDA backend has received the necessary extensions and performance
optimizations to make the performance better than `--split-mode layer`."*

---

## How to measure this stuff without fooling yourself

Hard-won, each of these produced a wrong conclusion first.

**1. Do not rank configs on MTP / speculative decode throughput.**
Draft acceptance is content-dependent and swamps everything. Across 36 samples
of 4 identical configs here, **correlation(acceptance, tg) = 0.918**, acceptance
ranged **0.10 to 0.90**, and decode ranged 26.02 to 67.62 t/s *on the same
config*. A 3-sample comparison said `n_max 2` beat `n_max 3`; 9 samples reversed
it. Neither was significant. Prose and code differ by ~2x on identical settings.
Rank on **raw** decode (`llama-bench -p 0 -n 512`), then measure MTP once on the
config you already chose.

**2. Measure VRAM as server-only, not whole-card.**
If a GPU drives your desktop, `mem_info_vram_used` includes it, and the desktop
is not a constant -- it moved 1,696 → 3,000 MiB here within one day. The same
config measured 23,092 / 24,034 / 24,282 MiB on three runs and looked
"unstable". It wasn't. Rule: `server_only + worst_case_desktop <= ceiling`, where
`server_only` = peak under load minus a baseline sampled immediately before launch.

**3. Throughput collapses before you hit an OOM.**
At 24,180 MiB this card silently spilled to GTT: prefill fell 79% (671 → 140
t/s), decode 38%. No crash, no failed health check, correct output throughout.
A config that "got slower" may be out of memory. Always read peak VRAM next to
the speed number.

**4. Tag every number with its context depth.**
`pp8192` at empty KV and a marginal rate at 65k depth are different quantities.
Comparing them across configs manufactures a 3x difference out of nothing.

**5. Isolate your environment per build.**
Setting `LD_LIBRARY_PATH` for one build and then running a different binary makes
it load the wrong `libggml`. This silently produced a full round of wrong numbers
here. Use `env -u LD_LIBRARY_PATH` for anything that doesn't need it.

**6. Watch the separators.** `-ts` uses `/`. `-dev` uses `/` to mean "these
devices together" -- a **comma** makes llama-bench benchmark each GPU
*separately* and report single-GPU numbers that look like dual-GPU ones.

**7. llama-bench has no `-c` flag.** Context comes from `-p + -n + -d`.

---

## Sources

- [PR #19378 — ggml: backend-agnostic tensor parallelism](https://github.com/ggml-org/llama.cpp/pull/19378) — the meta backend; the author's own AMD performance caveat
- [PR #25051 — vulkan: allreduce with cross-device CPU proxy](https://github.com/ggml-org/llama.cpp/pull/25051) — the 2.91x Vulkan tensor result
- [Discussion #21043 — RDNA4 Llama Experiments, R9700](https://github.com/ggml-org/llama.cpp/discussions/21043) — ASPM, `rm_kq`, graphics queue, RADV vs AMDVLK
- [Discussion #25198 — Pushing MTP to the limit with --spec-draft-p-min](https://github.com/ggml-org/llama.cpp/discussions/25198) — `n_max 16 + p_min 0.8`
- [Issue #20934 — ROCm vs Vulkan token generation on 7900 XTX](https://github.com/ggml-org/llama.cpp/issues/20934)
- [PR #24549 — disable graph reuse under SPLIT_MODE_TENSOR](https://github.com/ggml-org/llama.cpp/pull/24549) — fixes an MTP+tensor crash, confirmed on 2x 7900 XTX
- [PR #24566 — meta: key external view cache by backend context](https://github.com/ggml-org/llama.cpp/pull/24566) — alternative fix for the same crash
- [PR #25319 — ggml: asynchronous scheduler memory copies](https://github.com/ggml-org/llama.cpp/pull/25319)
- [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) — states plainly that only CUDA is expected to perform well
