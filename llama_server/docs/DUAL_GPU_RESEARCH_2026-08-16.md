# Dual-GPU llama.cpp speedups — external research

**Date:** 2026-08-16
**Scope:** what other people have measured for multi-GPU llama.cpp, filtered to
things that could apply to this box (RX 7900 XTX gfx1100 24 GiB + AI PRO R9700
gfx1201 32 GiB, ROCm + Vulkan, llama.cpp b10358 / `030ebb558`).

Research only. Nothing here has been tested on this machine yet. Every claim is
attributed; where a source contradicts what we measured locally, that is called
out explicitly.

---

## 0. Local baseline captured for this document

Read-only, taken 2026-08-16. Used to decide which findings are already satisfied.

| Thing | Value here | Notes |
|---|---|---|
| llama.cpp | `030ebb558` (b10358) | pinned |
| `GGML_SCHED_MAX_COPIES` | **4** (default) | both `build/` and `build-vulkan/` |
| `GGML_HIP_RCCL` | **OFF** | `librccl.so.1` **is** installed system-wide |
| `GGML_HIP_ROCWMMA_FATTN` | OFF | deliberately — see §6 |
| `GGML_HIP_GRAPHS` | ON | |
| `GGML_HIP_MMQ_MFMA` | ON | |
| `GGML_HIP_NO_VMM` | ON | |
| `GGML_LTO` | **OFF** | |
| `GGML_NATIVE` | ON | |
| PCIe ASPM policy | ~~`[default]`~~ → **`[performance]`** | **changed 2026-08-16**, see §4.1. Runtime only — reverts on reboot |
| `amdgpu.runpm` | **-1** (auto) | not `0` |
| `amdgpu.pcie_p2p` | **Y** | driver-side P2P already permitted |
| `power_dpm_force_performance_level` | `auto` on both cards | already the recommended value |
| Mesa | 26.0.3 | ≥25.2, so `nogttspill` is available |
| PCIe link, R9700 (card0) | 32.0 GT/s **x16** (Gen5) | full width |
| PCIe link, XTX (card1) | 16.0 GT/s **x16** (Gen4) | full width, not bifurcated |
| kernel cmdline | `amdgpu.ppfeaturemask=0xfff7ffff` | no ASPM/runpm params set |

Two structural facts that shape everything below:

1. **Both cards are on full x16 links.** Interconnect is as good as this platform
   gets, so interconnect-bound optimisations are worth trying rather than being
   written off up front.
2. **The cards are different architectures** (gfx1100 vs gfx1201). Every
   multi-GPU result cited below was measured on *matched pairs*. Nothing here is
   guaranteed to transfer.

---

## 0.5 Verified on this machine — three findings that change existing profiles

Added 2026-08-16 after cross-checking
[`DUAL_GPU_OPTIMIZATION_REPORT_2026-08-16.md`](./DUAL_GPU_OPTIMIZATION_REPORT_2026-08-16.md)
(written by a second agent). These are read out of our own server log and source
tree, not from any external source. See §9 for what that report got wrong.

### 0.5.1 `--cache-reuse` has never worked here. Every profile. Every model.

`~/.local/state/local-llm/llama-server.log`, across 565 server starts:

```
53 × "cache_reuse is not supported by multimodal, it will be disabled"
63 × "cache_reuse is not supported by this context, it will be disabled"
```

The two reasons are separate guards in `tools/server/server-context.cpp`:

| Line | Guard | Hits |
|---|---|---|
| 1281 | `mmproj_path` is set | every `--mmproj` profile |
| 1293 | `!llama_memory_can_shift(llama_get_memory(ctx_tgt))` | every hybrid/recurrent model |

The second one is the important one. **All four of our models are hybrid**, so
`llama_memory_can_shift()` is false and `n_cache_reuse` is forced to 0 — with or
without a vision projector. There is no profile in this repo where
`--cache-reuse 256` does anything.

Consequences:

- The **"+3% prefill (2623 → 2702), 0/128/256/512 tested"** recorded in
  `preset-36-35b-a3b-q8-rocm-dual-q8-256k-mmproj.sh` is attributed to a flag the
  server had already disabled. Whatever produced that delta, it was not
  cache-reuse. That measurement is void and should be re-run.
- The same guard disables **`ctx_shift`** on these models, and
  `swa_full` on any model with `llama_model_n_swa(model) == 0`. Worth knowing
  before anyone tries to tune those either.
- `--cache-reuse 256` can be dropped from all profiles as dead config, or kept
  with a comment. It costs nothing but it is misleading.

This does **not** affect `--cache-ram` / `--ctx-checkpoints`, which are the
prompt-cache mechanism and a different code path.

### 0.5.2 Real deep-context numbers, from production traffic

Log line 157146–157148, a genuine user request, not a benchmark:

```
prompt eval time = 275588.01 ms / 135390 tokens (2.04 ms/tok,  491.28 t/s)
eval time        =  88933.17 ms /   1966 tokens (45.24 ms/tok,   22.11 t/s)
total time       = 364521.17 ms / 137356 tokens
```

Against what the harness reports for the same profile at ~10k depth
(982 pp / 38.5 tg):

| Depth | prefill | decode |
|---|---|---|
| ~10k (harness) | 982 t/s | 38.5 t/s |
| **135k (real)** | **491 t/s** | **22.11 t/s** |

Prefill halves, decode drops 43%. Every ranking decision in
`RESULTS_2026-08-16.md` was made at ~10k depth, and the user's actual sessions
run past 100k. Two things follow:

1. The harness needs a deep-context tier. A config that wins at 10k is not
   established to win at 135k, and MTP acceptance in particular is known to move
   with depth.
2. A 135k prompt costs **4.6 minutes of prefill**. That is the real cost of the
   full-context reprocesses the `--cache-ram 24576` / `--ctx-checkpoints 8` change
   was meant to prevent — and with `--cache-reuse` confirmed inert (§0.5.1), the
   prompt cache is the *only* thing standing between the user and that 4.6
   minutes.

### 0.5.3 GPU sampling is disabled — top-k falls back to host

234 occurrences of:

```
llama_sampler_backend_support: device 'ROCm1' does not have support for op TOP_K
                               needed for sampler 'top-k'
```

`src/llama-sampler.cpp:647-657` probes the sampler graph against the device and
returns false on the first unsupported op, which rejects the **whole** GPU
sampler chain — not just top-k. Sampling runs on the CPU, so logits cross PCIe
every token.

Expected impact is small next to a 25–45 ms/token decode step, so this is a
"measure it, don't assume it" item rather than a shortlist entry. But note the
device named is **ROCm1**, which is `--main-gpu 1` — the R9700. Whether gfx1100
supports `TOP_K` where gfx1201 does not is untested, and it folds neatly into the
`--main-gpu 0 vs 1` experiment already queued in §3.6.

---

## 0.6 MEASURED 2026-08-16 — tensor parallel tested end to end. Verdict: reject.

All on this box, Qwen3.8-27B Q8_0 (`arch = qwen35moe`), f16 KV, b10358, server
stopped, ASPM already set to `performance`.

### Headline

| config | pp | tg |
|---|---|---|
| **layer, ts 28,37, b6144, MTP (shipped)** | **982** | 38.5 (5-seed median) |
| tensor, ts 35,65, ub2048, MTP, 256k | 435 | **40.88** (5-seed median) |

**+6.2% decode for −56% prefill.** Not worth shipping.

### llama-bench, no speculation, same build, only `-sm` differs

| mode | pp512 | pp2048 | tg128 |
|---|---|---|---|
| layer (dual) | 719.59 | 1047.23 | 20.03 |
| tensor (dual) | 437.66 | 434.67 | **26.56** |
| none (R9700 alone) | 697.67 | — | 19.23 |

### Why tensor mode looked good and then wasn't

| | tensor | layer | tensor advantage |
|---|---|---|---|
| without MTP | 26.56 | 20.03 | **+32.6%** |
| with MTP | 40.88 | 38.5 | **+6.2%** |

Tensor parallelism minimises *single-token* latency. MTP turns decode into a
verify batch of `1+n_draft` tokens — prefill-shaped work, which is precisely
where tensor mode is 2.4× slower. **The two optimisations compete for the same
win**, and MTP already captures most of it on layer split. Any future evaluation
of tensor mode on a speculating model has to account for this; benchmarking it
without MTP overstates the gain by ~5×.

### Prefill in tensor mode cannot be tuned

| lever | values tried | result |
|---|---|---|
| `--batch-size` | 2048 / 6144 / 8192 / 16384 | **no-op** — 410–415 t/s. The +12.3% it buys on layer split does not transfer |
| `--ubatch-size` | 512 / 1024 / 2048 | **no-op** — 403–415 t/s |
| `--tensor-split` | 1:1 / 40,60 / 35,65 | +8% — 403 → 425 → 438 |
| `GGML_CUDA_P2P=1` | on/off, both modes | **no-op**, see below |

### `-ts` is honoured in tensor mode — and it is what makes 256k fit

Source: `src/llama-model.cpp:680-699`. Even splitting is only the fallback when
`tensor_split` is null or all-zero; otherwise each tensor is sliced by the given
proportions.

| split | ctx | peak XTX |
|---|---|---|
| 1:1 (default) | 262144 | **24274** — over the 24200 ceiling |
| `40,60` | 131072 | 17630 |
| `35,65` | 262144 | **20492** |

At 1:1 the XTX holds half a 27 GiB model plus half the KV on a card that also
drives the desktop. Biasing to `35,65` reclaims 3.8 GiB. Killed the 1:1/256k run
at 24274; no desktop freeze occurred.

### `-DGGML_HIP_RCCL=ON` — rejected, it crashes

Built into `build-rccl/` (verified: `librccl.so.1` linked, absent from `build/`;
same commit `030ebb558`, single variable).

| build | `-sm layer` | `-sm tensor` |
|---|---|---|
| `build/` (no RCCL) | works | **works** |
| `build-rccl/` | works | **SEGFAULT at init** |

Not an OOM — peak VRAM 15 GiB when it died. Upstream's *"disabled by default
because it was not universally beneficial"* is an understatement for mixed
RDNA3/RDNA4: it is unusable.

### `GGML_CUDA_P2P=1` — applies to the ROCm build, and does nothing

The docs present this as CUDA-only, but `ggml/src/ggml-cuda/ggml-cuda.cu:392` and
`:607` read the env var and that file is what HIP compiles. So it is live here.
It is also inert:

| mode | P2P off | P2P on |
|---|---|---|
| tensor pp2048 | 434.67 | 434.76 |
| tensor tg128 | 26.56 | 26.43 |
| layer pp2048 | 1047.23 | 1047.24 |
| layer tg128 | 20.03 | 20.03 |

Layer identical to two decimals — the path is not being exercised.

### Single vs dual: the second GPU earns its place on prefill, not decode

| | R9700 alone (`-sm none`) | both cards, layer split | delta |
|---|---|---|---|
| pp512 | 699.71 | 715.40 | +2.2% |
| pp2048 | 688.18 | 1048.59 | **+52.4%** |
| pp8192 | 641.95 | 1122.13 | **+74.8%** |
| tg128 | 19.24 | 20.17 | +4.8% |

**Pipeline parallelism is enabled and working.** Verified directly:
`llama_context: pipeline parallelism enabled` appears when the server is run at
`-lv 10`. Prefill scales +75% at pp8192.

Decode gains only +4.8%, and that is **by design, not a defect**. Pipeline
parallelism runs different layers on different GPUs and passes tokens through
sequentially; with a single token in flight one GPU computes while the other
waits. It parallelises batches, not single-token decode. Tensor parallelism is
the only mode that parallelises decode — and MTP already captures most of that
win on layer split, which is exactly why tensor mode measured only +6.2% here.

**Conclusion for decode:** on this hardware, with a 27 GiB dense model at Q8_0
split over PCIe, ~38.5 t/s with MTP is close to the achievable ceiling. The
levers that would move it are a smaller quant (so the model fits on the
higher-bandwidth XTX alone) or a faster single card. No flag or split ratio
gets there.

> **RETRACTIONS — three claims made earlier in this session were wrong.**
>
> 1. *"Tensor mode is +25–40% decode."* That came from single-seed runs. Five
>    seeds gave a median of 40.88 vs 38.5 = **+6.2%**. Single-seed MTP decode
>    swings ±20% on this box; that is documented in the profile headers in this
>    same repo and I ignored it.
> 2. *"Pipeline parallelism never engages."* False. It is enabled. The claim came
>    from grepping a production log running at verbosity 3, where
>    `pipeline parallelism enabled` is an INFO line and is filtered out. The
>    supporting `42/42 layers` figure belonged to a different model — Qwen3.8 is
>    `n_layer = 64`, `offloaded 66/66`.
> 3. *"The second GPU is functioning almost entirely as VRAM."* False, and it was
>    generalised from pp512 — the one batch size too small for pipeline
>    parallelism to do anything. At pp8192 the second card is worth +74.8%.
>
> Common failure mode in all three: a confident conclusion from one unchecked
> observation. Rank on ≥5 seeds, verify log absence against log level, and pick
> the batch size that can show the effect before concluding it is absent.

### `GGML_SCHED_MAX_COPIES=1` — measured, it is a trade not a win

Built into `build-copies1/` (single variable vs `build/`, verified in cache).

Speed, layer split, `llama-bench`:

| | pp512 | pp2048 | tg128 |
|---|---|---|---|
| `MAX_COPIES=4` | 731.34 | **1065.32** | **20.61** |
| `MAX_COPIES=1` | 725.42 | 1039.64 | 19.94 |

VRAM, at the real production config (262144, ts 28,37, MTP, b6144):

| | XTX | R9700 |
|---|---|---|
| `MAX_COPIES=4` | 21717 | 28557 |
| `MAX_COPIES=1` | **20914** | **26844** |
| freed | **803 MiB** | **1713 MiB** |

**−2.4% prefill / −3.3% decode, for 2.5 GiB.** The 40 → 60 t/s from
[#13751](https://github.com/ggml-org/llama.cpp/issues/13751) does not reproduce;
that report was from a system where pipeline parallelism was active *and*
hurting. Here it is active and helping, so cutting the copies costs performance.

Worth taking only if 803 MiB on the XTX is more valuable than 3% — which is a
real question on this box, since the 24,200 MiB ceiling has constrained every
tensor-split decision in this repo.

---

## 1. Ranked shortlist

Ordered by (expected gain) × (probability it applies here) ÷ (effort + risk).

| # | Change | Claimed effect | Type | Risk |
|---|---|---|---|---|
| 1 | `-DGGML_SCHED_MAX_COPIES=1` | **+50% decode** on 2-GPU layer split; **−0.5 to −1.5 GiB VRAM per card** | rebuild | low |
| 2 | `--parallel 3` + `--spec-draft-n-max 7` + `-ub 288` (Qwen3.8) | **64.95 t/s** vs our 40.4 | flags | none |
| 3 | `-sm tensor` + `-DGGML_HIP_RCCL=ON` | **+56% decode vs single GPU** on dense 27B Q8 | rebuild + flags | medium — **prerequisite confirmed, see §4.7** |
| 4 | PCIe ASPM → `performance` | **+10.8% decode**, dense | system | needs approval |
| 5 | `RADV_PERFTEST=nogttspill` (+`aco,cswave32`) | fixes random GTT-spill slowdowns; ~10% on one dual-R9700 report | env var | none |

Everything else is in §2–§6.

---

## 2. Build-time changes

### 2.1 `GGML_SCHED_MAX_COPIES=1` — the biggest single item

llama.cpp's pipeline parallelism (which is what `--split-mode layer` *is*)
allocates `GGML_SCHED_MAX_COPIES` copies of the compute buffer, default 4.

**VRAM:** measured at ~1022 MB → ~242 MB on GPU1 and ~910 MB → ~242 MB on GPU2,
with **inference speed "virtually identical"** (362 t/s prefill, 17.2 t/s decode
in all three configurations tested). Context capacity went from ~88,576 to
~113,920 tokens on the same hardware.
([prismix.dev writeup](https://prismix.dev/news/b0e27b2381c2),
[mirror](https://bittide.aicompass.dev/article/5666a8e4-c0a3-4ff9-9c70-c9138c19403c))

**Speed:** a separate report on 2× RTX PRO 6000 found layer split at
**~40 t/s** (tg128) with the default 4 copies versus **~61 t/s** with `-sm none`
— and setting `GGML_SCHED_MAX_COPIES=1` closed the gap entirely (**~60 t/s**).
A 235B model went from 25 → 60 t/s. Affected Qwen3, Qwen3MoE, Llama and Gemma3.
([llama.cpp#13751](https://github.com/ggml-org/llama.cpp/issues/13751))

Why this is the top item for *this* box specifically:

- We have a hard 24,200 MiB ceiling on the XTX because it drives the desktop, and
  the whole tensor-split tuning exercise has been fighting that ceiling. ~0.8 GiB
  back on the XTX is roughly what moving 2 layers to the R9700 bought us.
- Our shipped Qwen3.8 decode (38.5 t/s) is dense-model-slow in a way that looks a
  lot like the #13751 symptom.
- It is free at runtime: a build flag, no behaviour change.

Caveat: pipeline parallelism *does* help prefill on systems where it works
properly (see §2.2), so this must be measured on both pp and tg, not just tg.
One counter-datapoint: on A40s, PP scales close to linearly (1462 → 2546 → 3530
t/s for 1/2/3 GPUs on pp8192).
([llama.cpp#20252](https://github.com/ggml-org/llama.cpp/discussions/20252))

### 2.2 Pipeline parallelism can silently be off

PP disables itself without logging when there is insufficient VRAM for the
compute buffer, when RPC devices are active, when GPU-to-GPU P2P is unavailable,
or when resizable BAR / modern PCIe features are missing. Diagnosis: the verbose
log line `pipeline parallelism enabled (n_copies=4)`. A "sawtooth" utilisation
pattern where only one GPU is busy at a time means PP is not working.
`--override-tensor` also disables it.
([#20252](https://github.com/ggml-org/llama.cpp/discussions/20252),
[#13751](https://github.com/ggml-org/llama.cpp/issues/13751))

**Action:** grep our server logs for that line before doing anything else — if PP
was never on, item 2.1 is free VRAM with zero downside, and if it *is* on, the
speed comparison is the real experiment.

### 2.3 `-DGGML_HIP_RCCL=ON`

The AMD equivalent of NCCL for cross-GPU reductions in `--split-mode tensor`.
Upstream docs: *"RCCL is by default **disabled** because (unlike NCCL) it was not
universally beneficial during testing."*
([docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md))

`librccl.so.1` is already installed here, so this is a rebuild flag and nothing
else. Only matters in `-sm tensor`. A dual-R9700 user runs
`-DGGML_HIP=ON -DGGML_HIP_RCCL=ON` with RCCL 2.27.7 in production.
([#23567](https://github.com/ggml-org/llama.cpp/issues/23567))

### 2.4 `-DGGML_LTO=ON`

Currently OFF here. Used in the R9700 tuning discussion's reference build line:
`cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DGGML_LTO=ON`.
No isolated measurement of its contribution was published.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

### 2.5 Vulkan shader compiler version

Building the Vulkan backend with an old system `glslc` (2023.8) causes a **30%+
regression on Qwen3.8**. The fix is to point cmake at a modern SDK:
`-DVulkan_GLSLC_EXECUTABLE=$HOME/opt/1.4.341.1/x86_64/bin/glslc` (SDK 1.4.357.1
cited as the correct baseline).
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

**Action:** check which `glslc` our `build-vulkan/` used. This is a
30%-shaped hole that would be invisible in any A/B we ran, because every Vulkan
number we have came from the same build.

### 2.6 `rm_kq=1` — one-line source patch

Change `uint32_t rm_kq = 1;` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. Reduces
VGPR pressure and improves occupancy. Measured: +1% RADV MoE decode, +2% AMDVLK
MoE, **+13% AMDVLK dense decode**.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

Measured on RDNA4 (gfx1201). Unknown on RDNA3.

### 2.7 The `-funsafe-math-optimizations` patch (already on our list)

We measured b10450 as −4.8% pp / −4.0% tg vs b10358 and attributed it to
[#26696](https://github.com/ggml-org/llama.cpp/pull/26696) removing
`-funsafe-math-optimizations` from the HIP build. Restoring it as a local
one-line patch would let us take newer server fixes at no cost. Still untested.

---

## 3. Runtime flags

### 3.1 The Qwen3.8-27B Q8_0 config that hit 64.95 t/s

This is the headline number, on a **single R9700**, Vulkan:

```
llama-server --model qwen38-q8.gguf --device Vulkan0 \
  --n-gpu-layers 99 --spec-draft-ngl all --parallel 3 \
  --batch-size 2048 --ubatch-size 288 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 7
```

Reported: 27B Q8_0 **59.2 → 64.95 t/s** with MTP at `spec-draft-n-max 7`
(+10%), acceptance 95–97%. The 27B Q4_K_M went 28.2 → 47.4 t/s (+68%) at
n-max 3. `parallel=3` was optimal; 1, 2, 4 and 8 all underperformed.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

Every one of those contradicts a conclusion we shipped:

| Parameter | Ours | Theirs |
|---|---|---|
| `--parallel` | 1 | **3** |
| `--spec-draft-n-max` | 3 | **7** |
| `--ubatch-size` | 512 | **288** |
| `--batch-size` | 6144 | 2048 |
| KV type | f16 | q4_0 |
| GPUs | 2 | **1** |

Three honest caveats before treating 64.95 as a target:

1. **It is a single 32 GiB card**, not a split. Our own XTX-only Vulkan test on
   Qwen3.6-27B Q4_K_M hit 70.4 t/s — 2× the dual-GPU number — so "single card is
   much faster for decode" is consistent with what we already found.
2. **`--parallel 3` with `--kv-unified` may silently disable speculative
   decoding.** There is an open bug where llama-server ignores `-md` when the
   unified KV cache is active, and `--no-kv-unified` does not appear to take
   effect in the log.
   ([#25345](https://github.com/ggml-org/llama.cpp/issues/25345)) If a
   `parallel 3` run shows 0% acceptance, that is the reason.
3. `--parallel 3` triples slot KV allocation. On a hybrid model that also
   inflates recurrent state — which is exactly why we set `--parallel 1` in the
   first place. This needs the memory predictor run before launching.

Our own acceptance numbers (42–72%) are far below their 95–97%, which suggests
either a different quant, a different prompt distribution, or that the MTP head
behaves differently on gfx1201.

### 3.2 `-ub 2048 -b 16384` for prefill

**+29% prefill (pp2048) on a 35B MoE**, +3% on the 27B dense.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

Our 35B profile is at `-ub 1024 -b 12288`, and our own sweep showed the ubatch
curve still climbing at 1024 (256=1642, 512=2138, 1024=2576). **2048 was never
tested.** This is the most obviously untested point on a curve we already know
is monotonic.

### 3.3 The ubatch danger zone on hybrid models

**`-ub` values between 65 and 256 cause a 40× throughput collapse on
Qwen3.5/3.6-class hybrid models.** Safe values: 64, 288, 512, 2048.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

Relevant because 288 is the value in the 64.95 t/s config — it sits just above
the cliff. Do not sweep ubatch continuously on these models.

### 3.4 `--split-mode tensor`

Upstream describes it as: *"EXPERIMENTAL. Tensor parallelism that splits both
weights and KV across the participating GPUs via a 'meta device' abstraction…
pipeline-parallel maximizes batch throughput; tensor-parallel minimizes
latency."* Hard requirements: `-fa on`, non-quantized KV (`f32`/`f16`/`bf16`),
and `--fit` is unsupported.
([docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md))

We already satisfy all three: f16 KV everywhere, `-fit off` in every profile.

**Which of our models are eligible** — checked against `llm_arch_supports_sm_tensor()`
in our own tree (`src/llama-arch.cpp:1009`):

| Model | `-sm tensor` |
|---|---|
| Qwen3.8-27B | allowed |
| Qwen3.6-35B-A3B / 27B | allowed |
| Muse Glimmer 30B | allowed |
| **Nemotron 3.5 30B-A3B** | **blocked** (`LLM_ARCH_NEMOTRON_H_MOE` is on the deny list) |

The deny list is a `switch` returning `false` for 27 named architectures and
`true` by default. **No Qwen architecture appears on it.** The running model
logs `general.architecture = qwen35moe` / `arch = qwen35moe`, which is not in the
list, so it takes the `default: return true` branch.

This is worth stating explicitly because being *hybrid* is not itself
disqualifying — the deny list names specific architectures, and several hybrid
Qwen variants are simply absent from it. Reasoning from "this model is
hybrid/recurrent, therefore tensor split is unimplemented" gives the wrong
answer here; the code is the authority.

Warning from the same doc: *"Performance should be good for multiple NVIDIA GPUs
using the CUDA backend, no guarantees otherwise."* And a general note that `row`
is deprecated and superseded by `tensor`.

Measured dual-R9700 `-sm` results, same discussion:

| Workload | Single | Dual | Δ |
|---|---|---|---|
| 35B MoE pp2048 | 3798 | 5224 | **+36.6%** |
| 35B MoE tg128 | 155.6 | 114.7 | **−26.0%** |
| 27B dense pp2048 | 930 | 954 | +1.9% |
| Dense 27B pp65536 | — | — | **+78.9%** |

([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

That −26% decode is the same shape as everything we measured locally, and it is
on a *matched* pair with no desktop load. It is the single best argument that our
dual-GPU decode deficit is inherent, not a misconfiguration.

### 3.5 Quantized KV is a hard error in tensor mode

`"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not
implemented"`. A dual-R9700 user bypassed the guard by patch and hit
`ggml op not implemented: TURBO_WHT` inside the meta-backend. Their working
numbers: tensor split + f16 KV = **29.7 t/s at ~32K/slot**, versus layer split +
turbo4 KV = 19 t/s at 131K/slot.
([#23567](https://github.com/ggml-org/llama.cpp/issues/23567))

Read that as: tensor mode is +56% on decode but costs you most of your context.
For our 256k profiles that trade is probably not worth it; for a small-context
preset it might be.

### 3.6 Flags worth checking that we have never touched

- `--kv-unified` / `--no-kv-unified` — enabled by default when slots are auto;
  interacts badly with speculative decoding per
  [#25345](https://github.com/ggml-org/llama.cpp/issues/25345).
- `-mg / --main-gpu 0 vs 1` on Muse and Nemotron — we only ever tested this on
  Qwen3.8 and inherited the answer.
- `--override-tensor` — note it **disables pipeline parallelism**
  ([#13751](https://github.com/ggml-org/llama.cpp/issues/13751)), which makes it
  a crude way to test §2.1 without rebuilding.

---

## 4. System, driver, and kernel

Everything in this section changes system settings. Per the standing rule on this
box, none of it gets applied without asking first.

### 4.1 PCIe ASPM → performance

**+10.8% decode on dense models (RADV)**, minimal on MoE. Zero impact on Gen3 x8
or eGPU with a 256MB BAR.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

```
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
# persistent: pcie_aspm.policy=performance on the kernel cmdline
```

**APPLIED 2026-08-16.** Verified: `default [performance] powersave
powersupersave`. Both cards remain at full width and speed (R9700 32.0 GT/s x16,
XTX 16.0 GT/s x16) — ASPM governs link power states, not negotiated speed, so no
change there was expected.

Both cards are on full x16 links, so the source's "no effect on Gen3 x8 or a
256MB-BAR eGPU" exclusion does not apply here.

**Not yet persistent.** This reverts on reboot. To keep it, add
`pcie_aspm.policy=performance` to the kernel cmdline — the existing line is:

```
BOOT_IMAGE=/boot/vmlinuz-7.0.0-29-generic root=UUID=… ro quiet splash \
  amdgpu.ppfeaturemask=0xfff7ffff crashkernel=…
```

Hold off on making it permanent until the +10.8% is confirmed on this hardware.
**Unmeasured here so far** — the claimed gain is from a dual-R9700 box on RADV
with a dense model, and it needs a same-harness A/B before it earns a place in
the profiles. Revert with `echo default | sudo tee /sys/module/pcie_aspm/parameters/policy`.

### 4.2 `amdgpu.runpm=0`

The RADV debugging guide attributes GTT spillover — memory landing in GTT instead
of VRAM despite free VRAM, causing severe slowdown — to the GPU being suspended,
and prescribes `amdgpu.runpm=0`.
([#23295](https://github.com/ggml-org/llama.cpp/discussions/23295))

Currently `-1` (auto) here. We have chased GTT/VRAM pressure on the XTX for two
sessions, including a config that started *timing out* when the desktop grew.
Worth knowing this knob exists even though our GTT numbers read 0.00.

### 4.3 `RADV_PERFTEST=nogttspill`

Environment variable, Mesa ≥25.2 (we are on 26.0.3). Forces the driver to move
allocations back to VRAM. Symptom it fixes: *shader clock high, memory clock low,
despite free VRAM* — random, and can appear with plenty of VRAM headroom. The
guide recommends setting it **whether or not you see the symptom**.
([#23295](https://github.com/ggml-org/llama.cpp/discussions/23295))

A dual-R9700 user reports ~10% over baseline Vulkan with:
```
RADV_PERFTEST=aco,cswave32,nogttspill
GGML_VK_VISIBLE_DEVICES=0,1
```
([#22871](https://github.com/ggml-org/llama.cpp/discussions/22871))

This is a pure env var in a profile — no system change, no rebuild. Cheapest
thing on the list.

### 4.4 Healthy-GPU reference numbers

From the same guide, what a correctly-running card looks like: graphics pipe
≥98%, shader and memory clocks at their limits, **GTT ≤2%**. Tools: `radeontop`,
`watch -n 1 sensors`, `vulkaninfo --summary`.
([#23295](https://github.com/ggml-org/llama.cpp/discussions/23295))

Useful because our benchmark harness currently logs VRAM but not clocks — a run
that is slow *because the memory clock never boosted* would look identical to a
run that is slow for configuration reasons.

### 4.5 Power state: `auto` beats `high`

`auto` boosts to **3348 MHz** versus a fixed 2350 MHz under `high`. `high` gives
+8–9% decode but −8–9% prefill.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

Both cards here are already `auto`. **Nothing to do — this one is already
optimal.**

### 4.6 BIOS / kernel checklist from the R9700 work

Recommended: Resizable BAR enabled, PCIe Above 4G Decoding enabled, Global
C-State disabled, kernel ≥6.19.8 (6.17 has MCLK boost issues), and GRUB
`amdgpu.runpm=0 pcie_aspm.policy=performance amdgpu.ras_enable=0`. LACT with a
330W limit and the COMPUTE profile gave +6–7% TG on dual R9700.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

LACT power-limit changes are explicitly out of scope under the standing rule.

### 4.7 P2P on this pair — **RESOLVED 2026-08-16: it works.**

**Measured here, not inferred.** `hipDeviceCanAccessPeer`, query-only, with the
production server running untouched:

```
[0] AMD Radeon RX 7900 XTX   gfx1100  24560 MiB
[1] AMD Radeon AI PRO R9700  gfx1201  32624 MiB

0 -> 1 : YES        1 -> 0 : YES
```

Supporting evidence, all read-only:

| Check | Result |
|---|---|
| KFD `p2p_links` node1↔node2 | present, `type 2` (PCIe), both directions |
| `NO_PEER_TO_PEER_DMA` flag (bit 4, value 16) | **not set** in either direction |
| Resizable BAR, XTX | BAR0 = **32G**, CPU-visible VRAM 24560 / 24560 MiB |
| Resizable BAR, R9700 | BAR0 = **32G**, CPU-visible VRAM 32624 / 32624 MiB |
| `amdgpu.pcie_p2p` | `Y` |
| gfx targets | node1 `110000` (gfx1100), node2 `120001` (gfx1201) |

Full BAR exposure is exactly the precondition AMD's P2P work requires, and it is
satisfied on both cards. This is *not* the dual-7900-XTX situation described in
[ROCm#2253](https://github.com/ROCm/ROCm/issues/2253), where
`rocm-bandwidth-test` reported N/A — that report predates ReBAR being enabled by
default and `amdgpu.pcie_p2p`.

Two asymmetries worth carrying into any tensor-parallel benchmark:

| Direction | `max_bandwidth` | `flags` | Meaning |
|---|---|---|---|
| XTX → R9700 | 64000 | 3 | enabled, non-coherent |
| R9700 → XTX | **8000** | **15** | enabled, non-coherent, **no 32-bit atomics, no 64-bit atomics** |

The 8× bandwidth asymmetry and the missing PCIe atomics on the R9700→XTX leg are
the two things most likely to make tensor mode underperform the matched-pair
results in §5.1. They are also a concrete reason to test `--main-gpu 0` and
`--main-gpu 1` separately under `-sm tensor`: the reduction traffic is not
symmetric, so the direction the collective runs in matters.

**Consequence: the tensor-parallel branch is live, not blocked.** Items §2.3
(`-DGGML_HIP_RCCL=ON`), §3.4 (`-sm tensor`) and §5.1 (direct-P2P AllReduce) all
clear their prerequisite. §5.1 in particular has never been tested on a
*mixed-architecture* pair — its author explicitly asked for exactly that.

Probe source: `scratchpad/bench/p2p_probe.cpp`. `rocm-bandwidth-test` is still
not installed; it is no longer needed to answer the gating question, but it would
give real measured GB/s per direction to check the 64000/8000 asymmetry above.

### 4.7b Historical context — why this was expected to fail

`rocm-bandwidth-test` on dual 7900 XTX shows **N/A for GPU-to-GPU transfers**,
i.e. no P2P path.
([ROCm#2253](https://github.com/ROCm/ROCm/issues/2253),
[Fang-Pen Lin](https://fangpenlin.com/posts/2025/06/11/two-amd-7900xtx-gpus-tinygrad-based-training-workstation-peer-to-peer-pcie-communication/))

It *can* be enabled, but it takes a kernel rebuild with `DMABUF_MOVE_NOTIFY=y`
and `HSA_AMD_P2P=y`; after that a dual-7900XTX tinygrad workload roughly doubled
(29 → 61 it/s). AMD's own P2P work requires the BAR to expose the full VRAM
capacity on the PCIe bus.
([Phoronix](https://www.phoronix.com/news/AMD-Multi-GPU-Compute-P2P))

Locally `amdgpu.pcie_p2p` already reads `Y`, so the driver permits it — but that
is the module parameter, not proof the link works. `rocm-bandwidth-test` is not
installed here; installing it is a package change and would need approval.

**Why this matters:** items 3 in the shortlist (`-sm tensor`) and §5.1 (the P2P
AllReduce patch) both depend on real P2P. If `rocm-bandwidth-test` shows N/A,
tensor mode will fall back to host-staged transfers and most of its benefit
evaporates.

---

## 5. Patches and forks

### 5.1 Direct-P2P HIP AllReduce — the closest match to our hardware

The HIP backend's AllReduce is a **stub**: tensor-parallel mode silently falls
back to a butterfly algorithm that stages through host memory and crosses PCIe
twice. A patch replaces it with a one-shot peer-access path.

Measured on 2× R9700 (gfx1201), ROCm 7.2.4, `llama-bench -sm tensor -fa on -n 128 -r 5`:

| Model | 1 GPU | Stock TP | Direct-P2P | vs single | vs stock |
|---|---|---|---|---|---|
| Bielik-11B Q8_0 | 45.09 | 55.06 | **61.22** | +36% | +11.2% |
| **Qwen3-27B Q8_0** | **19.50** | **28.30** | **30.43** | **+56%** | **+7.5%** |

Perplexity bit-identical to butterfly (6.0464 == 6.0464). Prefill unchanged —
butterfly is still used for large reductions. Falls back safely (returns
`nullptr`) when P2P is unavailable.
([#25197](https://github.com/ggml-org/llama.cpp/discussions/25197),
[patch repo](https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce))

Caveats, all from the author: only validated on gfx1201; RDNA3 and CDNA
unvalidated; not submitted as a PR pending independent testing on 7900 XTX and
mixed setups; use `HIP_VISIBLE_DEVICES` not `-dev` when benchmarking or you will
measure single-GPU by accident. One real bug fixed inside it:
`hipDeviceEnablePeerAccess` returning `hipErrorPeerAccessAlreadyEnabled` leaves a
sticky error that crashes unrelated kernels, fixed with
`(void) hipGetLastError();`.

The +56%-vs-single number is the most interesting datapoint in this whole
document, because *dense 27B Q8 decode across two cards* is exactly our worst
case. But it is gated on P2P working across a gfx1100/gfx1201 pair, which nobody
has tested.

### 5.2 TurboPrefill — layer-split prefill scheduling

A proof-of-concept implementing intra-prompt pipeline scheduling for multi-GPU
prefill in `--split-mode layer`. Identical outputs, better hardware utilisation.
Speedups over stock pipeline parallel at 16K context:

| Setup | Model | Speedup |
|---|---|---|
| 2× RTX PRO 5000 | Llama-3-70B | 1.7× |
| 4× RTX 3090 | GPT-OSS-120B | 1.9× |
| 4× RTX 3090 | Llama-3-70B | **3.0×** |
| 8× RTX 5060 Ti | GPT-OSS-120B | 2.2× |
| 12× P104-100 | Llama-3-70B | 5.3× |

([#24092](https://github.com/ggml-org/llama.cpp/discussions/24092),
[repo](https://github.com/sergey-automation/TurboPrefill))

All results are CUDA. ROCm/Vulkan support is not stated. Two cards is the bottom
of their tested range and shows the smallest gains, so this is speculative for us
— but our workload (long single-request prefill, layer split) is precisely the
target case.

### 5.3 UBBoost — separate ubatch for prefill vs decode

An RFC adding `--promptprocessing-ubatchboost-size N` so prefill can run a large
ubatch while decode keeps a small one. Reported up to 2× prefill in
VRAM-constrained cases; on **Qwen 3.6 35B** Q8_K_XL with MTP: **121.04 → 272.68
t/s (+125%)**; Q2_K_XL 389.12 → 538.69 t/s (+38%). Unmerged, `llama-server` only.
([#23262](https://github.com/ggml-org/llama.cpp/discussions/23262))

Directly relevant: our ubatch is a compromise between the prefill curve (still
climbing at 1024) and the VRAM ceiling on the XTX. This flag is exactly that
compromise removed. It is an unmerged patch on a build we deliberately pinned,
so it is a "later" item.

### 5.4 TurboQuant KV (`turbo3` / `turbo4`)

Randomized Hadamard transform + Lloyd-Max quantisation; 3.25 and 4.25 bits/value
(4.9× and 3.8× vs FP16). One implementer reports **output identical to f16 at
temperature 0** on a 35B. Example: 70B Q4_K_M with 34 GB of KV goes from ~109K
tokens (FP16) to ~536K (TQ3). Not merged; multiple community forks.
([#20969](https://github.com/ggml-org/llama.cpp/discussions/20969))

Flagged only because it is the one thing that would let us keep f16-equivalent
quality at q4-like memory — which is the constraint behind every tensor-split
decision in this repo. It is also explicitly incompatible with `-sm tensor`
today ([#23567](https://github.com/ggml-org/llama.cpp/issues/23567)).

---

## 6. Things that do NOT apply, and why

**`GGML_CUDA_P2P=1`** — CUDA-only runtime env var. The AMD path is RCCL at build
time.
([docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md))

**`-DGGML_HIP_ROCWMMA_FATTN=ON`** — discussion #15021 recommends it broadly
(MI300X pp512 4037 → 11945 with FA=1)
([#15021](https://github.com/ggml-org/llama.cpp/discussions/15021)), but our own
audit measured rocWMMA-off as faster on this box and it is off deliberately. Do
not "fix" this based on the external recommendation.

**ik_llama.cpp** — repeatedly surfaces in search results with large multi-GPU
claims including a new "split mode graph"
([Medium](https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2)),
but our own prior audit concluded it is a dead end on AMD. Disregarding.

**`--split-mode row`** — deprecated upstream, and errors here with *"device ROCm0
does not support split buffers"*.

**`-sm tensor` on Nemotron 3.5** — blocked at the architecture level
(`LLM_ARCH_NEMOTRON_H_MOE`), verified in our own source tree.

**Mixed-backend single process** (ROCm card + Vulkan card simultaneously) — a
toolkit exists for this but targets AMD+NVIDIA pairs; all four supported
combinations involve CUDA or are Vulkan-only.
([daimonionnn/multi-gpu-llm](https://github.com/daimonionnn/multi-gpu-llm))

**AMDVLK instead of RADV** — the R9700 work found AMDVLK better for *decode*
(+3.7% MoE, +13% dense with `rm_kq=1`) but catastrophically worse for prefill:
27B dense pp512 **203 vs 798 t/s**, a 4× deficit, only partly recovered by
`GGML_VK_DISABLE_COOPMAT=1` (207 → 243). Their overall verdict is RADV.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043)) Not worth a
driver swap; superseded by the earlier summary that suggested trying AMDVLK.

**Env vars measured as no-ops** (±0.3%) on RDNA4: `RADV_DEBUG=nocompute`,
`RADV_PERFTEST=sam/bolist`, `GGML_VK_DISABLE_COOPMAT/_F16/_BF16/_ASYNC/_GRAPH_OPTIMIZE`,
`GGML_VK_FORCE_MMVQ`, `GGML_VK_DISABLE_MMVQ`. Exception: `GGML_VK_DISABLE_MMVQ=1`
gave +0.4–1.7% on Qwen3.8 specifically.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

**ROCm dispatch-overhead bug** — worth knowing about but not actionable: a user
profiled Qwen 3.5 27B on ROCm at 14,977 kernel dispatches, 0.039 s of GPU compute
against ~4.1 s wall time — **99% dispatch overhead**, ~271 µs per dispatch.
Cold prefill 408.94 s on ROCm vs 40.37 s on Vulkan. `GGML_HIP_GRAPHS=ON` made it
*worse*; `GPU_MAX_HW_QUEUES=8`, `HSA_ENABLE_SDMA=0` and `GGML_CUDA_ENABLE_GRAPHS=1`
did nothing. Closed as a duplicate of
[#18823](https://github.com/ggml-org/llama.cpp/issues/18823), a ~92% prefill
regression on Qwen3-Next after PR #18683, unresolved on 4× R9700.
([#20292](https://github.com/ggml-org/llama.cpp/issues/20292))

We have `GGML_HIP_GRAPHS=ON` in both builds. Given it was reported as actively
harmful on a hybrid Qwen model, that is worth one A/B.

**`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`** — +4.7% AMDVLK MoE decode, **−8% AMDVLK
dense**, zero on RADV. We are on RADV, so this is a no-op for us.
([#21043](https://github.com/ggml-org/llama.cpp/discussions/21043))

---

## 7. Suggested test order

Grouped so each batch stays inside the 5-minute-per-run rule, and ordered so the
free things get measured before anything that needs a rebuild or a reboot.

**Batch A — free, no rebuild, no system change**
1. Grep server logs for `pipeline parallelism enabled (n_copies=4)`.
2. `RADV_PERFTEST=aco,cswave32,nogttspill` on both Vulkan profiles.
3. `-ub 2048 -b 16384` on the 35B ROCm profile (extends a curve we know is monotonic).
4. Add clock/GTT sampling to `batch.sh` alongside the existing VRAM logging.
5. **Re-measure the 35B `--cache-reuse` result** (§0.5.1) — the +3% was
   attributed to a disabled flag, so something else moved and we do not know what.
6. **Add a deep-context tier to the harness** (§0.5.2) — ~135k prefill plus a
   256-token decode, on the two profiles that see real use. Everything we have
   ranked so far was ranked at ~10k.

**Batch B — flags only, needs a memory prediction first**
5. Qwen3.8: `--parallel 3` + `n-max 7` + `-ub 288`, dual and XTX-only. Watch for
   0% acceptance (the `--kv-unified` bug) and run `xtx_predict.py` before launch.
6. `-mg 0` vs `-mg 1` on Muse and Nemotron.

**Batch C — one rebuild**
7. `-DGGML_SCHED_MAX_COPIES=1`, measure pp *and* tg on all four models.
8. Same build: `GGML_HIP_GRAPHS=OFF` A/B.
9. Check which `glslc` built `build-vulkan/`; rebuild against a current SDK if it
   is the system 2023-era one.

**Batch D — tensor parallel. Gate cleared 2026-08-16 (§4.7).**
10. ~~`rocm-bandwidth-test` → does gfx1100↔gfx1201 P2P exist?~~ **Done. YES,
    both directions.** No install was needed; `hipDeviceCanAccessPeer` answered
    it with zero allocation.
11. `-DGGML_HIP_RCCL=ON` rebuild, then `-sm tensor` on Qwen3.8 at reduced
    context. f16 KV and `-fit off` are already satisfied; context must come down
    because auto-fit is unavailable in tensor mode.
12. Same run, both `--main-gpu 0` and `--main-gpu 1` — the P2P legs are
    asymmetric (64000 vs 8000 max_bandwidth, and no PCIe atomics on R9700→XTX),
    so collective direction is a real variable here, not a formality.
13. The direct-P2P AllReduce patch (§5.1). First test of it on a
    mixed-architecture pair; the author asked for exactly this.

**Batch E — system changes, needs sudo at the terminal**
14. PCIe ASPM `performance` (+10.8% decode, dense). Runtime write, reversible.
15. `amdgpu.runpm=0` — kernel cmdline, needs a reboot.

---

## 8b. Cross-check against `DUAL_GPU_OPTIMIZATION_REPORT_2026-08-16.md`

A second agent produced a parallel report. Verified against the source tree and
the live log, 2026-08-16.

**Correct and valuable — folded into §0.5:**
- `--cache-reuse` is disabled at runtime. Understated, in fact: it identified the
  multimodal guard only, and the `llama_memory_can_shift` guard hits every one of
  our models regardless of vision.
- The 135,390-token / 491 t/s / 22.11 t/s production datapoint.
- Layer split remains the right default topology, and the two-profile
  (prefill-optimised vs decode-optimised) structure is worth keeping.

**Wrong:**
- *"`--split-mode tensor` … is not implemented for hybrid state-space
  architectures"*, therefore excluded for this model. The deny list in
  `src/llama-arch.cpp:1009` names 27 architectures and no Qwen appears among
  them; the running model reports `arch = qwen35moe` and is allowed. The
  conclusion was drawn from the doc's prose rather than the code, and it is the
  premise for that report's central "do not pursue tensor parallel"
  recommendation.

**Unreliable:**
- Its line-number citations do not resolve. `ppopt.sh:123` is cited for
  `--split-mode layer --device ROCm0,ROCm1 --tensor-split 1,1` but contains
  `LLAMA_TIMEOUT=3600`; `:99` is cited for the draft-depth setting but contains
  `LLAMA_SPEC_DRAFT_TYPE_V="q8_0"`; others land on blank lines. The underlying
  claims about the files are broadly right, but the anchors are not evidence.

**Incomplete for the question asked:**
- Its only external sources are llama.cpp's `multi-gpu.md` and an AMD
  compatibility page. It contains no survey of what other operators have
  measured, which is the gap this document exists to fill.

**Genuine defect it exposed in our own files:** `preset-38-27b-q8-rocm-dual-256k-ppopt.sh`
carries a comment block at lines 68–91 describing `f16 KV + ts 28,37` that was
copied from the mmproj profile, while the file actually sets
`LLAMA_CACHE_TYPE_K="q8_0"` at line 93 and `ts 1,1`. The header contradicts the
config. Worth fixing before it misleads a future reader.

---

## 9. Sources

**llama.cpp upstream**
- [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) — split modes, RCCL, NCCL, P2P, tensor-mode limits, troubleshooting table
- [#21043 — RDNA4 Llama Experiments: Squeezing Every Token/s from the R9700](https://github.com/ggml-org/llama.cpp/discussions/21043) — the single richest source here
- [#25197 — The internal HIP AllReduce is a stub, injected a direct-P2P path](https://github.com/ggml-org/llama.cpp/discussions/25197)
- [#23295 — RADV (Vulkan on AMD/Linux) Performance Debugging Guide](https://github.com/ggml-org/llama.cpp/discussions/23295)
- [#15021 — Performance of llama.cpp on AMD ROCm (HIP)](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [#10879 — Performance of llama.cpp with Vulkan](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [#20252 — Does llama.cpp ACTUALLY support pipeline parallelism?](https://github.com/ggml-org/llama.cpp/discussions/20252)
- [#24092 — TurboPrefill: Multi-GPU prefill acceleration](https://github.com/ggml-org/llama.cpp/discussions/24092)
- [#23262 — RFC: Speed up prefill up to 2x by increasing ubatch for prompt processing only](https://github.com/ggml-org/llama.cpp/discussions/23262)
- [#20969 — TurboQuant: Extreme KV Cache Quantization](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [#22871 — Dual AMD AI Pro R9700 cards failing](https://github.com/ggml-org/llama.cpp/discussions/22871)
- [#21112 — Optimize my llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/21112)
- [#13751 — Large performance drop when using pipeline parallelism and layer split](https://github.com/ggml-org/llama.cpp/issues/13751)
- [#23567 — Support quantized KV cache with --split-mode tensor](https://github.com/ggml-org/llama.cpp/issues/23567)
- [#25345 — llama-server silently ignores -md when the unified KV cache is active](https://github.com/ggml-org/llama.cpp/issues/25345)
- [#20292 — Qwen 3.5 CPU bound on rocm / terrible performance](https://github.com/ggml-org/llama.cpp/issues/20292)
- [#18823 — Qwen3-Next prefill regression on HIP after PR #18683](https://github.com/ggml-org/llama.cpp/issues/18823)

**Patches and forks**
- [JohnTDI-cpu/llama-hip-p2p-allreduce](https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce)
- [sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill)
- [daimonionnn/multi-gpu-llm](https://github.com/daimonionnn/multi-gpu-llm)
- [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm) — ROCm 7 nightly builds; note gfx1100/gfx1201 are not in the published target list

**AMD / ROCm / kernel**
- [ROCm#2253 — 7900XTX cannot pass rocm-bandwidth-test](https://github.com/ROCm/ROCm/issues/2253)
- [Phoronix — AMD kernel driver enabling peer-to-peer multi-GPU compute for Linux](https://www.phoronix.com/news/AMD-Multi-GPU-Compute-P2P)
- [Fang-Pen Lin — Two AMD 7900XTX GPUs with peer-to-peer PCIe communication](https://fangpenlin.com/posts/2025/06/11/two-amd-7900xtx-gpus-tinygrad-based-training-workstation-peer-to-peer-pcie-communication/)
- [HIP environment variables](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)
- [Debugging with HIP — HSA_ENABLE_SDMA](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/debugging.html)

**Third-party writeups**
- [Pipeline parallelism in llama.cpp may be wasting your VRAM](https://prismix.dev/news/b0e27b2381c2) ([mirror](https://bittide.aicompass.dev/article/5666a8e4-c0a3-4ff9-9c70-c9138c19403c))
- [llama.cpp Multi-GPU: 2 GPUs vs 1, tensor-split, and VRAM](https://knightli.com/en/2026/05/09/llama-cpp-multi-gpu-offload-performance/)
- [llama.cpp performance breakthrough for multi-GPU setups](https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2) — ik_llama.cpp claims, disregarded per §6
- [Llama.cpp on AMD gets 13% faster prompt processing with RADV driver update](https://www.hardware-corner.net/llama-cpp-amd-radv-vulkan-driver-update/)

**Local, this machine (2026-08-16)**
- `build/CMakeCache.txt`, `build-vulkan/CMakeCache.txt`
- `src/llama-arch.cpp:1009` — `llm_arch_supports_sm_tensor()` deny list
- `/proc/cmdline`, `/sys/module/pcie_aspm/parameters/policy`,
  `/sys/module/amdgpu/parameters/{runpm,pcie_p2p}`,
  `/sys/class/drm/card*/device/{current_link_speed,current_link_width,power_dpm_force_performance_level}`
