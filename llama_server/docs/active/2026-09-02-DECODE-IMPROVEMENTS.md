# Faster decode: what the fast setups do that we don't

Research-only. No GPU test run, no code built, no server started or stopped.
Model: Qwen3.8-Flash-Next 125B-A6B UD-Q2_K_XL, R9700 + 7900 XTX, Vulkan/RADV.

Current: **613 pp / 40.9 tg @11.5K**, **39-40 tg @15-18K**, **347.8 pp / 18.59 tg @199,340**.

---

## 0. Do these four things first — no build, no code

| # | Change | From | To | Evidence | Expect |
|---|---|---|---|---|---|
| 1 | `--threads` | auto (=16) | **4** | EngramHalo, same CPU class | **tg +17% @16K** |
| 2 | Fix swap, `--cache-ram` | 8192, swap 94% full | **0**, swap freed | our own `free` | removes stalls |
| 3 | New preset: ctx + ubatch | 200704 / ub192 | **65536 / ub768** | every fast setup uses ub2048 | **pp 600 → 850-1100** |
| 4 | `-ot` device placement | `-ts 400,600` only | `-ot ...=Vulkan0/1` | 3090 author's own note | breaks the flat ts band |

Then the one build that matters: **PR #28213**, section 5.

---

## 1. Side-by-side: us against the two fastest comparable systems

| | **ruashots 2×3090** | **EngramHalo Strix** | **us** |
|---|---|---|---|
| quant | UD-Q2_K_XL (same) | UD-IQ3_XXS | UD-Q2_K_XL |
| routed experts | 42.92 GiB (**identical**) | smaller | 42.92 GiB |
| ubatch | **2048** | **2048** | **192** |
| ctx allocated | **32,768** | 32K slot | **200,704** |
| KV | f16 | q8_0 | q8_0 |
| threads | 8 | **4** | **auto → 16** |
| `--cache-ram` | **0** | — | **8192** |
| placement | **`-ot` named tensors** | — | `-ts 400,600` |
| expert placement | 8 layers on CPU | all resident | all resident |
| decode @~16.5K | **42.11** | 20.96 | 39-40 |
| prefill @~16.5K | **892** | 381.9 | ~600 |

Five of those differences are ours to change. Three are one flag each.

The 3090 numbers are medians of 3 from `raw/results-master.jsonl`, temp 0.7 /
top_p 0.95, 512 output tokens, unique nonce per request, two 512-token warm-up
generations. Their headline **53.3 tg is the 128-token cell**, not 16.5K — use
42.11 when comparing against our 15-18K conversations.

---

## 2. What is actually eating our decode at depth

Our own numbers locate it precisely:

```
11,568 tok  → 40.9 tg = 24.4 ms/token
199,340 tok → 18.6 tg = 53.8 ms/token
                        ------------
depth cost           =  29.4 ms/token
```

The scan fix (28011 + 28e37926 + b88668b) removed ~51M CPU iterations per token
and bought **+39.3% at 199K**. That proves a large slice of the depth cost is
**CPU-side, per-token, and scales with cached cells**. Three such costs remain:

**a) The KQ mask is still built, filled, cast and uploaded every decode step.**

```c
// src/llama-graph.cpp:974   — allocated at full n_kv, every step
inp.kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, plan.n_kv, ...);

// src/llama-graph.cpp:454   — filled host-side across every element
fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), ...);

// src/models/qwen4exp.cpp:610 — and cast to F32, per sparse layer
ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32
                   ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);

// src/models/qwen4exp.cpp:775 — our gather still reads the full-width mask
ggml_tensor * kq_mask = inp->get_kq_mask();
ggml_tensor * mask_cells = ggml_view_3d(ctx0, kq_mask, 1, n_kv, n_q, ...);
```

At n_kv = 200,704 that is a 392 KiB host fill plus 12 sparse layers × 200,704 ×
4 B ≈ **9.6 MB of GPU write per decoded token**, purely to mask out cells the
gather already discarded. `d7ca8ea` gathers the KV rows but leaves all of this
in place — which is why it only bought us **+1.07%**.

**b) The PLE gather is CPU-side, every token, and the box is swapping.**

```
Mem:  60,854 MB total   17,559 used   706 free   47,896 buff/cache
Swap:  8,191 MB total    7,690 used  →  94% FULL
```

`per_layer_token_embd.weight` is **26.822 GiB** (IQ4_NL, [160 × 320,001,536]) and
is gathered by token id on the CPU on every decode step. It is served from page
cache through mmap. On a box 94% into swap with 706 MB free, those gathers are
one eviction away from becoming disk reads. `--cache-ram 8192` reserves another
8 GiB of host RAM on top. The 3090 setup runs `--cache-ram 0`.

**c) Thread oversubscription on those gathers.**

`LLAMA_THREADS="auto"` resolves to physical cores — 16 on the 9950X3D.
EngramHalo, on a 16C/32T Zen 5 part, measured `-t 16` → `-t 4` as
**13.4 → 15.7 tg at 16K, +17%**, and attributes it explicitly to the CPU-side
PLE gathers. Same CPU class, same CPU-resident PLE, same architecture.

---

## 3. The dual-3090 mechanism, verified in source

The +25% over `--n-cpu-moe` was **VRAM balance**, and it has no analogue here —
but the *technique* they used to get it does.

`--n-cpu-moe N` is pure `-ot` sugar. `common/arg.cpp:2789` calls
`llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, ...)`, which pushes the
expert weights of the **first N layers** to CPU. No runtime difference from a
hand-written band. Their telemetry (`scripts/sysmon.sh`) shows what went wrong:

- `--n-cpu-moe 32` → GPU0 5.3 GiB, GPU1 23.2 GiB. ~19 GiB of VRAM idle.
- Host set then exceeded page cache → **NVMe reads 766 MB/s during decode**,
  both GPUs idling at 7-25% utilisation.
- Two bands (0-8, 25-31) → 20.3 / 21.4 GiB, **NVMe fell to 38 MB/s**, +25% decode.

Their conclusion in their own words: *"`-ts` can also improve the balance, but it
steers more than just the expert placement, and the layers are not equal in size
anyway. I had better results naming the tensors directly."*

**That is the transferable part.** Our ts band 390-420 measured flat because
`-ts` moves whole layers and the XTX hits capacity at ~47%. `-ot` accepts any
device buffer name, not just `CPU` — `parse_tensor_buffer_overrides`
(`common/arg.cpp:252-283`) builds its map from `ggml_backend_dev_buffer_type()`
over every registered device, so `=Vulkan0` and `=Vulkan1` are legal targets.
That gives per-tensor placement granularity that no `-ts` value can express.

### Should we move experts to CPU? No.

Verified, and it settles question 2 of the brief:

```c
// ggml-backend.cpp:961 — host-buffer weights go to CPU unless a GPU claims the op
if (sched->op_offload && src_backend_id == sched->n_backends - 1
    && ggml_backend_buffer_is_host(src->buffer)) { ... }

// ggml-cuda.cu:5344  ==  ggml-vulkan.cpp:19314 — identical policy
return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;   // default 32
// get_op_batch_size(MUL_MAT_ID) == op->ne[2] == n_tokens
```

- **Decode, batch 1:** 1 < 32 → the expert matmul **runs on the CPU**. Activations
  cross PCIe both ways with a hard synchronise. No weights are uploaded.
- **Prefill, ubatch ≥ 32:** offloaded to GPU, and only the *routed* experts are
  copied, via the `MUL_MAT_ID` used-expert bitmap at `ggml-backend.cpp:1643-1690`.
- **CUDA vs Vulkan vs HIP:** no difference. Same threshold, same batch-size
  function, same selective-upload path. HIP compiles the CUDA file.
- **No concurrency:** `ggml_backend_sched_compute_splits` walks splits in order
  with an explicit `ggml_backend_event_synchronize` / `ggml_backend_synchronize`
  between backends. A CPU split blocks the GPUs.

At batch 1 this moves work from 640 GB/s onto DDR5 and adds two PCIe copies plus
a sync per layer. It strictly lengthens the critical path. The 3090 author agrees
from the other side: *"Every expert layer moved back to the GPU removes CPU
compute, transfer and synchronisation from every token. Published results with
all of the experts resident are much faster."*

---

## 4. Evidence table

| Source | Hardware | Engine | Commit | Quant | Model size | VRAM part | CPU part | ctx | Depth | Out | Sampler | Spec | Cold? | pp | tg | Command / mechanism | Applies | Conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **us** preset 14 | R9700 32.6G + XTX 24.5G | llama.cpp Vulkan/RADV | `aa1b06d0c`+fusions | UD-Q2_K_XL | 73.44 GiB | 46.21 GiB | 27.23 GiB | 200,704 | 199,340 | 128 | t1.0/k20/p0.95 | none | warm | 347.8 | 18.59 | `-sm layer -ts 400,600 -ub 192 -ctk/-ctv q8_0` | — | measured |
| **us** preset 14 | same | same | same | UD-Q2_K_XL | 73.44 GiB | 46.21 GiB | 27.23 GiB | 200,704 | 11,568 | 128 | t1.0/k20/p0.95 | none | warm | 613 | 40.92 | same allocation, shallow prompt | — | measured |
| ruashots 2×3090 | 2×3090 24G, 9800X3D, 64G DDR5-4800, Gen4 x16, no NVLink, 1 NUMA | llama.cpp CUDA 13.0, drv 595.71.05 | master post-#27742 | UD-Q2_K_XL | 93.67 GB | 20.8+19.1 GiB | 35.8 GiB | 32,768 | 16,525 | 512 | t0.7/p0.95 +GGUF k20, min_p .05 | none | warm, 2×512 tok | **892.1** | **42.11** | `-ngl 99 -sm layer -fit off -c 32768 -fa on -ctk f16 -ctv f16 -np 1 --cache-ram 0 -b 2048 -ub 2048 -t 8 -ot "^per_layer_token_embd\.weight$=CPU" -ot "blk\.([0-3]\|2[5-8])\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps=CPU"` | config yes, offload no | raw jsonl |
| ruashots 2×3090 | same | same | same | UD-Q2_K_XL | 93.67 GB | balanced | 35.7 GiB | 32,768 | 16,528 | 512 | same | none | warm | 778.0 | 37.82 | 16 host layers (bands 0-8, 25-31) — wastes ~8 GiB VRAM | offload costs | raw jsonl |
| ruashots 2×3090 | same | same | same | UD-IQ4_XS | 93.67 GB | 20.3+21.4 GiB | 48.9 GB | 32,768 | 16,528 | 512 | same | none | warm | 505.4 | 32.12 | larger expert pool 55.43 GiB → more host layers | quant choice | raw jsonl |
| ruashots long-ctx | same | same | same | UD-IQ4_XS | 93.67 GB | 21.8+22.5 GB | ~49 GB | 131,072 | 65,724 | 512 | same | none | warm | 413.5 | 21.88 | 22 host layers, `-ub 1024`; longest completed 119,482 tok | we beat this at 3× depth | raw jsonl |
| ruashots `-ncmoe 32` | same | same | same | UD-IQ4_XS | 93.67 GB | 5.3+23.2 GiB | rest | 32,768 | — | 512 | same | none | thrashing | — | −25% | 19 GiB VRAM idle → host set > page cache → 766 MB/s NVMe, GPUs 7-25% util | no analogue | telemetry |
| EngramHalo | Strix Halo gfx1151, 16C/32T Zen5, 96G UMA | llama.cpp HIP/ROCm 7.14 | `af1ffaf37`+patches | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB PLE | 262,144 | 131,000 | 64 | temp 0 | none | SSD-lazy | 192.3 | 9.53 | QSA gather + RDNA3.5 kernels; residual decay is the indexer | findings yes, kernels no | documented |
| EngramHalo threads | same | same | `b8bdf73bb` | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB | — | 16,384 | 128 | llama-bench | none | warm | — | 13.4 → **15.7** | `-t 16` → `-t 4`, +17%, CPU-side PLE gathers | **directly, same CPU class** | controlled |
| EngramHalo ubatch | same | same | same | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB | — | 16,384 | 128 | llama-bench | none | warm | 200.2 → 223.6 → **230.1** | flat | ub 512 → 1024 → 2048; +15% @16K, +22% @0 | **yes** | controlled |
| EngramHalo load mode | same | same | same | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB | — | 16,384 | 128 | llama-bench | none | **warm** | 376.5 → 376.9 | 21.5 → 21.4 | `-lm none` vs mmap: +5% pp @0, **flat at depth, flat decode** | deflates "1.87×" | controlled |
| EngramHalo MTP | same | same | same | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB | 163,840 | 156,400 | 400 | **temp 0** | draft-mtp,ngram-mod n-max 4 p-min .75, 66% accept | warm | — | 20.8 (plain 9.1) | win **grows** with depth: +47% @0K, +63% @78K, >2× @156K | greedy only | documented |
| EngramHalo ngram | same | same | same | UD-IQ3_XXS | 76.32 GiB | UMA | 26.82 GiB | 32,768 | code | 400 | temp 0 | `ngram-mod` alone | warm | — | 24.7 → **27.8** | **+12.6%, the honest isolated figure** | yes | controlled |
| PR #28213 | 2× RTX A6000 | llama.cpp CUDA | `50d50ba31` on `b96806d96` | UD-IQ4_XS | — | resident | PLE | — | 130,000 | — | temp 0 | none | warm | unchanged | 15.7 → **23.6** | gathers K/V **and drops the full KQ mask** (~17 MB/token @130K); `QWEN4EXP_QSA_GATHER=0` toggle | **backend-generic** | author A/B |
| PR #28136 + 3rd party | DGX Spark GB10; 125 GiB-RAM box | llama.cpp CUDA | open | UD-Q4/Q5_K_XL | — | — | 26.82 GiB PLE | 131,072 | 2.4-6.5K | 1 | — | none | **cold** | 250 → 337 (+20-32%) | unchanged | 31-61k major faults → 0, replaced by `pread`. **Warm = −0.4 to +2.1%** | only if PLE uncached | n=5 ±sd |
| Discussion #27864 | unstated | llama.cpp | — | Q8_0 | — | — | PLE | — | 512 | — | — | none | read vs resident | 554.6 vs 272.8 | — | PLE resident vs on-demand SSD = **2.03×** | this is where "1.87×" comes from | no hw specs |
| julianmb haloq38flash | Strix Halo 128G | llama.cpp **Vulkan/RADV** | fork | IQ4_XS-PLE 91G | 91 GiB | UMA | PLE | 262,144 | 128,000 | — | **temp 0** | none (plain row) | n=1 | 222 | 11.0 | ub2048, q8_0 KV, Vulkan FA/MMQ tuning | closest Vulkan peer | n=1, greedy |
| DominikBucko 2×3090 | 2×3090, **128 GB RAM** | **vLLM** fork | pinned | W4A16 + FP8 PLE | 120 GiB | LRU-104 hot experts | full expert pool | 262,144 | 128 | 4,096 | — | Adaptive MTP3 | warm | 1,402 | 135.2 | dynamic LRU expert cache, Marlin, fused QSA, MTP3; target-only is 61.3 | **not comparable** | documented |
| Dyluhn/R9V | **2× R9700** gfx1201, 128G | vLLM fork + R9V HIP kernels | pinned submodules | R9V IQ4_XS | 90.36 GiB | — | SSD PLE | 131,072 | — | — | — | — | — | — | — | model-and-machine-specific engine | **refuses gfx1100** | RC, ungated |

Local tensor table, measured with `gguf-py` (not estimated):

| group | size |
|---|---|
| `per_layer_token_embd.weight` (IQ4_NL, [160 × 320,001,536]) | **26.822 GiB** |
| routed experts | **42.920 GiB** — byte-identical to the 3090 report |
| other block tensors | 2.954 GiB |
| `token_embd.weight` | 0.407 GiB |
| `output.weight` | 0.340 GiB |
| **total** | **73.442 GiB** |

GPU-resident = 46.213 GiB = 47,322 MiB. Measured VRAM at 199,382 tokens was
30,564 + 23,915 = 54,479 MiB, so **KV + indexer + compute buffers = 7,157 MiB**
at ctx 200,704.

---

## 5. Ranked improvements

### 1. PR #28213 — gather-based QSA that drops the mask

`+112/−9` across `src/models/qwen4exp.cpp`, `src/models/models.h`,
`src/llama-graph.cpp`. Base `b96806d96`, mergeable, decode-only.

- **pp:** unchanged by construction — prefill keeps the masked path.
- **tg:** author measures +6% @31K, +19% @62K, **+50% @130K** (CUDA, temp 0).
  Discount for us: we already have `d7ca8ea`, which took the gather half and gave
  +1.07%. The untested increment is the mask elimination — section 2a. Prior:
  **+5% to +25% at 199K**, and it is the only candidate whose mechanism matches
  the shape of our decay curve.
- **memory:** strictly lower. Removes a full-width mask allocation and the
  per-layer F32 cast; adds a compact 2048-row buffer. **Cannot spill the XTX.**

**A/B:** build on **fusionscan** (preset 13), not qsagather — #28213 supersedes
`d7ca8ea`. Then A/B *in the same binary*: `QWEN4EXP_QSA_GATHER=0` (control) vs
unset. Same allocation, prompt, sampler, seed. Depths 11,568 then 199,340; n=3
per arm, medians. Gate on correctness first: the author reports byte-identical
retrieval answers between paths at 31K/62K/130K — reproduce at 11.5K before
trusting a deep number.

This is the cleanest experiment available: the confound that invalidated the
fusions-vs-stock and cross-session MTP comparisons cannot occur in a same-binary
env-toggle A/B.

### 2. `--threads 4` instead of auto

Zero code, zero VRAM, no rebuild. `--threads-batch` stays 16.

- **tg:** 0 to +17%. EngramHalo's controlled measurement on the same CPU class.
- **pp:** flat to slightly down; keep `--threads-batch` high.
- **memory:** none. Cannot spill.

**A/B:** sweep `FN_THREADS ∈ {4, 8, 16}` at 11,568 tokens, n=3, production
sampler. Winner goes to 199,340.

### 3. Reclaim host RAM: `--cache-ram 0` and clear swap

Swap is 7,690/8,191 MB used with 706 MB free. The PLE is 26.822 GiB gathered
from host memory every decode step. `--cache-ram 8192` reserves 8 GiB more.

- **tg:** unquantified but the failure mode is severe — a swapped PLE page turns
  a per-token gather into a disk read. Fixing it removes a stall source rather
  than adding throughput.
- **memory:** frees 8 GiB of host RAM. No VRAM change. Cannot spill.

**A/B:** `swapoff -a && swapon -a` on an idle box to drain swap, then compare
`--cache-ram 0` vs `8192` at 199,340, n=3, watching `free` and major faults
across the run. Note losing the prompt cache costs re-prefill on repeat prompts —
measure that separately before shipping it.

### 4. Second preset: ctx 32,768-65,536 at ub 768-1024

Real conversations sit at 15-18K. The 200,704 allocation is resident regardless
and is what caps us at ub192. Keep preset 14 for deep work; this is an addition.

Our own sweep at 11.5K, on fusionscan, with the full 200K allocation still held:

| ub | pp | tg | R9700 | XTX |
|---|---|---|---|---|
| 128 | 597.69 | 41.40 | 30,311 | 23,487 |
| 192 | 610.29 | 41.15 | 30,564 | 23,754 |
| 256 | 633.83 | 41.16 | 30,857 | 24,009 |
| 384 | **744.35** | 41.38 | 31,438 | 24,032 |
| 512 | failed | — | — | — |

ubatch costs ~2.1 MiB of XTX VRAM per unit; decode is flat across the sweep.

- **pp:** **850-1,100 @16K**, from ~600. The 3090 box reaches 892 @16.5K at
  ub2048; EngramHalo measured ub512→2048 as +15% @16K.
- **tg:** flat. Decode is depth-bound, not allocation-bound.

**Safety arithmetic.** KV + indexer + compute = 7,157 MiB at ctx 200,704. At
ctx 65,536 (32.6%) the KV-proportional share mostly releases; on a conservative
estimate of 5,000 MiB being KV+indexer, that frees ~3,370 MiB, of which the XTX's
40% share is ~1,350 MiB. Raising ub 192 → 1024 costs 832 × 2.1 ≈ **1,750 MiB** on
the XTX — *more than the estimate frees*.

**So start at ub512** (+320 units ≈ 672 MiB, comfortably inside the freed
budget), read `gpu-memory.tsv` at readiness *and at full depth*, and step up only
on measured headroom. At ctx 32,768 the budget roughly doubles and ub1024 becomes
plausible. XTX guard ceiling stays 24,200; measured headroom at 199K was
**645 MiB physical** on card1, so nothing here may be shipped on a shallow read.

> The 3090 report hit `cudaMalloc failed` at 17,849 tokens on a config that had
> loaded cleanly and answered short prompts. Compute buffers grow with cached
> tokens *and* ubatch. A 32K preset must be validated by actually filling 32K.

### 5. `-ot` inter-GPU placement instead of `-ts` alone

`-ts` moves whole layers, the layers are unequal, and the XTX caps at ~47% of
weights — which is why the 390-420 band measured flat. `-ot` targets individual
tensors at any device (`=Vulkan0`, `=Vulkan1`), which `-ts` cannot express.

- **Expected:** small. At 199K the decode budget is dominated by attention and
  the indexer, not weight traffic — only ~6B params are active per token. The
  flat ts band is consistent with that. Treat as a refinement after 1-4, not a
  headline.
- **memory:** fully controllable, that is the point. Compute the byte moves
  before running; never place a tensor onto the XTX without subtracting it from
  the 645 MiB deep-context headroom first.

**A/B:** at fixed ts 400,600, move one expert tensor group at a time from
Vulkan1 to Vulkan0 and back, measuring VRAM at load and speed at 11.5K. Only take
a configuration deep once its load-time XTX figure is below where ts 400 sits today.

### 6. MTP at depth under production sampling

The largest claimed upside, and still the weakest evidence — **every published
number is greedy**. EngramHalo: +47% @0K, +63% @78K, >2× @156K, all temp 0.
julianmb: 24.1 → 56.4 @8K, temp 0, n=1. Our own greedy test: +46.7% @151K. Our
controlled temp-1.0 test at 11.5K had MTP **losing by 21%**.

What changed upstream since our base: **#28123 (recurrent state rollback) merged
1 Sept as `0eadefebd`**; **#27941 (seq_cp, block position keying, mtmd input,
CUDA abort, tests) merged as `36b101543`**; #27991 optimises restoring
non-contiguous cells. The on-device checkpoint fix that explains the earlier MTP
disaster is now in master.

- **memory:** draft ≈1.77 GiB on GPU after pushing its two embedding tables to
  CPU with `-otd`. Fits the R9700's 2,060 MiB — but the draft does not fit at
  ctx 200,704, so this needs **ctx ≤ 163,840**.

**A/B:** four arms at ctx 163,840, temp 1.0 / top_k 20 / top_p 0.95, n=3:
(A) no spec, (B) `draft-mtp` n-max 4, (C) B + `--spec-draft-p-min 0.75`,
(D) C + `ngram-mod`. Report acceptance next to tok/s for every arm. Measure at
11.5K, 78K and 156K — a shallow loss is expected and is not the answer.

**Speculation, honestly.** Do not carry 80-110 tok/s into planning. EngramHalo
documents the trap: after a random-word filler prompt at temp 0 the model
degenerates, the n-gram speculator reaches ~100% acceptance and decode reads
2-3× high — **73 tok/s observed**. Repeating an identical request reads ~35% high
(52.7 vs 39.3). The honest isolated figure for `ngram-mod` on real code is
**24.7 → 27.8, +12.6%**. Flags where that was measured:
`--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75`,
`--spec-ngram-mod-n-match` at its default of 24.

### 7. Loading mode — lowest payoff, highest risk right now

Our server log already prints:

```
W llama_model_loader: tensor overrides to CPU are used with mmap enabled
                      - consider using --load-mode none for better performance
```

- **Is 1.87× real?** Not as a load-mode property. The large multipliers compare
  **PLE served from RAM/page cache** against **PLE read on demand from SSD** —
  Discussion #27864 measures 554.6 vs 272.8 pp512, a 2.03× ratio of exactly that
  kind. With a warm page cache, EngramHalo's controlled comparison gives `-lm none`
  **+5% pp @0, flat at 16K, flat decode**. PR #28136's independent tester:
  *"Warm is a no-op, −0.4% to +2.1% across all 12 cells."*
- **CUDA-specific?** No — the mechanism is page faults on a mapping, which is
  kernel-side. What is GB10-specific is the *magnitude*: the discrete-GPU
  replication gives +20-32% **cold**, decaying with prompt length (+31% @2.4K →
  +21% @6.5K) because PLE fetch is roughly fixed per unique row while compute
  grows. At 199K it is a rounding error.
- **RAM required:** with `-ot …=CPU` keeping only PLE and `token_embd` host-side,
  `-lm none` needs **27.23 GiB of anonymous RSS**.
- **Is 64 GB enough?** We have **59.4 GiB**, and swap is already 94% full.
  Converting 26.8 GiB of *evictable* page cache into 27.2 GiB of *unevictable*
  anonymous memory on a box that is already swapping is the one candidate that
  can hurt the desktop rather than just the benchmark. **Do #3 first.**

PR #28136's `--lazy-mode on-direct` is the better long-term answer — explicit
`pread`s that never enter the page cache, resident footprint ~1-1.5 GiB. Worth
watching, not porting: it is open, it lands on a loader path our base has since
diverged from, and our prefill is not currently PLE-bound.

---

## 6. Optimization inventory

| Optimization | Where | Backend | Status |
|---|---|---|---|
| QSA gather, mask-free (PR **28213**) | qwen4exp.cpp, models.h, llama-graph.cpp | generic | **untested — rank 1**, supersedes `d7ca8ea`, runtime toggle |
| QSA gather (PR 27977 `d7ca8ea`) | qwen4exp.cpp | generic | present in preset 14, **+1.07% @199K** |
| EngramHalo "true QSA sparse gather" (`2606d499`) | qwen4exp.cpp | generic | same class; env-gated `LLAMA_QSA_GATHER`, threshold n_kv 16384 |
| CPU sequence-scan fixes (28011, `28e37926`, `b88668b`) | llama-kv-cells.h | generic | present — this was our +39.3% |
| Indexer head slices (PR 28023) | qwen4exp.cpp | generic | tested, noise. Merged upstream `09412af38` |
| O(log n) get_prev_tokens (PR 28040) | llama-kv-cells.h | generic | tested, flat. Merged `b356fa262`; 28011 took the win |
| HIP wide top-k RDNA 3.5 (`33766da9`) | ggml-cuda/HIP | HIP only | **not applicable** — fixes a CPU fallback Vulkan lacks; our tree ships `topk_radix.comp` + `pipeline_topk_radix_f32` |
| FA-vec masked-slice skip, RDNA head-256 (`bf8412d7`) | fattn-vec | CUDA/HIP | wrong backend; solves the dense-mask cost the gather route avoids outright |
| Chunked GDN prefill (`62160a7c`) | gated_delta_net.cu | HIP, opt-in | wrong backend. `GGML_HIP_GDN_CHUNK=1`, RDNA3/4. Our ROCm path is 310 pp / 10.5 tg |
| Vulkan GDN cache-write fusion (PR 27973) | ggml-vulkan | Vulkan | tested, flat. We also fixed its cache-binding defect — worth upstreaming |
| Engram row prefetch + IQ4_NL get_rows (`c911e6bb`) | llama-mmap, ggml-cuda | CUDA/HIP | moot — the 160-value gather runs on CPU for us by design |
| `LLAMA_MMAP_DROP_BEHIND` (`5486559c`) | llama-mmap | generic | load-time only; relevant if `-lm none` is ever attempted |
| Direct PLE reads (PR 28136) | llama-mmap, qwen4exp.cpp | generic | rank 7. Cold-cache prefill only |
| `can_reuse()` graph-input fixes | qwen4exp.cpp | generic | already upstream — EngramHalo: *"part of the upstream arch now"* |
| Vulkan top_k radix select (PR 28032) | ggml-vulkan | Vulkan | **present** — confirmed in our tree |
| Quantized KV + `--split-mode tensor` (#23567) | llama-model, backends | generic | **closed, implemented** — PR #23792, June 2026 |
| MTP rollback (#28123), qwen4exp fixes (#27941) | qwen4exp.cpp, llama-kv-cache | generic | **merged 1 Sept** — prerequisites for rank 6 |

---

## 7. Two corrections to the local record

1. **Handoff 10 §5 names issue #23567 as the only escape from the pipeline
   ceiling. That issue is closed** — quantized KV with `--split-mode tensor` was
   implemented in PR #23792, 1 June 2026. Our own tree confirms it: the
   `"SPLIT_MODE_TENSOR and KV cache quantization not implemented"` string is gone,
   and `llm_arch_supports_sm_tensor` (`src/llama-arch.cpp:1115`) falls through to
   `default: return true` for qwen4exp. Our TP tests were run at f16/ctx16K under
   a constraint that no longer applies. This does not make TP promising — Vulkan
   TP still lost ~2× at 11.5K — but the stated blocker is wrong and the retest is
   now cheap.
2. **Handoff 10 §8 gives the PLE as "~20 GB". It is 26.822 GiB.** The
   `token_embd.weight` figure of 0.407 GiB is correct.

**Methodology note on our own numbers.** The 199,340-token prompt is the first
611,800 characters of `ggml-vulkan.cpp`. PR #28136's core finding is that
*repetitive* prompts need very few unique PLE rows, which reads prefill high — its
author saw 700+ tok/s synthetic against 300 on real input. C++ source is highly
repetitive. Our 348 pp at 199K may not survive a diverse-text prompt of the same
length. Worth one control run before treating it as the number to beat.

---

## 8. Not comparable

- **DominikBucko 2×3090, 135.2 tg / 1,402 pp.** vLLM, not llama.cpp. W4A16+FP8-PLE
  checkpoint, dynamic LRU-104 GPU expert cache, Marlin and Humming kernels, fused
  QSA, adaptive MTP3, **128 GB RAM**. Its own hillclimb puts target-only at
  61.3 tok/s. None of that machinery exists in llama.cpp or for AMD.
- **Every Qwen3.8-27B result.** 27B dense in vLLM on that same 3090 box runs
  96-99 tok/s; RTX PRO 6000 reports 119. Different model, ~4.6× fewer total
  params, fully resident, no PLE table, no QSA indexer, no gated DeltaNet.
  Dual-5060 Ti TP+MTP reports are in this category too.
- **halogen-flash-server, "3.8× faster end to end".** gfx1151 only, closed custom
  kernels, **no sampler at all** — any `temperature > 0` returns 400. Its own
  README calls the decode column "the soft one".
- **julianmb haloq38flash, 56.4 tok/s.** MTP at temp 0, n=1, Strix Halo UMA. Its
  *plain* Vulkan rows are the useful part.
- **Dyluhn/R9V.** The most tempting: dual R9700, gfx1201, ROCm. But it is a vLLM
  fork with a hard contract requiring *two* 32 GiB R9700s and refuses to run
  elsewhere. Our gfx1100 XTX disqualifies the machine; no profile is "Qualified".
- **Every DGX Spark / GB10 repo.** Unified memory changes the mmap, PLE-residency
  and expert-offload economics completely.
- **alesha-pro 4×3090 AWQ INT4.** vLLM, four GPUs, FP8 QSA KV, CUDA graphs.

---

## 9. First test

**Build fusionscan + PR #28213 in a fresh worktree and A/B it in one binary with
`QWEN4EXP_QSA_GATHER`.**

It is a same-binary A/B, so build, allocation, prompt, sampler and seed are
identical by construction. It is decode-only by construction, so a prefill change
is itself a red flag. It cannot raise VRAM, so the display-driving XTX is safe
without a guard argument. It targets the one cost whose shape matches our decay
curve. Cost: one build plus about an hour of benchmark time.

**Fold in while it builds, since neither needs a rebuild:** `FN_THREADS=4` versus
`auto`, and `--cache-ram 0` on a drained swap.

If #28213 is null, that is still worth knowing — it would mean the Vulkan fusion
stack already absorbs the mask cost, and attention moves to ranks 2, 3 and 4,
all of which are flag-only.

---

## Sources

`ruashots/flashnext-2x3090` — README, `raw/results-master.jsonl`,
`raw/raw-phaseD.log`, `raw/raw-matrix.log`, `scripts/run_ot.sh`,
`scripts/finalrun.sh`, `scripts/phaseD.sh`, `scripts/bench.py`,
`scripts/sysmon.sh`.
llama.cpp PR #28213, PR #28136 + comment thread, issue #23567, discussion #27864,
commits `09412af38` `0eadefebd` `36b101543` `b356fa262` `62acc89c2` `2d8d612e4`
`daef7b687`.
`Aristo94/EngramHalo.cpp` — `docs/strix-halo/BENCHMARKS.md`, commits `2606d499`
`33766da9` `bf8412d7` `62160a7c` `c911e6bb` `5486559c`.
`julianmb/haloq38flash`, `DominikBucko/qwen38-flash-next-2x3090`, `Dyluhn/R9V`,
`peonist-ai/halogen-flash-server`, `lEWFkRAD/qwen38-rtx-pro-6000`,
unsloth GGUF speed thread.

Local, read-only: GGUF tensor table via `gguf-py`; `ggml/src/ggml-backend.cpp`,
`ggml-cuda.cu`, `ggml-vulkan.cpp`, `src/llama-graph.cpp`, `src/llama-arch.cpp`,
`src/models/qwen4exp.cpp`, `common/arg.cpp` in `llama.cpp-qsagather` (base
`aa1b06d0c`); handoffs 9 and 10; `records/2026-08-31-flashnext-scanfix/RESULTS.md`;
`/tmp/fn-sweep/conf-qsagather-r3/`.
