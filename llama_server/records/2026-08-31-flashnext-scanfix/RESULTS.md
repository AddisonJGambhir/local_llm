# Flash-Next deep-context speed: the PR 27977 verdict and the PLE n-gram scan fix

Date: 2026-08-31. Supersedes the analysis in handoff 9.
All measurements use ONE identical prompt of **199,340 actual tokens**
(`/tmp/fn-prompt/deep199k.txt`, the first 611,800 chars of `ggml-vulkan.cpp`),
ctx 200704, split mode layer, tensor-split 400,600, ub128, batch 2048,
K/V q8_0/q8_0, flash-attn on, no speculation, all experts in VRAM,
`per_layer_token_embd.weight` + `token_embd.weight` on CPU.

## 1. Methodology correction

Handoff 9's tg figures were measured over **7 to 11 generated tokens**, because
`ignore_eos` was false and the model emitted EOS almost immediately. Those
numbers are not a usable decode measurement. Every run below uses
`ignore_eos: true` and `n_predict: 128`.

## 2. The same-prompt A/B

| build | pp t/s | tg t/s | tg sample |
|---|---|---|---|
| shipped preset 12 (fusions worktree) | **343.17** | 12.58 | 11 tok |
| PR 27977 complete (stock upstream base) | 167.96 | 14.26 | 7 tok |
| PR 27977 minus `618450d` indexer slices | 168.26 | **14.333** | 128 tok |

Run dirs: `/tmp/fn-sweep/baseline-fusions-layer-depth199k-sameprompt-*`,
`/tmp/fn-sweep/pr27977-layer-depth199k-*`, `/tmp/fn-sweep/noslices-d199k-*`.

## 3. Two conclusions from handoff 9 were wrong

**(a) `d7ca8ea` (QSA gather) is not the prompt-speed regression.** The gather is
gated on `4*n_tps*width < n_kv`. This model has `indexer_top_k = 2048` and QSA
compress ratio 4, so `width = 2051`. At ub128 that is
`4 * 128 * 2051 = 1,050,112`, which never drops below `n_kv <= 200,704`.
The gather therefore **cannot fire during prefill at all**; it is decode-only,
exactly as its author designed it.

**(b) The A/B was confounded.** The "baseline" was the *fusions* worktree, which
carries local Vulkan work absent from upstream (mat-vec epilog fusion, fused
REPEAT+MUL+ADD, skinny inject prefill path, radix TOP_K). The PR build was stock
upstream. The 343 -> 168 pp gap measures **fusions vs no fusions**, not PR 27977.

Removing the suspected commit moved prompt speed by 0.2% (167.96 -> 168.26),
which confirms this directly. The local Vulkan fusions are worth roughly **2x
prompt speed** at this depth and must be preserved in every future build.

## 4. The real lever: the PLE n-gram predecessor scan

`llama_kv_cells::for_each_token_in` is reached from `get_prev_tokens`, which is
called by **`build_ple`** in `src/models/qwen4exp.cpp`. This is the per-layer
embedding n-gram lookup, i.e. **part of the model architecture**, not
speculative decoding. It therefore runs on every decode step even with
`--spec-type none`.

The old loop walks all `LLAMA_MAX_SEQ = 256` sequences for every used cell whose
`m.none()` test passes. With one sequence every used cell passes, so at 199K
depth this is roughly **51 million iterations per generated token**, against a
79 ms/token budget at 12.58 tg.

Upstream fix, **already merged as `62acc89c2` (PR 28011)**, stops the scan once
the cell's own sequences are seen. Author's figures on this exact model:
55k 56.3 -> 74.3 t/s, 132k **33.6 -> 50.9 t/s**. Prompt speed unchanged.

**The fusions worktree (base `aa1b06d0c`) predates this merge and does not have
it.** It is 56 commits behind master and also lacks `2d8d612e4` (PR 27991,
non-contiguous cell restore, which matters for speculative rollback).

## 5. Builds produced

Original `llama.cpp-qwen4exp-vulkan-fusions` was **not modified**; its 7 dirty
files are intact. Two new worktrees carry copies of those files byte-for-byte:

- `llama.cpp-flashnext-fusionscan` — fusions + `39efb639` (= merged 28011)
  + `28e37926` (scan only the n-gram window) + `b88668b` (used-cell bitmap).
  Intent: keep 343 pp, gain the decode win.
- `llama.cpp-flashnext-full` — the fusions ported **forward** onto PR 28104's
  base (which already contains merged 28011), plus the same two extra scan
  commits, plus the complete NextN/MTP stack. All 5 fusion files and both new
  shaders applied cleanly.

## 6. MTP is viable on VRAM after all

The `MTP-Q4_K_M` sidecar is 2.44 GiB, but it decomposes as:

- 1.75 GiB MoE experts (`blk.48.ffn_down/up/gate_exps`)
- 0.67 GiB `token_embd.weight` + `output.weight` (341 MiB each)

Pushing the two embedding tables to CPU with `-otd` — the same treatment the
target already gets — leaves about **1.77 GiB** on GPU, which fits inside the
R9700's ~2.3 GiB of headroom at ctx 200704. The earlier "2.43 GiB does not fit"
conclusion assumed the whole file had to be resident.

PR 28104 is validated on **Vulkan/RADV + AMD + a Q4_K_M MTP head**, which is
this machine's exact configuration, and reports +50% decode at 70k. It bundles
the on-device checkpoint fix (also standalone as PR 28118) that explains the
earlier local MTP disaster: host round-trips of the full recurrent state cost
~600 ms of each ~825 ms speculative round.

## 7. Open / next

1. `fusionscan` at 199,340 tokens — expect ~343 pp retained plus the decode win.
2. `full` (fusions + scan + MTP) with
   `--spec-type draft-mtp -md <sidecar> -otd 'token_embd\.weight=CPU,output\.weight=CPU'`.
3. PR 28040 (O(log n) `get_prev_tokens`) — a further +9% decode upstream.
4. PR 27973 (Vulkan GDN cache-write fusion) — removes 48 CPY nodes per token.

Do not stack PR 27992 with 27977; the review thread reports the pair regresses
decode.

---

# Part 2: the 10K-depth bake-off (2026-08-31, evening)

Fast-iteration depth per user instruction. Prompt `/tmp/fn-prompt/d10k.txt`,
**11,568 actual tokens**. Context allocation kept at the shipped 200,704 so the
memory layout matches preset 12; only the depth changes. 1 warmup + 3 measured
passes, `ignore_eos`, `n_predict 128`, medians reported.

## Baseline to beat (preset 12, ub128)

| metric | median | min | max |
|---|---|---|---|
| pp | **599.37** | 596.97 | 601.39 |
| tg | **38.55** | 37.75 | 38.81 |

VRAM: card0 30,311 MiB, card1 23,460 MiB.

## Builds, all at ub128

| build | pp | tg | verdict |
|---|---|---|---|
| preset 12 (baseline) | 599.37 | 38.55 | — |
| fusionscan (fusions + scan fixes) | 597.69 | **41.40** | tg +7.4%, pp tied |
| ppfix (fusionscan + PR 28023 slices) | 598.62 | 40.93 | slices are noise on Vulkan |
| full (fusions fwd-ported to new master) | 589.55 | 40.70 | no better than fusionscan |

Two findings:

- **The scan fix is worth +7.4% decode even at 11.5K**, where upstream says it
  should be nearly invisible. pp is unchanged, exactly as upstream claims.
- **PR 28023 (indexer head slices) is neutral on Vulkan/RADV**: 598.62 vs 597.69
  pp is inside the noise band. This matches the PR author's own follow-up
  comment that outside very fast CUDA parts it is "mostly a readability win".
  It is *not* the deep-context prompt regression the earlier handoff blamed.

## ubatch sweep on fusionscan (the prompt-speed lever)

pp is GPU-bound here, not CPU-bound, so no source change moved it. ubatch did:

| ub | pp | tg | card0 | card1 |
|---|---|---|---|---|
| 128 | 597.69 | 41.40 | 30,311 | 23,487 |
| 192 | 610.29 | 41.15 | 30,564 | 23,754 |
| 256 | 633.83 | 41.16 | 30,857 | 24,009 |
| 384 | **744.35** | 41.38 | 31,438 | 24,032 |
| 512 | failed | — | — | — |

Guard ceilings are 32,500 (card0 R9700) and 24,200 (card1 XTX). ub384 leaves
about 1.1 GiB and 170 MiB of headroom respectively **at 11.5K depth only**.

## Status against the goal

At 11.5K, `fusionscan` at ub192/256/384 beats preset 12 on **both** axes.
This is not yet a shippable claim: preset 12 chose ub128 precisely because
ub256 migrated ~1 GiB into GTT late in a populated 200K prompt, twice. Shallow
VRAM readings cannot see that. Deep validation at 199,340 tokens is running for
baseline ub128, fusionscan ub192 and fusionscan ub256.

## MTP is blocked, not dismissed

`full` + `--spec-type draft-mtp` loads (the draft does fit once `token_embd` and
`output` go to CPU via `-otd`) but aborts on the first draft decode:

```
ggml-vulkan.cpp:7872: GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16
                                  || b_type == GGML_TYPE_Q8_1) failed
  ggml_vk_get_dequantize_mul_mat_vec
  <- common_speculative_impl_draft_mtp::process
```

The assert is upstream code, reached from the MTP draft graph. Whether the local
mat-vec fusion reroutes something into it is not yet established. Next step is a
fusion-free PR 28104 build to isolate it.

---

# Part 3: full-depth validation — the goal is met on decode

Same prompt, **199,340 actual tokens**, ctx 200704, split layer 400,600, q8_0 K/V,
flash-attn on, no speculation, all experts in VRAM, `ignore_eos`, 128 generated
tokens, single measured pass.

| config | pp | tg | card0 VRAM | migration |
|---|---|---|---|---|
| preset 12, ub128 (baseline) | 346.75 | 12.854 | 30,311 | none |
| **fusionscan, ub192** | **348.09** | **17.910** | 30,564 | **none** |

**Decode is +39.3% at real depth**, with prompt speed unchanged (348.09 vs
346.75 is 0.4%, inside single-pass noise — not claimed as a win).

The gradient across depth matches the mechanism exactly:

| depth | tg gain from the scan fix |
|---|---|
| 11,568 | +7.4% |
| 199,340 | **+39.3%** |

The waste removed scales with used cells, so the deeper the context the more it
pays. This is the single highest-value change found in this whole effort, and it
is a **merged upstream fix the local build simply predated**.

## Residency trace, fusionscan ub192

```
   depth  pp_band  c0_vram  c1_vram  c0_gtt  c1_gtt
   20480    468.1    30563    23735      48    1361
   81920    372.4    30563    23545      48    1290
  120832    315.1    30563    23567      48    1291
  161792    315.1    30563    23721      48    1210
  199398    255.7    30564    23711      51    1186
```

card0 VRAM is pinned at 30,563 MiB for the entire prefill; GTT *falls* rather
than grows. This is the signature of a genuinely resident run, and it is what
distinguishes it from the silent Vulkan KV offload that appears past ~80K.

## Correction to the shipped menu text

Preset 12 advertises `348 pp / 13.5 tg @d199k`. The 13.5 came from an
EOS-truncated 11-token sample. Measured over 128 tokens the same build gives
**12.854 tg**. The menu should quote the honest number.

---

# Part 4: ROCm 10 and ROCm.AI / Hyperloom

## Hyperloom is not applicable to this workload

Installed as `hyperloom-inference-optimizer==1.0.0` (venv `/tmp/fn-prompt/hlvenv`).
It is **not** usable here, for three independent reasons, the first of which is
explicit in its own source:

1. `hyperloom/inference_optimizer/cli/model_gate.py::_detect_gguf_only_checkpoint`
   **rejects GGUF-only checkpoints**: "checkpoint ships only GGUF weights
   (llama.cpp ...) with no HF safetensors/pytorch_model.bin weights; the default
   vLLM/sglang loader finds no model weights and fails in engine init."
2. It drives **vLLM and SGLang**, not llama.cpp. Its documented containers are
   `vllm-openai-rocm` and `sglang-rocm`. vLLM ROCm was already rejected on this
   machine because RDNA3 is not supported.
3. `gpu_type` defaults to `mi300x`; validation covers MI300X/MI325X/MI355X.
   gfx1100 / gfx1201 are not in the table.

It also requires ROCm torch and an Anthropic API key for its agent loop.

## ROCm 10 loader gotcha (important, reusable)

`--load-mode mmap` together with `-ot '...=CPU'` **hangs the ROCm build at load**:
VRAM fills to ~49 GB, then freezes with one core at ~95% and no log output.
Waited >13 minutes, never ready. The harness reports `status=load_timeout` with
an empty server log.

`--load-mode none`, identical otherwise: **ready in 25 seconds**.

llama.cpp prints the hint itself and it should be taken literally. Vulkan is
unaffected and the shipped Vulkan presets keep mmap.

## ROCm 10 vs Vulkan at 11,568 tokens

ROCm build: fresh worktree `llama.cpp-rocm10-fn` at master `458681e1d`
(has qwen4exp + merged 28011) plus 28e37926 and b88668b, `GGML_HIP=ON`,
`AMDGPU_TARGETS=gfx1100;gfx1201`. The pre-existing `llama.cpp-rocm10` worktree
was useless: it predates qwen4exp entirely (`strings libllama.so | grep -c
qwen4exp` = 0).

| backend | ctx alloc | pp | tg |
|---|---|---|---|
| Vulkan fusionscan ub192 | 200,704 | 610.29 | **41.15** |
| ROCm 10 ub192 | 32,768 | 613.57 / 608.15 | 25.50 |

Prompt speed is a tie at this depth; **Vulkan wins decode by ~61%**. Note the
ROCm run had a much smaller context allocated, which if anything favours it.

PR 28102 (gfx1201 flash-attention tuning, which targets the R9700 specifically)
conflicted on `ggml/src/ggml-cuda/fattn-mma-f16.cuh` and
`tests/test-backend-ops.cpp` against current master and was dropped rather than
hand-merged. It remains the one ROCm item aimed at this hardware and is worth
revisiting if ROCm ever becomes competitive here.

---

# Part 5: MTP under representative sampling — controlled A/B (2026-09-01)

GPT's handoff rejected MTP but never ran the control it said was required. This
is that control: **one build, one prompt, one allocation, one sampler, one seed**,
only the speculation config varying.

Build `llama.cpp-flashnext-full`, prompt 11,568 tokens, ctx 184320, ub192,
ts 400,600, q8_0 KV, `temp 1.0 / top_k 20 / top_p 0.95 / seed 42`,
`ignore_eos`, 128 predicted, 1 warmup + 3 measured, medians:

| arm | pp | tg | tg spread |
|---|---|---|---|
| A  no speculation | **613.39** | **40.92** | 0.55 |
| B  MTP, host sampling (n3/p0.7) | 380.12 | 33.76 | 8.12 |
| C  MTP, backend sampling (n3/p0.7) | 378.49 | 31.67 | 0.94 |
| D  MTP, backend sampling (n5/p0.5) | 377.89 | 33.90 | 6.12 |

**The no-spec control wins both axes: +62% pp, +21% tg.**

- `--spec-draft-backend-sampling` was the hypothesis for GPT's poor result
  (its command line omits it, and it defaults to *disabled*). It did **not**
  help: arm C is worse than B on tg, though far more stable.
- Attaching the draft costs ~38% prompt speed (613 -> 380) before speculation
  earns anything back. At this depth that is not recoverable.
- MTP tg variance under real sampling is large (spread 8.1 on arm B). Single
  measurements are not usable evidence here; medians of 3+ are the minimum.

Conclusion: GPT's rejection of MTP **holds**, now on valid evidence. The only
untested regime is deep context under representative sampling, where the per-step
cost MTP amortises is largest and the no-spec baseline falls to 17.9 tg.

# Part 6: PR 27973 Vulkan GDN cache-write fusion — no measurable benefit

Ported onto the fusionscan build as `llama.cpp-gdnfusion`. The upstream PR is a
**draft, never run on a Vulkan device by its author**, and contains a real defect:
`cache_buf` is initialised to `dst_buf` as a dummy and never bound to the actual
recurrent-cache buffer, while `state_out_off` is an offset *into that cache*.
The port here fixes it by carrying the CPY destination tensor in
`vk_gdn_fused_cache` and binding it with `ggml_vk_tensor_subbuffer`.

Correctness: greedy output differs from baseline — but the **baseline differs
from itself** run to run at temp 0 (GPU reductions are non-deterministic), so
divergence is not evidence of a defect.

A/B, 11,568 tokens, ctx 200704, temp 1.0/top_k 20/top_p 0.95, median of 3:

| build | pp | tg |
|---|---|---|
| preset 13 | 557.11 | 40.59 |
| preset 13 + GDN fusion | 534.68 | 40.46 |

Identical within noise. The 48 CPY nodes per token it removes are not a
bottleneck for this model on this hardware. **Not worth shipping.**

# Standing conclusion

Preset 13 (SCANFIX, ub192, ctx 200704, no speculation) remains the best known
configuration: **348.09 pp / 17.910 tg at 199,340 tokens**, fully resident.

Measured and rejected: MTP (real sampling), larger ubatch (dead at depth),
PR 28023 indexer slices (noise on Vulkan), ROCm 10 layer split (310 pp /
10.5 tg at depth), tensor split (requires f16 KV; loses), PR 27973 GDN fusion.

# Part 7: PR 28040 (O(log n) get_prev_tokens) — no gain over preset 13

Ported onto the fusions build as `llama.cpp-p28040`. PR 28040 *replaces* the
28011 scan with a sequence-position index lookup, so `llama-kv-cache.cpp` and
`llama-kv-cells.h` were reverted to base before applying it (it supersedes
28011 / 28e37926 / b88668b rather than stacking with them). One conflict in
`for_each_token_in`, resolved in favour of PR 28040's `seq_pos_tok_le`.

Real sampling (temp 1.0 / top_k 20 / top_p 0.95 / seed 42):

| depth | build | pp | tg |
|---|---|---|---|
| 11,568 | preset 13 | 626.06 | 41.55 |
| 11,568 | + PR 28040 | 619.13 | 40.82 |
| **199,340** | **preset 13** | **350.76** | **17.998** |
| **199,340** | **+ PR 28040** | 350.05 | 18.012 |

Dead even at depth and slightly behind shallow. PR 28011 (already in preset 13)
removed the dominant cost; the O(log n) refinement has nothing material left to
recover on this model. **Not worth shipping.**

Note preset 13 reproduces its shipped figure under representative sampling:
350.76 pp / 17.998 tg vs the recorded 348.09 / 17.910.

# Part 8: QSA gather (d7ca8ea) — small but real decode win → preset 14

Handoff 9 blamed this commit for a prompt-speed regression. That was wrong (see
Part 3): the gather is gated on `4*n_tps*width < n_kv`, so with width 2051 it
**cannot fire during ub192 prefill** (1.57M > 200704) and only runs at decode
(n_tps=1: 8204 < 200704). It was never tested on the fusions build until now.

`llama.cpp-qsagather` = fusionscan state + d7ca8ea. Deep, real sampling, n=3:

| build | pp mean | pp runs | tg mean | tg runs |
|---|---|---|---|---|
| preset 13 | 348.74 | 350.76 / 347.96 / 347.50 | 18.395 | 17.998 / 18.586 / 18.600 |
| **preset 14** | 347.83 | 348.32 / 347.62 / 347.55 | **18.591** | 18.443 / 18.660 / 18.671 |

tg **+1.07%**, higher in all three paired runs, with the two ranges barely
overlapping. pp **-0.26%**, inside noise (one pairing is a tie: 347.55 vs 347.50).

Shipped as preset 14; preset 13 left untouched as the reference.

# Part 9: knobs that did nothing (deep, real sampling)

| change | pp | tg | verdict |
|---|---|---|---|
| tensor-split 410,590 | 349.14 | 17.917 | flat |
| tensor-split 390,610 | 349.09 | 18.004 | flat |
| PR 28040 (O(log n) lookup) | 350.05 | 18.012 | flat |
| PR 27973 (GDN fusion) | — | 40.46 @11.5K vs 40.59 | flat |

Note `ts 410` was previously called "unsafe". It runs fine; its VRAM dipped and
GTT rose to 2,237 MiB mid-run while pp/tg held — more evidence that GTT movement
alone is not a spill signal on Vulkan. It simply offers no gain.

The tensor split is flat across 390-410, so the shipped 400,600 is already at a
local optimum.
