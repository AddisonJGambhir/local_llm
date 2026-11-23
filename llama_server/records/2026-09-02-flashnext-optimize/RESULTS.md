# Flash-Next optimisation pass — 2026-09-02

Acts on `docs/active/2026-09-02-DECODE-IMPROVEMENTS.md`, which was research-only.
Model: Qwen3.8-Flash-Next 125B-A6B UD-Q2_K_XL. R9700 (Vulkan1) + 7900 XTX (Vulkan0,
drives the display). Mesa 26.0.8 / RADV.

All runs: `flashnext_sweep.sh`, 1 warmup + 3 measured, medians, `ignore_eos`,
`n_predict 128`, temp 1.0 / top_k 20 / top_p 0.95, seed 42, `cache_prompt:false`.

`/tmp` had been wiped, so the prompt corpus was regenerated from `ggml-vulkan.cpp`.
Depths differ slightly from the August records (**10,776** tokens here vs 11,568
there) — compare within this document, not across.

---

## 1. Research refresh — what changed since the brief

| finding | status |
|---|---|
| **PR #28244** "attend the selected cells instead of masking the window" | **already in our tree.** It is `d7ca8ea` reopened standalone — identical `build_qsa_gather`/`build_qsa_scan`/`qsa_mask_sel` at `qwen4exp.cpp:677-787`. Already measured at +1.07%. Closed. |
| **PR #25494** Vulkan FA dequant-once for q8 KV prefill | **already in our tree** (`pipeline_dequant_transpose`, `dequant_q8_0_transpose`). Merged 11 Aug, before our base. This is why prefill is as good as it is. Closed. |
| **Valve/Rhys Perry RADV "CU mode when LDS is used"**, +13% pp512 | landed in **Mesa 25.3**; we run **26.0.8**. Already have it. Closed. |
| **PR #28136** direct PLE reads | now has `--lazy-mode on-direct` and one approval. Mostly cold-cache, but one tester (Xeon+3090) reports **+25% warm** prefill — the brief recorded warm as a no-op. Re-opened as a lead. |
| **PR #28243** Qwen3.8-Flash-Next MTP + `--mtp-shared-embd` | **new, and it changes the brief's memory arithmetic.** Draft reuses the target's embeddings and LM head instead of duplicating them. +25% at 0.767 acceptance (RTX 5090+5060 Ti). |
| `LaurentZuijdwijk/llama.cpp` 1.9x MoE prefill; `Nathanw1014` MoE row-list prepass | un-upstreamed, **gfx1151 LDS-tuned**, driver-gated. Real but a risky port into our fusions tree. Not attempted. |
| `apollosenvy/kernel-anvil` "2x decode on 7900 XTX" | **HIP/ROCm MMVQ only.** Our ROCm path already loses (310 pp / 10.5 tg vs 348 / 18.6 Vulkan). Its 2.25x baseline was confounded with TurboQuant KV overhead — a 12 tok/s baseline for a 27B on a 7900 XTX is not credible. Rejected. |

---

## 2. PR #28213 (mask-free QSA gather) — shallow: it LOSES

Same-binary A/B in `llama.cpp-qsa28213` (fusions + scan + #28213), so build,
allocation, prompt, sampler and seed are identical by construction.
ctx 200704, ub192, ts 400,600, **10,776 tokens**.

| arm | pp | tg |
|---|---|---|
| `QWEN4EXP_QSA_GATHER=0` (masked scan) | **615.19** | **40.11** |
| gather on (#28213) | 612.15 | 38.85 |
| | −0.49% | **−3.14%** |

pp spread within arms was ~1.5 t/s, tg spread 0.29 / 0.44 — the decode loss is
outside noise.

This is the expected shape: the author's own gains start at ~31K (+6%) and only
reach +50% at 130K. A shallow loss does not settle it. Deep test in §4.

---

## 3. `--threads` — the +17% does NOT transfer

Production binary (`llama.cpp-qsagather`), ctx 200704, ub192, 10,776 tokens.

| `-t` | pp | tg | tg spread |
|---|---|---|---|
| **4** | **611.43** | **39.57** | 0.95 |
| 8 | 608.49 | 38.89 | 0.71 |
| 16 (auto) | 608.05 | 38.97 | 0.58 |

Between-arm spread (1.7% on tg) is the same size as within-arm spread
(0.58-0.95). `-t 4` is nominally ahead on **both** metrics, which is weak but
consistent evidence; call it **+0.5% pp / +1.5% tg, inside noise**.

**Why the EngramHalo +17% does not reproduce.** Strix Halo is UMA: CPU and GPU
share one memory bus, so oversubscribed threads doing CPU-side PLE gathers steal
bandwidth from the GPU. Here the PLE gather runs on separate DDR5 while both GPUs
work out of their own VRAM. The contention the finding relies on does not exist
on a discrete-GPU box. Mechanism absent, result null.

Harmless to adopt `-t 4`; do not expect it to show up.

---

## 4. The prefill win: ubatch at a conversational context

The 200,704 allocation is what caps ubatch at 192 — not prefill itself. Dropping
the allocation to **ctx 65536** frees VRAM on both cards and buys ubatch headroom.
Depth held at **15,375 tokens** (the 15-18K band real conversations sit in);
production binary `llama.cpp-qsagather`, ts 400,600.

| ub | pp median | vs ub192 | tg median | R9700 | XTX | XTX headroom to guard |
|---|---|---|---|---|---|---|
| 192 (control) | 605.7 | — | 38.20 | 28,726 | 21,774 | 2,426 |
| 512 | 814.7 | **+34.5%** | 38.47 | 29,182 | 22,241 | 1,959 |
| **1024** | **973.2** | **+60.7%** | 38.11 | 30,157 | 23,233 | **967** |
| 2048 | **aborted before load completed** | — | — | — | proj. ~25,220 | **negative** |

**Decode is flat across the sweep** (38.11-38.47, inside the 0.70-1.07 within-arm
spread). This is a pure prefill lever, exactly as predicted.

Measured ubatch VRAM cost on the XTX: **1.46 MiB/unit** (192->512), rising to
**1.94 MiB/unit** (512->1024). Extrapolating 1024->2048 gives ~+1,987 MiB, i.e.
**~25,220 MiB against 24,560 MiB of physical XTX** — not merely over the guard
ceiling but over the card. The guard only arms *after* readiness, so this would
have been a load-time spill on the GPU that drives the display. The arm was
killed during load; both cards returned to desktop baseline (57 / 1,209 MiB).

**ub2048 is not reachable at ctx 65536 on this hardware.** It is also not
reachable by shifting `--tensor-split`: moving 6% of the 46.2 GiB of weights off
the XTX (ts 400 -> 340) frees ~2.8 GiB there but adds the same to the R9700,
which at ub1024 already sits at 30,157 of a 32,500 ceiling. Both cards pay the
ubatch cost, so there is no split that fits it.

### The first measured pass understates a new ubatch shape

ub192: 606.0 / 604.9 / 605.7 — flat.
ub512: **629.9** / 814.7 / 814.7.
ub1024: **973.2** / 973.8 / 971.3 — here the cost fell in the warmup instead.

One warmup pass is not always enough to absorb the shader compilation for a
ubatch shape the binary has not run before. Medians handle it; single-shot
numbers on a fresh ubatch would not. Raise `FN_WARMUP_PASSES` to 2 when changing
ubatch.

### Validation at a filled context — the step that makes it shippable

A ctx-65536 preset is only real if it survives 65536 actual tokens. Prompt
`d64k.txt`, **61,426 tokens**:

| ub | pp median | tg median | R9700 | XTX at depth | vs ready |
|---|---|---|---|---|---|
| 512 | 628.1 | 29.65 | 29,182 | 22,962 | +478 |
| **1024** | **723.2** (+15.1%) | 29.53 | 30,081 | **22,940** | **−180** |

**VRAM does not grow with depth here.** ub1024 read 23,233 MiB at 15K and
**22,940** at 61K — compute buffers are sized at load from the context
allocation, not from cached tokens. 1,260 MiB of XTX headroom at full depth, and
GTT fell rather than rose. Nothing migrated.

This is the opposite of the failure mode the brief warned about, and it is only
knowable by filling the context. ub1024 ships.

---

## 5. What shipped

**New preset 15**, `preset-flashnext-q2kxl-vulkan-conv-65k-ub1024.sh`, registered
in `llama-launch` as menu entry 15. Same binary and same tensor placement as
preset 14; only `ctx` and `ubatch` differ.

| | preset 14 (deep) | **preset 15 (conversational)** |
|---|---|---|
| ctx | 200,704 | 65,536 |
| ubatch | 192 | **1024** |
| pp @15.4K | 605.7 | **973.2** — **+60.7%** |
| tg @15.4K | 38.20 | 38.11 (flat) |
| pp @61.4K | — | 723.2 |
| tg @61.4K | — | 29.53 |

Preset 14 stays the default for work past ~60K tokens; preset 15 does not
allocate enough context to hold a longer conversation.

**Nothing was changed about preset 14**, and no worktree was modified.

---

## 6. Open leads after this pass, re-ranked

1. **MTP with `--mtp-shared-embd` (PR #28243, new 2026-09-02).** The largest
   remaining decode lever, and the brief's memory arithmetic for it is now stale.
   The draft reuses the target's embedding tables and LM head instead of
   duplicating them, so the brief's "draft does not fit at ctx 200704, needs
   ctx <= 163840" may no longer hold. Reported +25% at 0.767 acceptance, mean
   accepted length 2.53 (RTX 5090 + 5060 Ti). Builds on #27836; our
   `llama.cpp-flashnext-full` worktree already carries the MTP stack plus the
   inject-swap fix.
   **Still test it the brief's way**: 4 arms, temp 1.0 / top_k 20 / top_p 0.95,
   n=3, at 11.5K / 78K / 156K, reporting acceptance beside tok/s. Every published
   MTP number for this model is greedy, and our own controlled temp-1.0 test at
   11.5K had MTP losing by 21%.

2. **PR #28136 direct PLE reads, `--lazy-mode on-direct`.** Promoted from the
   brief's rank 7. It now has one approval, and a third-party tester on a
   Xeon E5-2620 + RTX 3090 reports **+37% cold and +25% warm** prefill — the
   brief recorded warm as a no-op (−0.4% to +2.1%). If the warm gain is real it
   is a prefill win on a machine whose PLE is 26.8 GiB. Cheap proxy first:
   `--load-mode none` (tested in §7 below).

3. **The un-upstreamed Vulkan MoE prefill work.** `Nathanw1014`'s MoE row-list
   prepass (`vulkan-mmid-rowlists`) and `LaurentZuijdwijk`'s tiled
   concat-transpose + f16-B `mul_mat_id` (1,648 vs 870 t/s pp2048 on a 35B-A3B).
   Both are gfx1151 LDS-tuned and driver-gated, and both touch the same Vulkan
   files our fusions modify. Real, but the riskiest port available — attempt only
   with the fusions worktree preserved and a same-binary A/B.

**Closed by this pass:** PR #28244 (already in tree, = `d7ca8ea`), PR #25494
(already in tree), the Valve RADV CU-mode fix (already in Mesa 26.0.8),
`kernel-anvil` (HIP-only, confounded baseline), and `--threads` (§3).

---

## 8. Session end — corrections and unfinished work

**Ended early on user direction:** *"kill the job entirely ... this is way too
long for knob tuning and the time is better spent seeing which llama cpp PRs and
forks have bigger wins for my hardware."* All jobs killed; GPUs left at desktop
baseline (57 / 2,024 MiB). See `docs/active/2026-09-02-FLASHNEXT-HANDOFF-11.md`.

**#28213 at depth is UNRESOLVED.** The 199K control arm ran one warmup
(343.69 pp / 17.59 tg) and one measured pass (362.65 pp / 17.82 tg) at 199,220
tokens before being killed. **The gather arm never ran.** Nothing about #28213 at
depth was established. The worktree is built and toggle-ready.

**A correction to §4 of this document.** While proposing a higher-context version
of preset 15, this agent projected ctx 131072 / ub1024 at ~23,937 MiB XTX using
the ubatch cost measured at ctx 65536 (1.94 MiB/unit). That was wrong: **ubatch
VRAM cost scales with n_kv**, since the mask and compute buffers are sized
`ub x n_kv`. Handoff 10's measured 2.96 MiB/unit at ctx 200704 (ub384 guard-killed
at XTX 24,552) already showed this. A ctx-dependent fit puts ctx 131072 / ub1024
nearer ~24,480 — over the 24,200 ceiling. **The fit itself is a two-point
extrapolation and was not measured; treat it as unproven.**

**The load-bearing conclusion, which IS measured:** at ctx 200704 filled to 199K
the XTX has ~217 MiB of headroom at ub192, and handoff 10 measured ub256 at
349.44 pp against ub192's 348.09 — **+0.4%**. Both cards are at capacity and
`--tensor-split` cannot rescue it because both pay the ubatch cost.
**At a filled 200K there is no flag-level prefill win left; it has to come from
code.**

**Preset 15's status is open.** It is validated and real, but the user requires
ctx >= 200704, which preset 15 does not provide. It may belong in the launcher as
an optional entry or may want removing. Do not invest further in it without asking.

---

## 9. Vulkan MoE row-list prepass — 200K result, not shipped

Imported Nathanw1014's `vulkan-mmid-rowlists` commit (`ffe5cb4`) into the
isolated `llama.cpp-flashnext-rowlists` worktree at the Flash-Next base, then
added one local guard: row-list construction is used only when `nei1 > 1`
(batched prompt evaluation). Decode retains the existing count-only path. This
is necessary because the upstream/fork implementation ran a prefix/scatter
dispatch for every one-token decode step.

Build: Vulkan Release, `GGML_NATIVE=ON`, shaderc 2026.1. The targeted
`test-backend-ops test -o MUL_MAT_ID` suite passed all 872 cases independently
on the 7900 XTX and R9700.

All timing arms: same row-list binary, ctx **200704**, ubatch 192, batch 2048,
`--split-mode layer --tensor-split 400,600`, q8_0 KV, 199,220-token prompt,
temp 1.0 / top_k 20 / top_p 0.95 / seed 42, 128 generated tokens. One measured
pass per arm; the initial control attempt is excluded because its old 600-second
harness HTTP timeout canceled the still-running prompt evaluation at 122,880
tokens. Valid arms used `FN_CURL_TIMEOUT=1800`.

| arm | PP | TG | XTX VRAM | verdict |
|---|---:|---:|---:|---|
| row lists off (control) | 174.21 | **9.69** | 23,885 MiB | baseline |
| original row lists (also decode) | **179.06** (+2.8%) | 7.41 (-23.5%) | 23,444 MiB | reject |
| prefill-only row lists (`nei1 > 1`) | **178.61** (+2.5%) | 9.12 (-5.9%) | 23,358 MiB | reject |

All three arms reached the required 199,220-token depth inside the 32,500 MiB
R9700 and 24,200 MiB XTX guard ceilings; no guard action or Vulkan spill
occurred. The small PP gain does not justify the measured depth-TG loss, so no
production profile, launcher entry, or production binary was changed. The
experimental source and binary are retained for future integration with the
production QSA/fusion tree, where it must be re-measured as a same-binary A/B.

---

## 10. Direct lazy PLE reads — ported and gated, measurement pending

PR #28136's direct PLE-read implementation was ported into the isolated
`llama.cpp-flashnext-lazy-direct` worktree (commit `11a156c46`). The older
Flash-Next base still exposes `--tensor-read-lazy`, rather than the PR's newer
`--lazy-mode`, so the port adds the compatible value
`--tensor-read-lazy on-direct` without changing the existing `on`, `auto`, or
`off` behavior.

The direct path opens an independent buffered model descriptor, sorts and
deduplicates PLE row indices, uses parallel `pread()` calls, dequantizes to the
same F32 layout produced by `ggml_get_rows`, and supplies that as the existing
graph input. It is deliberately limited to Qwen4Exp's lazy PLE table; all
downstream Vulkan graph operations remain unchanged.

Verification completed before runtime benchmarking:

- Vulkan Release build of `llama-server` and `test-backend-ops` succeeded.
- `llama-server --help` exposes `--tensor-read-lazy on-direct`.
- Targeted `MUL_MAT_ID` backend correctness passed **872/872** cases on each
  of the 7900 XTX and R9700.

There is **no PP or TG result yet**. The required short model diagnostic, then
the same 199,220-token / ctx-200704 A/B against `--tensor-read-lazy on`, are
pending a free GPU window. On 2026-09-02 the production 200K `llama-server`
was running on both cards; it was not interrupted. This source candidate is not
integrated into the production QSA/fusion binary or its profile.

---

## 9. `ngram-mod` — the one deployed win, at ctx 200704

Found after the session was redirected. It needs **no draft model, no build, no
root**, and unlike every ubatch/context lever it works at the required ctx 200704.

DFlash and DSpark were ruled out first, from source: `common/speculative.cpp:2341`
requires a draft GGUF whose `general.architecture` is literally `dflash`
(`markov_w1.weight` distinguishes DSpark). Released weights exist only for Qwen3
4B/8B/14B and Gemma4; z-lab's DFlash2 is Qwen3.8-**27B dense**. No `dflash` draft
exists for Flash-Next. Anything else falls through to a `nextn.eh_proj` check for
MTP. **For this model the speculative options are exactly MTP and n-gram.**

All rows: `llama.cpp-qsagather`, **ctx 200704**, ub192, ts 400,600, temp 1.0 /
top_k 20 / top_p 0.95, seed 42, medians of 3, ignore_eos, n_predict 128.

| depth | prompt | spec none | ngram-mod | decode |
|---|---|---|---|---|
| 10,776 | code | 609.4 pp / 38.51 tg | 611.2 / 39.18 | +1.7% |
| 12,952 | prose | 605.6 / 37.98 | 605.6 / 37.68 | −0.8% |
| **61,426** | code | 492.1 / **29.86** | 492.1 / **35.61** | **+19.3%** |

**Prefill is identical at 61K (492.1 both).** Decode-only, as expected.

**The 61K result is not noise.** Every ngram pass beat every control pass:

```
ngram-mod : 32.00  35.61  36.13
none      : 29.73  29.86  29.96
```

Worst ngram exceeds best control by 6.8%.

**Why it only pays at depth.** Speculation amortizes the per-token costs that
grow with n_kv — the CPU-side PLE n-gram lookup, the QSA indexer, the mask build
— across every accepted token. At 10K those are a small share of a 38 tg step; by
61K decode has fallen to 29.9 tg and they dominate. Same mechanism that makes
EngramHalo's MTP gains grow with depth on this model.

The prose row matters: it is the honest control. n-gram acceptance inflates on
repetitive input, and the variance rises with it (tg spread 4.13 vs 0.23).

### The VRAM limit — measured, and it caps this at ~64K

ngram-mod costs **+439 MiB on the XTX** (23,815 vs 23,376 at 61K depth).
Preset 14's control measured **23,983 MiB at 199K**. 23,983 + 439 projects to
**~24,422** against a 24,200 guard ceiling and 24,560 MiB of physical XTX — about
138 MiB of physical margin on the card that drives the display.

**That projection is an extrapolation, not a measurement.** Shipped as
**validated to 61,426 tokens only**. To clear full depth, run the ngram arm at
`deep199k.txt` and read card1 at depth; if it does not fit, lower
`--tensor-split` toward 390,610 (the R9700 had 1,936 MiB spare) or shrink the
window via `--spec-ngram-mod-n-max` / `-n-min`, then re-measure.

### Shipped

**Preset 16** — `preset-flashnext-q2kxl-vulkan-ngram-200k.sh`, launcher entry 16.
Preset 14 plus `--spec-type ngram-mod`, everything else identical.

`src/server/common.sh` gained `ngram-mod` as a valid `LLAMA_SPEC_MODE` (it was
previously rejected by the validator) plus a case arm emitting `--spec-type
ngram-mod` and the optional `--spec-ngram-mod-*` knobs.

**Gotcha caught by dry-run:** `default.sh` forces `LLAMA_SPEC_DRAFT_N_MAX=2`, but
the benchmark ran bare `--spec-type ngram-mod`, i.e. llama.cpp's default of **3**.
The preset pins `LLAMA_SPEC_DRAFT_N_MAX=3` so the shipped config is the measured
one. Verified with `llamactl command`.

Speculation is a runtime flag — dropping `--spec-type` reverts instantly.
