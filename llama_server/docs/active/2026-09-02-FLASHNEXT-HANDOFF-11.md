# Handoff 11: stop tuning knobs — go find the code wins

Supersedes the *action plan* of `2026-09-02-DECODE-IMPROVEMENTS.md` (its research
is still good; §1 below corrects it). Model: Qwen3.8-Flash-Next 125B-A6B
UD-Q2_K_XL. R9700 32,624 MiB (Vulkan1, gfx1201/RDNA4) + 7900 XTX 24,560 MiB
(Vulkan0, gfx1100/RDNA3, **drives the desktop**). Mesa 26.0.8 / RADV,
shaderc 2026.1, Ryzen 9 9950X3D, 60 GiB RAM.

---

## 0. READ FIRST

**No server was running at the end of this session.** GPUs were left at desktop
baseline (card0 57 MiB, card1 2,024 MiB). All benchmark jobs were killed.

Handoff 10 §0 still applies in full: compute memory fit before running anything,
the XTX can never hold more than ~47% of the 52 GB of weights, the VRAM guard
only arms *after* server readiness so it cannot stop a load-time spill, and the
XTX drives the display. `pgrep -a -x llama-server` first; **do not stop a server
you did not start.**

**`pkill -f <pattern>` matched this agent's own shell twice this session** (exit
144, killing the command mid-script). Kill by explicit PID from `pgrep -x
llama-server` / `ps -eo pid,cmd`, never by `-f` on a pattern that also appears in
your own command line.

---

## 1. THE USER'S DIRECTIVE — this is the whole point of the handoff

Verbatim: *"fuck these tests that require reading massive context entirely this is
way too long for knob tuning and the time is better spent seeing which llama cpp
PRs and forks have bigger wins for my hardware"*

and: *"under 200k context is unacceptable"*, and on an extrapolated VRAM estimate,
*"your calculation is pure speculation - unproved"*.

**Concretely:**

1. **Do not ship or propose a preset below ctx 200704.** Context is a hard
   requirement, not a tunable.
2. **Do not run more 199K-depth A/B sweeps to chase single-digit percentages.**
   A 199,340-token pass costs ~10 minutes of prefill; a 2-arm n=3 A/B is ~80
   minutes for what has repeatedly turned out to be noise.
3. **Do not present a fitted/extrapolated number as a result.** If it was not
   measured, say so in the same sentence.
4. **Go after algorithmic wins in upstream PRs and third-party forks.** §5 is the
   ranked list. That is the assignment.

---

## 2. What shipped

### Preset 16 — `ngram-mod` at ctx 200704: +19.3% decode at depth

**The one real win of the session, and it is at the required context.** No draft
model, no build, no root. Preset 14 plus `--spec-type ngram-mod`.

| depth | spec none | ngram-mod | decode |
|---|---|---|---|
| 10,776 (code) | 609.4 pp / 38.51 tg | 611.2 / 39.18 | +1.7% |
| 12,952 (prose) | 605.6 / 37.98 | 605.6 / 37.68 | −0.8% |
| **61,426 (code)** | 492.1 / **29.86** | 492.1 / **35.61** | **+19.3%** |

Prefill identical at 61K. Not noise — every ngram pass (32.00/35.61/36.13) beat
every control pass (29.73/29.86/29.96). It pays only at depth because it
amortizes the per-token PLE lookup / QSA indexer / mask-build costs that grow
with n_kv.

**LIMIT: validated to 61,426 tokens only.** ngram-mod costs +439 MiB of XTX
VRAM; against preset 14's measured 23,983 MiB at 199K that projects to ~24,422
versus a 24,200 ceiling and 24,560 physical. **That is an extrapolation.** To
clear full depth, run the ngram arm on `deep199k.txt` and read card1 at depth;
if short, lower `--tensor-split` toward 390,610 or shrink
`--spec-ngram-mod-n-max`. Reverts instantly by dropping `--spec-type`.

`src/server/common.sh` was extended: `ngram-mod` is now a valid
`LLAMA_SPEC_MODE` (the validator rejected it before). Note `default.sh` forces
`LLAMA_SPEC_DRAFT_N_MAX=2` while the benchmark ran llama.cpp's default of 3 — the
preset pins 3 to match. **Always confirm a new preset with
`llamactl command --profile <path>` before trusting it**; that is how this was
caught.

### Preset 15 — the open question

**Preset 15** — `config/profiles/preset-flashnext-q2kxl-vulkan-conv-65k-ub1024.sh`,
registered as launcher menu entry 15. ctx 65536, ub1024, otherwise identical to
preset 14.

Measured, `llama.cpp-qsagather`, ts 400,600, temp 1.0/k20/p0.95, seed 42,
medians of 3:

| | preset 14 | preset 15 |
|---|---|---|
| pp @15,375 tok | 605.7 | **973.2** (+60.7%) |
| tg @15,375 tok | 38.20 | 38.11 (flat) |
| pp @61,426 tok | — | 723.2 |
| tg @61,426 tok | — | 29.53 |

It is real and validated at a filled context (XTX 22,940 at 61K, 1,260 MiB under
the guard ceiling; VRAM does **not** grow with depth here because compute buffers
size at load).

**But the user has said under 200K is unacceptable.** So preset 15 is at best an
optional extra entry, and possibly should be removed. **Ask before doing more
work on it.** Preset 14 remains the production config. Nothing about preset 14
was changed, and no worktree was modified.

---

## 3. Measured this session — all of it

Depth note: `/tmp` had been wiped, so prompts were regenerated from
`ggml-vulkan.cpp`. Depths are **10,776 / 15,375 / 61,426 / 199,220** tokens here,
versus 11,568 / 199,340 in the August records. Compare within this document.

### PR #28213 (mask-free QSA gather) at 10,776 tokens — it LOSES

Same-binary A/B in `llama.cpp-qsa28213` via the `QWEN4EXP_QSA_GATHER` env toggle,
so build/allocation/prompt/sampler/seed are identical by construction.

| arm | pp | tg |
|---|---|---|
| `QWEN4EXP_QSA_GATHER=0` (masked scan) | **615.19** | **40.11** |
| gather on (#28213) | 612.15 | 38.85 |
| | −0.49% | **−3.14%** |

**The 199K arm was started but killed before completing** (user directive). Partial
control-arm data only, at 199,220 tokens: warmup 343.69 pp / 17.59 tg, one measured
pass 362.65 pp / 17.82 tg. **The gather arm never ran, so #28213 is UNRESOLVED at
depth.** Its author reports +6% @31K, +19% @62K, +50% @130K on CUDA at temp 0.
The worktree is built and ready if anyone wants to finish it.

### `--threads` — the EngramHalo +17% does not transfer

ctx 200704, ub192, 10,776 tokens, production binary.

| `-t` | pp | tg | within-arm tg spread |
|---|---|---|---|
| 4 | 611.43 | 39.57 | 0.95 |
| 8 | 608.49 | 38.89 | 0.71 |
| 16 (auto) | 608.05 | 38.97 | 0.58 |

Between-arm spread (1.7%) equals within-arm spread. **Mechanism explains it:**
Strix Halo is UMA, so oversubscribed CPU threads doing PLE gathers steal bandwidth
from the GPU. Here the PLE gather runs on separate DDR5 while both GPUs work out
of their own VRAM. No contention to relieve. Closed.

### ubatch at ctx 65536, 15,375 tokens

| ub | pp | tg | XTX MiB |
|---|---|---|---|
| 192 | 605.7 | 38.20 | 21,774 |
| 512 | 814.7 | 38.47 | 22,241 |
| 1024 | **973.2** | 38.11 | 23,233 |
| 2048 | killed during load | — | projected ~25,220 |

Decode is flat across the whole sweep. **ubatch is a pure prefill lever.**

### Why ubatch cannot help at ctx 200704 — this is measured, not fitted

**The ubatch VRAM cost scales with n_kv**, because the mask and compute buffers
are sized `ub x n_kv`:

- ctx 65536: 1.46 MiB/unit (192->512), 1.94 MiB/unit (512->1024) — measured here
- ctx 200704: **2.96 MiB/unit** (192->384) — from handoff 10, where ub384 was
  **guard-killed at XTX 24,552** against a 24,200 ceiling at 199K depth

At ctx 200704 filled to 199K the XTX sits at 23,983 with ~217 MiB of headroom.
Handoff 10 measured ub256 at 349.44 pp versus ub192's 348.09 — **+0.4%, nothing.**
`--tensor-split` does not rescue it: moving 4% of the 47,322 MiB of GPU-resident
weights off the XTX puts the R9700 at ~32,457 against its 32,500 ceiling, and
**both cards pay the ubatch cost.**

**Conclusion: at a filled 200K there is no flag-level prefill win left.** Both
cards are at capacity. Anything further must come from code. This is the single
most important result of the session.

---

## 4. Closed — do not spend time re-deriving these

| lead | why it is closed |
|---|---|
| **PR #28244** "attend the selected cells instead of masking the window" | **Already in our tree.** It is `d7ca8ea` reopened standalone — identical `build_qsa_gather`/`build_qsa_scan`/`qsa_mask_sel` at `qwen4exp.cpp:677-787`. Already worth +1.07%. |
| **PR #25494** Vulkan FA dequant-once for q8 KV prefill | **Already in our tree** (`pipeline_dequant_transpose`, `dequant_q8_0_transpose`). Merged 11 Aug, before our base `aa1b06d0c`. |
| **Valve/Rhys Perry RADV "CU mode when LDS is used"** (+13% pp512) | Landed in **Mesa 25.3**; we run **26.0.8**. |
| **glslc 2023.8 shader regression** (discussion #21043) | **Not us.** We build with shaderc **2026.1**, Vulkan headers 1.4.341. Verified in `build-vulkan/CMakeCache.txt`. |
| `apollosenvy/kernel-anvil` "2x decode on 7900 XTX" | **HIP/ROCm MMVQ only**, and its 2.25x baseline was confounded with TurboQuant KV overhead (a 12 tok/s baseline for a 27B on a 7900 XTX is not credible). Our ROCm path already loses badly (310 pp / 10.5 tg vs 348 / 18.6 Vulkan). |
| `--threads` | §3. |
| ubatch / tensor-split at ctx 200704 | §3. |

---

## 5. THE ASSIGNMENT — ranked research targets

Source for 1-3: **llama.cpp Discussion #21043, "RDNA4 Llama Experiments —
Squeezing Every Token/s from the R9700"** — 50+ manual experiments on the user's
exact R9700, Vulkan/RADV. This was not in the earlier brief and is the single
most relevant document found. Read it in full before acting.

### 1. Free system-level settings, never tested here (do these first)

Zero build, zero VRAM, minutes to test — and they apply at **full 200K context**,
which is the constraint that killed everything else.

| setting | reported | our current state |
|---|---|---|
| `pcie_aspm.policy=performance` (kernel cmdline) | **+10.8% decode** on RDNA4/RADV | **`[default]` — NOT set.** Verified `/sys/module/pcie_aspm/parameters/policy` |
| `power_dpm_force_performance_level=high` | +14% TG, **−9% PP** | both cards `auto` |
| `rocm-smi --setperflevel auto` | reported optimal overall; idle ~20W vs 50W | already `auto` |
| LACT `COMPUTE` profile | +4-5% TG | not installed |

Also worth noting: **the XTX is on PCIe 4.0 (16 GT/s x16) while the R9700 is on
PCIe 5.0 (32 GT/s x16).** Decode under `--split-mode layer` is pipeline-parallel
and pays a cross-card transfer every token, so the ASPM and link-state findings
have a plausible mechanism here, unlike the threads finding.

### 2a. DFlash and DSpark are NOT available for this model — verified in source

Asked directly, so settled definitively. `common/speculative.cpp:2341` in our own
tree auto-detects the speculative type from the **draft** GGUF:

```c
const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
if (arch != "dflash") {
    // falls through to: blk.<n-1>.nextn.eh_proj.weight present -> DRAFT_MTP
    ...
    return {};
}
// the Markov head distinguishes draft-dspark from draft-dflash
const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
```

DFlash/DSpark require a **purpose-trained sidecar whose `general.architecture` is
literally `dflash`**, plus `markov_w1`/`markov_w2` for DSpark. PR #25173 states
*"only Qwen3 backbones are supported for now"*; DeepSeek released draft weights
for Qwen3 4B/8B/14B and Gemma4, and z-lab's DFlash2 is for **Qwen3.8-27B dense**.
**No `dflash` draft trained against Qwen3.8-Flash-Next exists.** The widely-cited
"DFlash 2 + n-gram, 65 -> 305 tok/s (4.68x)" result is that dense 27B on an
RTX PRO 6000, not this 125B MoE.

**So for Flash-Next the speculative options are exactly two: MTP and n-gram.**
Flash-Next natively ships a 4B MTP layer (part of its 177B total), and MTP draft
heads exist on HF (`dzannotti/Qwen3.8-Flash-Next-MTP-GGUF`,
`quimmedes/Qwen3.8-Flash-Next-MTP-GGUF`). Do not spend time hunting for a DFlash
sidecar.

### 2b. MTP on MoE — the two data points are different regimes, not a contradiction

#21043 on RDNA4:

| model | baseline | MTP-3 | gain |
|---|---|---|---|
| Qwen3.6-27B **dense**, single R9700 | 28.2 | 47.4 | **+68%** |
| Qwen3.6-35B-A3B **MoE**, dual R9700 | 83.2 | 87.8 | **+5.5%** |

*"Dense models see 60%+ gains; MoE models see 5-10% due to lower activated
parameter ratio and dispatch overhead."*

**Do not read that as "MTP is dead for us" — the regimes differ.** That MoE row is
a 35B-A3B at an **83 t/s** baseline: dispatch-bound, with little per-step overhead
left to amortize. Ours at 200K runs **18.6 t/s**, and what dominates there is
precisely what speculation amortizes across accepted tokens — the CPU-side PLE
n-gram lookup, the QSA indexer, and the per-step mask build. That is the
mechanism behind EngramHalo's MTP gains on *this exact model* **growing with
depth**: +47% @0K, +63% @78K, >2x @156K.

The two published results are therefore consistent, and our regime is the
favourable one. The genuine counter-evidence is our **own** controlled temp-1.0
test at 11.5K (MTP losing by 21%) — which is the shallow end, where it should
look worst. **MTP at depth under production sampling has never been measured
here.** It remains the largest untested decode lever.

**PR #28243** (new, 2026-09-02) adds `--mtp-shared-embd`, letting the draft reuse
the target's embeddings and LM head, which may invalidate the brief's "draft does
not fit at ctx 200704, needs ctx <= 163840" constraint — important, since ctx
200704 is now a hard requirement. Draft heads exist:
`dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` (a draft head, used with `-md` beside any
Flash-Next target) and `quimmedes/Qwen3.8-Flash-Next-MTP-GGUF`.

If tested: 4 arms, temp 1.0 / top_k 20 / top_p 0.95, n=3, acceptance reported
beside tok/s, at depth. Every published number for this model is greedy, and
handoff 10 §4.3 explains why greedy flatters speculation.

### 2c. `ngram-mod` — TESTED AND SHIPPED as preset 16, see 2

Model-agnostic, works with the current build today, no sidecar and no download.
The brief's honest isolated figure on real code is **24.7 -> 27.8, +12.6%**
(`--spec-type ngram-mod`, `--spec-draft-n-max 4`, `--spec-draft-p-min 0.75`,
`--spec-ngram-mod-n-match` at its default 24).

**Beware the measurement trap** documented by EngramHalo: after a degenerate or
repetitive prompt the n-gram speculator reaches ~100% acceptance and decode reads
2-3x high (73 tok/s observed); repeating an identical request reads ~35% high.
Measure on genuinely fresh text.

### 3. Dual-GPU is a decode liability, confirmed independently

#21043 on dual R9700: decode tg128 **−26% MoE / −16% dense versus a single card**
(dispatch overhead), while prefill pp2048 **gains +34-55%**. This matches handoff
10 §5's "structural ceiling" from a completely independent setup.

**Implication worth taking seriously:** the second GPU is helping prefill and
*hurting* decode. Nobody has measured what this model does on the R9700 alone —
it cannot hold 46.2 GiB of weights, so it is not directly testable, but it
reframes the split as a prefill/decode tradeoff rather than a pure win.

### 4. Un-upstreamed Vulkan MoE prefill work — the real prefill ceiling

Since flags are exhausted at 200K (§3), this is where prefill gains have to come
from. All target `mul_mat_id` / MoE prefill, which is our architecture.

- **`Nathanw1014/strix-halo-llamacpp`** — MoE row-list prepass, branch
  `vulkan-mmid-rowlists`; also FA GEMM2 P-load hoist (`feat/fa-p-hoist`) and f16
  KV contiguize. Reports 71.9 -> 191 pp at 64k on Qwen3-Coder-30B (2.66x), though
  that figure is stock-f16 vs fixes+q8-KV, so it is **not** an isolated measure of
  the prepass.
- **`LaurentZuijdwijk/llama.cpp`** — tiled concat-transpose, f16 B operand for
  `mul_mat_id`, LDS stride padding. 1,648 vs 870 t/s pp2048 on a 35B-A3B (**1.9x**).

**Both are gfx1151 (Strix Halo) LDS-tuned and driver-gated, and both touch the
same Vulkan files our fusions modify.** This is the riskiest port available and
the most likely to be a real win. Preserve `llama.cpp-qwen4exp-vulkan-fusions`
(the untouched source of truth) and A/B in one binary.

### 5. PR #28136 — direct PLE reads, `--lazy-mode on-direct`

Promoted from the brief's rank 7: it now has one approval, and a Xeon E5-2620 +
RTX 3090 tester reports **+37% cold and +25% warm** prefill. The brief recorded
warm as a no-op (−0.4% to +2.1%), so the warm claim is contested. Our PLE is
26.8 GiB. Cheap proxy available with no build: **`--load-mode none`** (needs
27.2 GiB anonymous RSS — impossible when the brief was written at 706 MB free,
but there was ~33 GB free this session). Untested; a script for it exists at
`scratchpad/decodelevers.sh`.

### 6. Also unresolved

- **#28213 at depth** (§3) — worktree built, gather arm never ran.
- **`rm_kq=1`**, a one-line change in #21043 worth +13% decode on *dense* under
  *AMDVLK* (+2% on MoE). Probably not ours, but it is one line.
- **Mesa**: #21043 reports a custom **Mesa 25.3.6** build giving +48% prefill
  pp2048 on 35B MoE/RADV. We run 26.0.8. Whether 26.x kept that win is unknown.

---

## 6. Methodology that held up

1. **Same-binary env-toggle A/B is the only clean comparison.** `QWEN4EXP_QSA_GATHER`
   made build/allocation/prompt/sampler/seed identical by construction. Two earlier
   wrong conclusions in this project came from confounded cross-build comparisons.
2. **A fresh ubatch shape needs 2 warmup passes, not 1.** ub512's first measured
   pass read 629.9 against 814.7 for the next two — shader compilation for a shape
   the binary had not run. Medians absorbed it; a single-shot number would not.
   Set `FN_WARMUP_PASSES=2` when changing ubatch.
3. **Validate at a filled context, never a shallow one.** VRAM behaviour differed
   in *direction*: at ctx 65536 the XTX *fell* from 23,233 @15K to 22,940 @61K,
   while at ctx 200704 it *rose* from 23,226 @10.8K to 23,983 @199K.
4. **`tee | grep` buffers.** `/tmp/<run>.out` looked empty for 10 minutes while the
   run was fine. Read the raw `tee` target (`/tmp/fn-*.log`) for live progress.
5. Handoff 10 §4's rules all still apply, especially: GTT is not a spill signal on
   Vulkan (the tell is pp/tg collapsing), and never record MTP metrics at temp 0.

---

## 7. Files touched

**Added**
- `config/profiles/preset-flashnext-q2kxl-vulkan-conv-65k-ub1024.sh` (preset 15 — see §2, may want removing)
- `config/profiles/preset-flashnext-q2kxl-vulkan-ngram-200k.sh` (**preset 16 — the shipped win**)
- `records/2026-09-02-flashnext-optimize/RESULTS.md` (full numbers)
- `records/patches/p28213/` (`INFO.txt`, `tracked.patch`, untracked shaders)
- this handoff

**Modified**
- `bin/llama-launch` — menu entries 15 and 16 + case arms; prompt now "Preset (1-16)"
- `src/server/common.sh` — `ngram-mod` accepted as `LLAMA_SPEC_MODE` + case arm

**Unchanged:** every worktree, preset 14, `bin/flashnext_sweep.sh`.

Scratchpad drivers (throwaway, in
`/tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/95d61afc-.../scratchpad/`):
`common.env`, `ab28213.sh`, `threads.sh`, `prefill.sh`, `depth64k.sh`,
`deep28213.sh`, `ctxsweep.sh`, `decodelevers.sh`. Prompts regenerated into
`/tmp/fn-prompt/` (`d10k` 10,776 tok, `d16k` 15,375, `d64k` 61,426, `deep199k`
199,220) — `/tmp` is wiped on reboot, regenerate with `head -c N ggml-vulkan.cpp`.
