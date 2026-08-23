# Overnight run 2026-08-23 — depth A/B, PR #27173, PR #25666

Qwen3.8-27B Q8_0, RX 7900 XTX + AI PRO R9700, Vulkan. Two rigs:

- **depth rig**: `-c 262144`, tensor 49/51, f16 KV, ub 1024, novel content built
  from llama.cpp sources, 2048-tok generations, temp 0 + top_k 1 + ignore_eos,
  7 reps/cell, exact permutation tests.
- **record rig**: the 2026-08-22 conditions — `-c 98304`, tensor **50/50**,
  f16 KV, ub 1024, `matrix.SHORT_PROMPT`, 2048 tok, temp 0 + top_k 1.

## Headline: the shipped adaptive config is not the fastest config

`draft-mtp n3` beats `draft-mtp-adaptive n_max 12` in BOTH regimes.

| regime | adaptive n12 | n3 | delta | p |
|---|---:|---:|---:|---:|
| depth 0 (novel) | 42.69 ± 0.75 | 42.10 ± 1.14 | +1.4% adaptive | 0.336 |
| depth 60k (novel) | 44.10 ± 2.29 | 42.27 ± 1.06 | +4.3% adaptive | 0.105 |
| **depth 120k (novel)** | **40.94 ± 1.02** | **42.19 ± 0.45** | **-3.0% adaptive** | **0.016** |
| record rig (`-c 98304`) | 46.27 ± 5.43 | **47.14 ± 1.39** | -1.9% adaptive | — |

The only significant depth cell says n3 wins, and it is the deepest one — the
regime the box actually runs in. Adaptive also carries 4x the run-to-run
variance (sd 5.43 vs 1.39) at c=98304.

## The 81.67 t/s figure: the NUMBER is unreproducible, the FINDING is not

Re-running the identical build, split (50/50), context (98304), KV, ubatch and
sampler with `matrix.SHORT_PROMPT` gives **46.27 t/s** (n=7). Peak VRAM matched
the original within 350 MiB (20,041 vs 20,388), so the configuration was right.
The prompt was the difference, and it was never saved -- RESULTS.md of
2026-08-22 describes it only as "agentic prompt (1958 tok of repeated tool calls)".

But the FINDING reproduces almost exactly. On a fully repetitive prompt
(acceptance ~0.99) adaptive beats n3 by the same margin the original reported:

| source | adaptive vs n3 at acc ~0.99 |
|---|---:|
| original RESULTS.md 2026-08-22 | +43.3% |
| ceiling probe (this run) | +42.7% |
| crossover mix100 (this run) | +43.7% |

Three measurements, two prompts, 1 point of spread. The original measurement was
sound. The mistake was generalising a degenerate-prompt result to the shipped
config.

**Mechanism.** At acceptance ~0.99 a 12-deep draft gets ~12 tokens accepted per
verify against n3's 3, so adaptive wins by ~43%. At acceptance ~0.6 most of the
deep draft is rejected and the wasted verify compute exceeds the gain, so n3
wins. Adaptive is not broken; it is tuned for a regime this box never runs in.

Measured endpoints, same box, minutes apart:

| content | acceptance | winner | margin |
|---|---:|---|---:|
| pure source code | 0.54 | n3 | +9.0% |
| technical prose (SHORT_PROMPT) | 0.62 | n3 | +1.9% |
| agentic tool calls | 0.59 | n3 | +8.6% |
| 120k novel code | 0.58 | n3 | +3.0% (p=0.016) |
| one sentence repeated | 0.99 | adaptive | +43.7% |

Fixed going forward: `src/benchmarks/prompts/` holds committed deterministic
prompts, and any tg claim must name its prompt file.


## THE RULE: adaptive vs n3 is decided by draft acceptance ~0.62

13 prompts, each measured with BOTH configs on the same rig (-c 98304, tensor
50/50, shipped binary). Sorted by adaptive's own acceptance, the winner flips
exactly once, with no misordering:

| adaptive acceptance | winner | margin |
|---|---|---|
| 0.538 - 0.619 (7 prompts) | **n3** | 1.8% - 8.3% |
| 0.627 - 0.990 (6 prompts) | **adaptive** | 0.9% - 43.7% |

> If draft acceptance on the workload is above ~0.62, use draft-mtp-adaptive.
> Below it, use draft-mtp n3.

**This box sits below the line.** Measured adaptive acceptance at 120k is 0.597
with the bench sampler and **0.549 with the production sampler**; F24's real
session figure was 0.442. So n3 is correct here, and the two findings compound:
temp 1.0 lowers acceptance (0.602 -> 0.549), pushing further into n3 territory.

### Two analyses that were WRONG on the way to this

1. **Prompt mix % as the x-axis.** Two prompt sets were built (concatenated, then
   interleaved) to sweep predictability. Both came out non-monotone, because what
   the model *generates* from a mixed prompt is itself mixed -- mix% only loosely
   controls acceptance. The cells are valid (acceptance, tg) observations; only
   the intended x-axis was broken.
2. **Regressing tg on acceptance pooled across configs.** This gave a crossover of
   0.406 and was retracted: acceptance is accepted/drafted, and n3 drafts 3 while
   adaptive drafts up to 12, so on identical content n3 reports 0.872 where
   adaptive reports 0.597. The two configs do not share an acceptance axis.
   Pairing by prompt is the valid comparison.

## PREFILL RECORD: +9.1%, and it is free

`--ubatch-size 1536 --batch-size 4096` on row split, measured twice:

| reps | pp | vs prior record 1240.47 |
|---|---:|---:|
| 3 | 1369.93 | +10.4% |
| 5 | 1353.06 | +9.1% |

Peak 13.5 GiB -- far under the ceiling. The prior record reproduced first
(1237.19 measured vs 1240.47 recorded at matching VRAM), so this is a real
improvement over a verified baseline, unlike the tg situation.

The pp curve is sharply non-monotone, which is why this was missed: every prior
sweep used powers of two and stepped over 1536.

| ub | b2048 | b4096 |
|---|---:|---:|
| 1024 | 1237 | (control pending) |
| **1536** | 1080 | **1370 / 1353** |
| 2048 | 950 | 1259 |
| 3072 | - | 1083 |
| 4096 | - | 938 |

## The production sampler costs 12.4%, not ~2%

One variable isolated at 120k depth, shipped binary, everything else identical:

| sampler | tg | acceptance |
|---|---:|---:|
| temp 0 / top_k 1 | 41.26 +/- 0.90 | 0.602 |
| temp 1.0 / top-p 0.95 / top-k 20 (production) | 36.15 +/- 1.52 | 0.549 |

Disjoint ranges, n=7 each (exact permutation p at its 1/3432 floor).

F26 estimated ~1.5-2% by timing the sampler call. End to end it is **12.4%**,
because there are two costs:
1. CPU sampling time (backend sampling is hard-disabled under -sm tensor)
2. **acceptance drops 0.602 -> 0.549** -- stochastic sampling diverges from the
   draft's greedy pick more often, so more speculative work is discarded

The second is specific to speculative decoding and was not previously accounted
for. It also raises the value of upstream discussion #24048 (backend sampling
under SPLIT_MODE_TENSOR) from ~2% to ~12% for this box.

This explains ~40% of the bench-vs-real gap (41.26 -> 36.15 vs ~27 observed).
The rest is content type and possibly grammar-armed requests disabling MTP.

## PR #27173 (chained MTP draft + lm-head mirror + sched pool): rejected here

Ported onto `llama.cpp-tp-chain` as a PR-only diff (NOT a merge — merging drags
766 commits of master in and reproduces the known-broken tensor rebase).
Port verified correct: with all knobs off the build is **bit-identical** to the
shipped one (same tg, same draft counts, same acceptance to 4 dp).

| knob | result |
|---|---|
| all off | 46.18 (baseline 46.27) — correct port |
| `LLAMA_SPEC_CHAIN=1` | **CRASHES**: `GGML_ASSERT(buffer)` in `draft()` |
| `LLAMA_META_MIRROR_OUTPUT=1` | 45.16 (-2.4%) |
| `+ LLAMA_SCHED_POOL=8` | 45.33 (-2.0%) |

The chain graph needs the `out_ids` / `n_outputs_max_per_seq` output API that
newer master added with the DFlash work. This fork does not have it (it was the
137-line "master API churn" dropped during the #27210 port). Reviving
`LLAMA_SPEC_CHAIN` means porting that API first — the chain carries most of the
PR's claimed +12.8%, so this is the one item still worth real effort.

## PR #25666 (no MMVQ in speculative steps on AMD): rejected, but diagnostic

| config | MMVQ on | MMVQ off | delta |
|---|---:|---:|---:|
| n3 | 47.14 | 38.51 | **-18.3%** |
| adaptive n12 | 46.27 | 42.60 | -7.9% |

The PR's acceptance claim holds (adaptive: 0.619 -> 0.640) but the speed cost
dwarfs it here. The asymmetry confirms the mechanism: n3's verify is always 4
rows, always under the PR's cutoff of 8, so it is always affected; adaptive's
deeper rounds exceed 8 and keep MMVQ.

### Side finding: MMVQ is the source of this box's acceptance nondeterminism

With MMVQ on, acceptance varies run-to-run at temp 0 with a fixed prompt
(n3 0.639-0.738, adaptive 0.522-0.720). With MMVQ off it is flat in both
(0.628-0.629 and 0.595 after settling). This is the mechanism behind the
long-standing "three runs, three different hashes" rule.

Only adaptive converts that acceptance noise into *tg* variance (sd 5.43 vs
n3's 1.39), because acceptance moves the controller's depth, which multiplies
the effect. n3 absorbs the same noise at fixed depth.

Practical use: `GGML_VK_DISABLE_MMVQ=1` buys a reproducible box for A/B work at
a known tg cost — worth it when a comparison would otherwise need n>=7 and a
permutation test to see through the noise.

## Null results

- (RETRACTED) `--spec-draft-type-k/v f16` was called null on n=7 data. At n=22
  it is **+2.0%, p=0.00535** (42.04 vs 41.24 at 120k). A null at low n is not a
  null. The n=7 cells showed +0.8% and were simply underpowered.

## No depth cliff on this box

Raw n3 tg is flat across depth: 42.10 -> 42.27 -> 42.19 from 0 to 120k on novel
content. The reported no-P2P collapse (9-11 t/s at 60-160k) and PR #26595's
"decode collapses at 127k" both concern the `p021/nc` mat-vec path, which is
gated on `ne[1] <= 65535`. We run `--flash-attn on`, so attention does not take
that path. The "depth cliff" framing should be retired for this configuration.

## Open: the 42 vs 27 t/s gap

Bench says ~42 t/s at 120k; real use showed ~27. Depth is now excluded as the
cause. Remaining candidates: content type, the production sampler chain
(temp 1.0 / top-p 0.95 / top-k 20 under CPU-side sampling), and grammar-armed
requests disabling MTP entirely. Sampler isolation is queued.

---

# FINAL: what to change (2026-08-23, session end)

## 1. PREFILL -- row and layer presets only: `--ubatch-size 512 --batch-size 8192`

| split | d0 | d65536 |
|---|---|---|
| row | 1237 -> 1647 (**+33.1%**) | 487 -> 791 (**+62.3%**) |
| layer 28/37 | 1241 -> 1648 (**+32.9%**) | 492 -> 805 (**+63.5%**) |
| **tensor 49/51** | **no effect** | **no effect** |

Row and layer land within 1% of each other at depth (independent cross-check).
Tensor is completely flat: all six ub/b combos within 2% (384-393 t/s).
Peak VRAM unchanged (16,408 MiB across all four layer configs).

**ubatch is the bigger lever at depth** (+43.7% from ub512 alone at d65k), batch
adds ~14% on top. At d0 the ordering reverses and batch dominates. That is why
prior sweeps missed it -- they tuned at d0, where ub looks unimportant.

## 2. DECODE -- tensor split, fixed draft depth beats adaptive

At 120k depth, novel content, temp 0, n>=15 per cell:

| config | tg | sd |
|---|---|---|
| n5 + q8_0 draft | **46.70** | 0.14 |
| n4 + f16 draft | 45.79 | 0.27 |
| n4 + q8_0 | 44.57 | 0.44 |
| n3 + f16 | 42.69 | 0.18 |
| adaptive n8 | 42.51 | 0.77 |
| n3 + q8_0 | 42.11 | 0.30 |
| adaptive n12 + q8_0 (**SHIPPED**) | 41.24 | 0.86 |

+13.2% for n5, p<0.00001, 6x tighter variance.

**CAVEAT, not yet resolved:** every one of these ran at temp 0. Per the user,
optimal n in real use is 2-4 because deeper drafts lose acceptance at high
temperature. Temperature was independently measured here to drop acceptance
0.602 -> 0.549 at n_max 12, and that effect should compound with depth. The
temp-1.0 n-sweep was staged (`harness/wave_temp.sh`) but NOT RUN. Treat n5 as a
temp-0 result only. Do not ship it without that sweep.

## 3. The adaptive-vs-fixed rule: draft acceptance ~0.62

13 prompts, both configs each, the winner flips exactly once with no misordering
(see the "THE RULE" section above). This box measures 0.549 with the production
sampler, i.e. below the line, so fixed depth is correct here.

## 4. Split-mode tradeoff (measured, incomplete)

Decode with MTP at 120k: tensor n3 42.11 vs row n3 36.27 (**tensor +16%**).
Raw unspeculated decode favours tensor by 13-18%, so MTP multiplies both
equally rather than closing the gap. But layer/row prefill is ~2x tensor's.
`layer_n3`, `layer_n5`, `row_n5` were not measured (session ended).

## Data-quality problems found in the existing records

1. **`MASTER_RESULTS.csv` Q8_0 vulkan-pr tensor 50/50 pp8192 d0 = 940.78 does
   not reproduce.** Both `tp-adaptive` and `vulkan-tp` measure ~396 t/s at the
   same peak VRAM (17,157 vs the recorded 17,192). Not a build regression --
   `vulkan-tp` IS the build it was recorded on. The layer row (1232.00)
   reproduces exactly (1240.65), so the rig is sound and this is one bad row.
   PRIME's research doc used 940.78 to conclude tensor prefill is "within 1.31x
   of layer"; the real same-build ratio is **row 1238 vs tensor 396 = 3.1x**.
2. **`llama-bench -ts` takes SLASHES, not commas.** `--tensor-split 49,51` makes
   llama-bench SWEEP the values 49 and 51 as separate single-value runs. Every
   tensor-split llama-bench attempt in this session used commas before this was
   found; they produced 24 GiB "does not fit" results and 28 t/s prefill, all
   artifacts. `llama-server` takes commas correctly, so no server result is
   affected.
3. **`llama-bench` has no `-c`.** Context is derived from `-p + -n + -d`, and
   `--fit-target` is not implemented for SPLIT_MODE_TENSOR.

## VRAM notes

- row split at `-c 262144` reaches **24,043 MiB** -- over the line. Use
  `-c 131072` (19,771 MiB) or lower for row at depth.
- llama-bench on tensor split sits at ~17.2 GiB; the server at tensor 49/51
  `-c 262144` peaks 22.9-23.3 GiB.
