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

## The 81.67 t/s figure is NOT reproducible

Re-running the identical build, split (50/50), context (98304), KV, ubatch and
sampler gave **46.27 t/s** (n=7). Peak VRAM matched the original within 350 MiB
(20,041 vs 20,388), so the configuration is right. The difference is the prompt:
RESULTS.md of 2026-08-22 describes it only as "agentic prompt (1958 tok of
repeated tool calls)" and that prompt was never saved.

Fixed going forward: `src/benchmarks/prompts/agentic_toolcalls.txt` (~1,985 tok,
seed 7, deterministic) is committed, and any tg claim must name its prompt file.

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

- `--spec-draft-type-k/v f16` vs `q8_0` (M3): null at all three depths
  (d0 42.10 vs 42.69, d60k 44.10 vs 44.10, d120k 41.28 vs 40.94).

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
