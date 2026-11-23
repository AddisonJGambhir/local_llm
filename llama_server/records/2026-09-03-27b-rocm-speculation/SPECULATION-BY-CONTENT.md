# What speculative decoding actually buys, by content type and depth

Qwen3.8-27B UD-Q6_K_XL, ROCm 10, production preset (`-sm tensor 50,50`,
**ctx 262144**). Supersedes the tg figures in `RESULTS.md`, which were measured
with a wrong sampler at a shallow context on code only.

## Method (corrected on user instruction)

- **Sampler: Unsloth's official Qwen3.8-27B *thinking* mode** — temp **1.0**,
  top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repeat 1.0. This is
  what the preset ships (`LLAMA_REASONING=on`). Earlier runs used temp 0.7,
  which is Unsloth's *instruct* temperature paired with thinking's top_p — a
  config nobody recommends. **Low temperature inflates draft acceptance**: the
  identical config read 57.3 tg at temp 0.7 versus 49.9 at temp 1.0.
- **Depth: 10k minimum, 40k realistic.** A coding-agent harness reads 10k-40k of
  context before it does anything, so anything shallower is inadmissible.
- **Content: code AND prose, measured separately.** Code and tool calls accept
  far more draft tokens than prose; a code-only benchmark reports the best case
  as the average. Code = `ggml-vulkan.cpp`; prose = Gutenberg *Iliad*.
- 8 seeds per cell, `cache_prompt` so each prompt prefills once. Headline is
  **aggregate throughput** (total tokens / total generation time), not a median.

## Result

| prompt | no-spec | MTP n3 | acceptance | speedup |
|---|---|---|---|---|
| code 10k | 29.70 | **50.46** | 56.3% | **1.70x** |
| code 40k | 28.30 | **50.85** | 63.2% | **1.80x** |
| prose 10k | 29.72 | 33.15 | 25.2% | 1.12x |
| prose 40k | 28.29 | 28.02 | 19.7% | **0.99x** |

**Code and prose diverge with depth.** Code acceptance *rises* with context
(56.3 -> 63.2%) so the speedup grows to 1.80x. Prose acceptance *falls*
(25.2 -> 19.7%) until at 40k speculation is exactly break-even: the drafting
cost cancels the gain.

**The control validates the method:** without speculation, tg is
content-independent to two decimals — 29.70 vs 29.72 at 10k, 28.30 vs 28.29 at
40k. Every code/prose difference above is draft acceptance, nothing else.

**Practical reading:** MTP is a large win for agentic/coding work at realistic
depth and costs nothing for prose. Keep it on. *(Note: a lazy tool/structured
grammar disarms speculation entirely — see the `mtp-disabled-by-active-grammar`
note — so real tool-call turns may not see the code-path win.)*

## Variance — why 3 passes was never enough

At temp 1.0 speculative arms spread 5-24 tg across seeds while the no-spec
control spreads **0.06**. All the variance is draft acceptance. Worse, the
earlier runs used ONE fixed seed for all three passes and returned byte-identical
drafted/accepted counts every time — they measured timing jitter, not sampling.
Use ~8-10 distinct seeds and report aggregate throughput.

## MTP n-max: settled

User decision: *"MTP 3 is the default for this model as shipped by Qwen. MTP 3
is final."* For the record, 10-seed runs at temp 1.0 on code put n2/n3/n4/n5
within a few percent of each other, well inside their seed-to-seed spread —
there was never a real n-max win to find. The "+19.4% at n-max 5" reported
earlier was an artifact of temp 0.7 at ctx 32768.

## DFlash2 and DSpark: measured, rejected

Layer split (neither can run under `-sm tensor` — both abort in the Meta
Backend), ctx 262144, same sampler, 5 seeds. tg / acceptance:

| arm | code 10k | code 40k | prose 10k | prose 40k |
|---|---|---|---|---|
| **MTP n3** | **33.74** / 56.5% | **32.07** / 59.6% | **21.83** / 23.9% | **18.18** / 19.0% |
| DFlash2 n7 | 32.04 / 33.3% | 28.71 / 32.6% | 16.62 / 10.2% | 13.34 / 7.4% |
| DFlash2 n12 | 31.79 / 33.3% | 28.48 / 32.6% | 16.50 / 10.2% | 13.27 / 7.4% |
| DSpark n7 | **will not load** | — | — | — |

**MTP n3 wins every cell**, by 5-10% on code and 24-27% on prose. DFlash2's
acceptance is roughly half MTP's throughout. Its `n-max` knob is inert (n7 and
n12 produce identical accept counts) — the controller saturates.

This reverses the earlier finding that DFlash2 beat MTP by 11%; that was temp
0.7, ctx 32768, code only. **z-lab's advertised ~2x and "acceptance length
5.13-5.39" are low-temperature numbers.** The mechanism fits: DFlash2 drafts
whole blocks by diffusion, so at temp 1.0 the sampled path leaves the drafted
block far more often than a native MTP head does.

**CORRECTION — DSpark does fit.** An earlier version of this document said it
"will not load". That was wrong, and the reasoning was lazy: the model is
23.6 GiB and the draft 1.36 GiB, i.e. **25 GiB of weights against 56 GiB of
VRAM**. The OOM was a *placement* failure, not a capacity one — the draft's
compute buffer landed on `ROCm0`, the smaller display-driving XTX:

```
allocating 2288.08 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
```

**How to make it fit at ctx 262144:** `--spec-draft-device ROCm1` (draft onto the
32 GiB R9700) **plus q8_0 KV**. The failed allocation drops 2288 -> 1010 MiB from
the device flag alone; q8_0 KV then halves the dominant consumer and it loads.
Small `--tensor-split` nudges do NOT work — ts 45,55 hit the identical 1,010 MiB
wall and ts 48,52 was worse (1,981 MiB), the layer-boundary rounding the tensor
preset already documents.

**With it running, DSpark is the worst of the three.** Like-for-like, all at
q8_0 KV, layer split, ctx 262144, 5 seeds:

| arm | code 10k | accept | prose 10k | accept | R9700 VRAM |
|---|---|---|---|---|---|
| **MTP n3** | **32.14** | 54.5% | **22.42** | 26.2% | 27,321 |
| DFlash2 n7 | 30.52 | 32.5% | 15.27 | 8.9% | 26,069 |
| DSpark n7 | 27.17 | 27.6% | 12.93 | 5.6% | **32,312** |

It is both the slowest and by far the hungriest — 32,312 of the R9700's 32,624
MiB, against DFlash2's 26,069. That appetite is what caused the OOM. Ranking is
unchanged from the f16-KV runs: **MTP > DFlash2 > DSpark**, on code and prose.

Drafts tested: `z-lab/Qwen3.8-27B-DFlash2-GGUF` Q4_K_M and
`erlidev/Qwen3.8-27B-DSpark-GGUF` Q8_0 (verified genuine DSpark — carries
`markov_w1/w2.weight`). Binary: `llama.cpp-rocm11-dflash2` (master 67a17c17c);
the production `llama.cpp-rocm10` build predates DFlash2 and cannot load either.

## Corpora

`/tmp/fn-prompt/`: `code10k.txt` `code40k.txt` (ggml-vulkan.cpp),
`prose10k.txt` `prose40k.txt` (Gutenberg Iliad, narrative body only — the
illustration index and table of contents are stripped, they are not
representative prose). `/tmp` is wiped on reboot; regenerate with `head -c`.
