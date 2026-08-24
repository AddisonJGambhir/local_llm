# UD-Q6_K_XL preset — 2026-08-24

Unsloth Dynamic v3.0 `UD-Q6_K_XL` (23.56 GiB) replacing Q8_0 (27.05 GiB), at full
262144 context with f16 KV, Vulkan tensor split.

## Shipped

`config/profiles/preset-38-27b-udq6kxl-vulkan-tensor-f16-256k.sh` (Vulkan menu 10)
tensor **50,50**, main-gpu 1, c=262144, f16 KV, ub 1024, batch 2048,
draft-mtp n3 with **f16 draft KV**, vision tower on Vulkan1.

Production: **XTX 23,648 MiB (552 MiB under the 24,200 ceiling)**, R9700 23,197.

## Ratio: raw decode, no speculation

Measured with llama-bench `-p 0 -n 256` because draft depth is NOT the target and
IRL acceptance is 0.5-0.6, where depth barely moves tg. Earlier server-based
numbers were speculation-inflated (0.83 acceptance on source-code prompts) and
did not represent real throughput.

| ratio | raw tg @d0 | raw tg @d65k | fit-check @262k | production |
|---|---|---|---|---|
| 45,55 | 20.24 | - | ~20,100 | - |
| 49,51 | 20.85 | 18.63 | 21,052 | ~20,800 |
| **50,50** | **21.52** | **19.62** | 23,886 | **23,648** |
| 51,49 | 21.77 | 19.84 | 24,552 | over |
| 52,48 | 21.84 | 19.99 | 23,984 | 24,193 (7 MiB) |
| 53,47 | - | 20.07 | 24,220 | over |
| 54,46 | - | 20.21 | 24,179 | 21 MiB |
| 55,45 | - | - | 24,361 | over |

Raw decode rises **monotonically** with weight on the XTX (the faster card):
+5.3% from 49/51 to 50/50. No chunkiness in speed at all.

VRAM is non-monotonic but perfectly reproducible (identical to the MiB across two
passes): 51,49 and 53,47 land badly on tensor-boundary rounding while 52,48 fits.

## Why not 52,48 despite being faster

Fit-checked at 23,984, but landed at **24,193 in production** — 7 MiB of margin.
The gap is the **desktop baseline**, which moved between 1,551 and 1,874 MiB over
one morning. Any margin smaller than that swing is not shippable regardless of
what a clean fit check says. 50,50's 552 MiB covers it.

## Model facts

- MTP head is **Q6_K** here (Q8_0 in the old files) but measured draft acceptance
  is **identical** (0.831 both), so the lower-precision draft head costs nothing.
- Not a uniform Q6: 310 tensors Q8_0, 168 Q6_K, 27 Q5_K, 1 Q4_K, 360 F32 — the
  body is higher precision than a plain Q6_K, which has only 49 Q8_0 tensors.
- Built with the v3.0 imatrix: **1251 calibration chunks vs 45** in Q8_0 and the
  plain Q6_K.
- Unsloth publish **no KLD or perplexity for Qwen3.8-27B at any quant**; their
  only Qwen3.8 data is a top-1%-accuracy chart, which puts UD-Q6_K_XL ~0.2 points
  under Q8_0. Their KLD tables are Gemma 3 only and stop at Q4_K_XL.

## Q8_0 was NOT re-quantized for v3.0

The local Aug-15 Q8_0 is **byte-identical** to the current repo file (same size,
same head and mid-slice SHA-256). HuggingFace reports 2026-08-20 for every path
because of a repo-wide "Upload imatrix" commit; Q8_0's content did not change.
Dynamic quantization has nothing to do at Q8_0 — it is uniform 8-bit, so there
are no per-layer bit-width decisions for a better imatrix to inform. The
`imatrix.chunks_count = 45` in that file is vestigial metadata.

## Measurement traps hit along the way

1. **Omitting `MTMD_BACKEND_DEVICE=Vulkan1`** put the vision tower on the XTX
   (~1.1 GiB) and made every ratio appear not to fit.
2. **Guarding at 23,850 instead of the real 24,200 ceiling** wrongly rejected
   50,50 at 23,899.
3. **Reading the load value as "settled"** — peak and steady came out identical
   because they were sampled at the same instant. Measure after a real generation,
   once the reading stops moving.
4. **llama-bench `-ts` takes SLASHES**; commas make it sweep values as separate
   single-GPU runs.

## Hermes

`~/.hermes/config.yaml` -> `context_length: 252144` (server context - 10,000).
Already pointed at `http://localhost:1234/v1`, `default: local`, matching the
server's `--alias local`. Gateway active on :8642.

---

## Head-to-head: Q8_0 vs UD-Q6_K_XL (added after the fact)

The quant swap was originally shipped on the ASSUMPTION that a smaller file is
faster on a bandwidth-bound workload. That was never measured. It is now --
same build, same flags, llama-bench (no speculation), pp and tg at two depths.

At matched ratio 49/51, ub1024/b2048:

| metric | Q8_0 | UD-Q6_K_XL | delta |
|---|---|---|---|
| tg @d0    | 19.40 | 20.76 | **+7.0%** |
| tg @d65k  | 17.68 | 18.66 | **+5.5%** |
| pp @d0    | 392.92 | 385.74 | -1.8% |
| pp @d65k  | 321.68 | 316.30 | -1.7% |

At 50/50 the same pattern holds (tg +6.5%/+5.5%, pp -1.4%/-1.6%).

**Verdict: the swap is justified.** ~+6% decode for ~-1.6% prefill, plus 2.9 GiB
of VRAM at matched settings (16,748 vs 19,608), which is what makes full 262144
context possible at all.

Best per model at d65k:

| config | tg | pp |
|---|---|---|
| UD-Q6 52/48 | 19.95 | 310.35 |
| **UD-Q6 50/50** | **19.70** | **321.19** |
| Q8_0 50/50 | 18.67 | 326.49 |
| Q8_0 49/51 | 17.68 | 321.68 |

52/48 is +1.3% tg but **-3.4% pp** at depth AND only 7 MiB of production VRAM
margin. 50/50 is the balanced, safe optimum.

## Batch/ubatch does NOT transfer to tensor split

The +33% d0 / +63% d65k prefill win from `--ubatch-size 512 --batch-size 8192`
is row/layer ONLY. Under tensor split it is NEGATIVE:

| config | pp @d65k | tg |
|---|---|---|
| UD-Q6 50/50 ub512/b8192 | -2.1% | -0.2% |
| Q8_0 49/51 ub512/b8192  | -2.6% | +0.3% |

Tensor split keeps ub1024 / b2048.
