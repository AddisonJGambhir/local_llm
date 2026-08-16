# Muse Glimmer 30B + DFlash: Vulkan vs ROCm audit (gfx1100 / RX 7900 XTX)

**Date:** 2026-08-10 · **Host:** Ryzen 9 9950X3D + RX 7900 XTX 24 GB, Ubuntu, Mesa 26.0.3 (RADV)
**llama.cpp:** version 10358, commit `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` (both builds)

---

## TL;DR

The Vulkan backend was **not** broken, misbuilt, or misconfigured. The whole
"ROCm decodes much faster than Vulkan" result was caused by a single setting:
**`--spec-draft-n-max 15`**.

That value is DFlash's *architectural ceiling*, not a tuned optimum. On this card
the Vulkan backend pays a 3.16× penalty for a 16-wide speculative verify pass,
while ROCm pays only 1.49× — so the same setting is near-optimal on ROCm and
pessimal on Vulkan.

Changing Vulkan to `--spec-draft-n-max 4` (plus `--ubatch-size 512`) took decode
from **25.10 → 73.07 t/s** on the standard chat prompt: **2.91×**, and **1.60×
faster than ROCm's own best configuration**.

The original conclusion — "Vulkan wins prefill, loses decode" — was wrong on
*both* halves once measured properly.

---

## 1. Audit findings (read-only)

### Builds — no version or flag mismatch

| | ROCm (`build/`) | Vulkan (`build-vulkan/`) |
|---|---|---|
| version | 10358 (`030ebb558`) | 10358 (`030ebb558`) |
| compiler | GNU 15.2.0 | GNU 15.2.0 |
| `CMAKE_BUILD_TYPE` | Release | Release |
| `GGML_NATIVE` | ON | ON |
| backend | `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1100` | `GGML_VULKAN=ON` |
| debug/validation | — | `VULKAN_DEBUG/VALIDATE/CHECK_RESULTS` all **OFF** |
| `GGML_HIP_ROCWMMA_FATTN` | OFF | n/a |

No Vulkan validation layers or debug shaders were enabled — those would have
been an obvious performance trap and were ruled out.

### Device selection — both correct

```
ROCm0  : AMD Radeon RX 7900 XTX                          (HIP_VISIBLE_DEVICES=0)
Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31)   <- selected by --device Vulkan0
Vulkan1: AMD Ryzen 9 9950X3D (RADV RAPHAEL_MENDOCINO)  <- iGPU, correctly avoided
```

Vulkan device capabilities on the XTX:

```
uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64
shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
```

`KHR_coopmat` (coopmat1) is present and used. `coopmat2` is NVIDIA-only, so its
absence is expected, not a defect.

### Resolved commands — identical except the three intended fields

Token-level diff of the two resolved DFlash command lines:

```
< .../llama.cpp/build/bin/llama-server
> .../llama.cpp/build-vulkan/bin/llama-server
> --device Vulkan0
```

Nothing else differed. Both resolve to `-ngl 99 -fit off -c 16384 -t 16 -tb 32
--parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
--context-shift --spec-type draft-dflash --spec-draft-ngl 99 ...`.

### Startup logs — clean

Full GPU offload on both, no CPU fallback, no Flash-Attention fallback, no
unsupported-op messages. Benign warnings present on both backends:

- `special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect`
- `KV cache shifting is not supported for this context, disabling KV cache shifting`
  → `--context-shift` is **inert** for this model; harmless but misleading in the profile.

### Model shape (explains the behaviour below)

`Muse-Glimmer-30B-UD-Q5_K_M.gguf` — arch `muse-glimmer`, 17.86 GiB, 27.85 B params:

| key | value |
|---|---|
| block_count | 52 |
| embedding_length / ffn | 6656 / 19968 |
| heads / kv heads | 32 / 2 (GQA 16:1) |
| **sliding_window / pattern** | **2048 / 4** (3 of every 4 layers are SWA) |
| vocab | 202 048 |
| final_logit_softcapping | 20.0 |

`dflash-kquant.gguf` — arch `dflash`, 5 blocks, **`block_size = 16`**,
`target_layers = [2,14,26,38,50]`, `mask_token_id = 201818`, `n_extract = 5`.

---

## 2. Root cause: the cost of a wide verify pass

Speculative decoding runs the *target* model with `n_tokens = 1 + n_draft`. So
the profitability of DFlash depends entirely on how much a wide forward pass
costs relative to a 1-wide one.

Measured with `llama-bench` (same model, `-fa on -ctk q8_0 -ctv q8_0 -ub 192 -t 16 -r 3`),
converted to **milliseconds per forward pass**:

| batch N | Vulkan ms | ROCm ms | Vulkan ×N=1 | ROCm ×N=1 |
|---:|---:|---:|---:|---:|
| 1 | **26.01** | 32.31 | 1.00× | 1.00× |
| 2 | 26.39 | 38.48 | 1.01× | 1.19× |
| 4 | 28.08 | 53.63 | 1.08× | 1.66× |
| 8 | 37.02 | 95.83 | 1.42× | 2.97× |
| **16** | **82.16** | **48.14** | **3.16×** | **1.49×** |
| 192 | 311.2 | 207.3 | 11.96× | 6.42× |

Two different kernel-selection curves:

- **Vulkan** is the *faster* backend at N=1–8 in absolute terms, then falls off a
  cliff between 8 and 16 (37.0 → 82.2 ms).
- **ROCm** is slower at N=1–8 but hits its quantized-matmul (MMQ) sweet spot at
  16, where 16 tokens cost only 1.49× a single token.

With `n-max=15` (a 16-wide verify), ROCm verifies 16 tokens for 1.49× the price
while Vulkan pays 3.16×. Same acceptance rate on both (~0.16), opposite outcome.

This is confirmed at the server level — **DFlash at n-max=15 is a net loss on Vulkan**:

| config | pp t/s | tg t/s | DFlash effect on decode |
|---|---:|---:|---|
| `vulkan-base` (no DFlash) | 262.11 | **38.20** | — |
| `vulkan-dflash` n-max=15 | 239.72 | **31.86** | **−17 %** |
| `rocm-base` (no DFlash) | 145.37 | 32.01 | — |
| `rocm-dflash` n-max=15 | 124.82 | 54.22 | **+69 %** |

Note the control: **without DFlash, Vulkan decodes 19 % faster than ROCm**
(38.20 vs 32.01), exactly as expected from prior experience on this rig.

---

## 3. `--spec-draft-n-max` sweep

Fixed chat prompt (77 tok), `max_tokens=256`, temp 0, **server restarted between
every run** (`cache_n=0` verified on all runs), 3 runs each, medians:

| n-max | Vulkan tg t/s | Vulkan accept | ROCm tg t/s | ROCm accept |
|---:|---:|---:|---:|---:|
| 1 | 56.99 | 0.809 | 40.52 | 0.776 |
| 2 | 71.38 | 0.681 | 43.40 | 0.673 |
| 3 | 74.81 | 0.540 | 41.69 | 0.543 |
| **4** | **75.06** | 0.450 | — | — |
| 5 | 66.35 | 0.354 | 32.05 | 0.364 |
| 6 | 65.63 | 0.325 | — | — |
| 8 | 32.46 | 0.291 | 49.21 | 0.255 |
| 15 (shipped) | **31.86** | 0.159 | **54.22** | 0.166 |

**The two backends have opposite optima:** Vulkan peaks at n-max 3–4, ROCm at
n-max 15. The shipped value was the worst possible choice for Vulkan and the
best possible one for ROCm — which is precisely why the comparison looked the way
it did.

Acceptance *rate* falls monotonically with depth (later draft tokens are less
likely to be right), so the win comes from the depth where `1 + n_max × accept`
grows faster than the verify-pass cost. Vulkan's cost curve caps that at ~4.

---

## 4. `--ubatch-size` sweep (long prompt, 2927 tokens)

| ubatch | Vulkan pp t/s | ROCm pp t/s |
|---:|---:|---:|
| 192 | 530.62 | 704.38 |
| 256 | 637.37 | 710.35 |
| **512** | **687.74** | **801.46** |

`-ub 512` is worth **+30 % prefill on Vulkan** and **+14 % on ROCm**, at a cost of
~1.4 % on short-prompt decode (74.07 → 73.07 t/s). VRAM stays safe: 19.12 GiB
resident, 2.44 GiB free on the card at 16k context.

---

## 5. The prefill claim was also wrong

The original test used a **77-token prompt**, where `prompt_ms` is dominated by
fixed per-request overhead rather than kernel throughput. At a realistic prompt
length the ordering **reverses**:

| prompt length | Vulkan pp t/s | ROCm pp t/s | winner |
|---|---:|---:|---|
| 77 tok | 239.72 | 124.82 | Vulkan (artifact) |
| 2927 tok, ub 192 | 530.62 | 704.38 | **ROCm +33 %** |
| 2927 tok, ub 512 | 687.74 | 801.46 | **ROCm +16 %** |

This matches `llama-bench` pp192 (ROCm 926 vs Vulkan 617 t/s). So the intuition
"use Vulkan for long-document/RAG prefill" is backwards — **ROCm is the faster
prefill backend on this rig**; Vulkan only appeared to win at a prompt too short
to measure prefill at all.

---

## 6. Break-even by workload

Using the tuned ub-512 configurations at 2927-token depth:

| | Vulkan (n-max 4) | ROCm (n-max 15) |
|---|---:|---:|
| prefill | 1.454 ms/tok | 1.248 ms/tok |
| decode (long ctx) | 12.60 ms/tok | 13.96 ms/tok |
| decode (short ctx) | 13.69 ms/tok | 22.50 ms/tok |

Vulkan is faster end-to-end when **generated tokens > ~15 % of prompt tokens**.
For a 2927-token prompt the crossover is ≈ **445 output tokens**.

- **Chat / code / agentic (short prompt, long output):** Vulkan wins decisively —
  73.07 vs 44.45 t/s decode, a 64 % lead.
- **Pure RAG / summarisation (huge prompt, few output tokens):** ROCm wins on
  total latency, driven entirely by its prefill advantage.
- Everything in between favours Vulkan.

**Latency consistency also favours Vulkan.** Across 3 runs Vulkan decode varied
79.06 / 79.46 / 79.38 t/s (0.5 % spread, and identical acceptance every run);
ROCm varied 52.00 / 71.65 / 77.10 t/s (48 % spread) with acceptance drifting
0.23–0.38. ROCm's kernels are non-deterministic here; Vulkan's are reproducible.

---

## 7. What the official sources actually say

**Meta (HuggingFace blog, `huggingface.co/blog/muse-glimmer`)** — recommends:

```
llama serve -hf meta-models/Muse-Glimmer-30B-GGUF --spec-type draft-dflash --spec-draft-n-max 15
```

with the rationale: *"Muse Glimmer's DFlash model was trained with a block size of
16, one anchor token plus 15 proposed tokens, so **any value above 15 will be
clamped to 15**."*

That is a statement about the **drafter's ceiling**, not about optimal
throughput. Our local source confirms it is literally a clamp
(`common/speculative.cpp:979-982`: *"requested draft size … exceeds the trained
block size … clamping"*). Meta's page makes **no backend recommendation** at all,
and `dev.meta.ai/docs/muse-glimmer/spec-decode` doesn't cover llama.cpp.

**AMD (`amd.com/en/blogs/2026/run-meta-muse-glimmer-30b-on-amd-ryzen-ai-max-and-radeon-gpus.html`)**
— body text: *"dFlash must be enabled with **the correct number of draft tokens**
for optimal performance."* Its measurement footnotes:

| footnote | hardware | backend | setting |
|---|---|---|---|
| SHO-77 | Ryzen AI Max+ 395 (Windows) | Vulkan | **`--spec-draft-n-max=4`** |
| RPW-537 | Radeon AI PRO R9700 (Windows) | Vulkan | **`--spec-draft-n-max=2`** |

So **AMD's own Vulkan numbers use n-max 2–4, never 15.** Our independently
measured optimum of 3–4 lands squarely in AMD's range.

### Scope corrections

- Vulkan is **not technically required** — ROCm runs DFlash correctly here.
- Vulkan is what **AMD benchmarked**, on **Windows**, on **RDNA4 (R9700)** and an
  **RDNA3.5 APU** — not Linux + RDNA3. The backend choice transfers; the n-max
  guidance transfers and is confirmed.
- AMD reports "up to 53 t/s on a single R9700 with dFlash". This rig now reaches
  **73–75 t/s**, so nothing here is underperforming published figures.
- The premise that "n-max=15 is the official recommendation" is true of *Meta's*
  page, but it is a clamp ceiling; AMD's *measured* Vulkan configs contradict
  using it for throughput.

---

## 8. Final results table

All runs: server restarted between every run, `cache_n=0` verified, temp 0,
tool-free prompt, reasoning `auto`, 3 runs each, **medians** reported.

### Short prompt (77 tok) — chat/code profile

| config | ubatch | n-max | pp t/s | tg t/s | accept |
|---|---:|---:|---:|---:|---:|
| **vulkan (restored best)** | **512** | **4** | **239.32** | **73.07** | 0.435 |
| vulkan | 192 | 4 | 238.18 | 74.07 | 0.435 |
| vulkan | 192 | 3 | 241.15 | 73.65 | 0.521 |
| vulkan *(original)* | 192 | 15 | 240.46 | **25.10** | 0.110 |
| vulkan, no DFlash | 192 | — | 289.87 | 37.92 | — |
| rocm | 512 | 15 | 124.27 | 44.45 | 0.117 |
| rocm *(original)* | 192 | 15 | 125.07 | 45.84 | 0.123 |
| rocm, no DFlash | 192 | — | 145.37 | 32.01 | — |

*(768-token generations; the 256-token runs in §3 show the same ordering.)*

### Long prompt (2927 tok) — RAG profile

| config | ubatch | n-max | pp t/s | tg t/s | total ms |
|---|---:|---:|---:|---:|---:|
| vulkan | 512 | 4 | 687.74 | 79.38 | 5062 |
| vulkan | 192 | 4 | 530.62 | 68.09 | 6456 |
| rocm | 512 | 15 | **801.46** | 71.65 | **4540** |
| rocm | 192 | 15 | 704.38 | 64.79 | 5223 |

### Headline

| metric | before | after | change |
|---|---:|---:|---|
| Vulkan chat decode | 25.10 t/s | **73.07 t/s** | **2.91×** |
| vs. ROCm best (44.45 t/s) | 0.56× | **1.64×** | — |
| Vulkan long-prompt prefill | 530.62 t/s | **687.74 t/s** | **1.30×** |
| VRAM resident | 18.76 GiB | 19.12 GiB | +0.36 GiB (2.44 free) |

---

## 9. Changes made

**Modified — 2 files, 3 values:**

1. `config/profiles/muse-glimmer-30b-q5km-vulkan-q8-16k-dflash.sh`
   - `LLAMA_SPEC_DRAFT_N_MAX`: `15` → `4` (+ explanatory comment)
   - `LLAMA_UBATCH`: `192` → `512`
2. `config/profiles/muse-glimmer-30b-q5km-rocm-q8-16k-dflash.sh`
   - `LLAMA_UBATCH`: `192` → `512` (n-max stays 15 — measured optimum for ROCm)

**Created:** this document.

**Inspected only, unmodified:** both base (non-DFlash) profiles, both
`CMakeCache.txt`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
`common/speculative.cpp`, `bin/llamactl`, `src/server/common.sh`, both GGUFs'
metadata, and the rotating server log.

**Not done:** no rebuild, no source change, no checkout change, no model or
download changes, no mixed-device execution. Every measurement used the existing
binaries at commit `030ebb558`.

**Restored state:** `muse-glimmer-30b-q5km-vulkan-q8-16k-dflash` running on
`127.0.0.1:1234`, `/health` → `{"status":"ok"}`, confirmation run 238.9 t/s
prefill / **73.07 t/s** decode.

Resolved command now in service:

```
.../build-vulkan/bin/llama-server -m .../Muse-Glimmer-30B-UD-Q5_K_M.gguf \
  --alias local --device Vulkan0 -ngl 99 -fit off -c 16384 -t 16 -tb 32 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --ubatch-size 512 --context-shift \
  --spec-type draft-dflash --spec-draft-model .../dflash-kquant.gguf \
  --spec-draft-ngl 99 --spec-draft-n-max 4 --spec-draft-n-min 0 \
  --temp 0.7 --top-p 1.0 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning auto --metrics --port 1234 --host 127.0.0.1 --timeout 0 --log-colors off --jinja
```

---

## 10. Open items (not acted on)

1. **`--spec-type draft-dflash` is correct** — kept. The sidecar's `general.architecture`
   is `dflash` with `block_size`/`target_layers`/`mask_token_id`, and the server logs
   `adding speculative implementation 'draft-dflash'` with `block_size=16, n_extract=5`.
   No reason to try legacy `dflash`.
2. **`LLAMA_CONTEXT_SHIFT=1` is inert** for this model (server disables KV shifting at
   startup). Cosmetic only — left as-is.
3. **Switch backend by workload** if a pure-RAG profile is ever wanted: the ROCm
   DFlash profile at ub 512 is the faster choice above ~6:1 prompt:output ratio.
4. **The Vulkan N=8→16 cliff** (37.0 → 82.2 ms) looks like a kernel-selection
   threshold in `ggml-vulkan`. Potentially worth an upstream report, but it needs
   isolation with `test-backend-ops` first — not investigated here.
