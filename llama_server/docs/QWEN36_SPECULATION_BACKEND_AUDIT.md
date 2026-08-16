# Qwen3.6: MTP vs DFlash, backends, and BeeLlama vs mainline (gfx1100 / RX 7900 XTX)

**Date:** 2026-08-11 · **Host:** Ryzen 9 9950X3D + RX 7900 XTX 24 GB, Ubuntu, Mesa 26.0.3 (RADV)
**Mainline llama.cpp:** 10358 (`030ebb558`) · **BeeLlama:** 11011 (`32040e79d`, `preview-v0.4.3`)

Follow-up to `MUSE_GLIMMER_DFLASH_BACKEND_AUDIT.md`, re-running the Qwen stack
against current binaries after llama.cpp, BeeLlama, ROCm and Vulkan all moved.

---

## TL;DR

1. **MTP beats DFlash on both Qwen models** — by 22 % on the MoE and 12 % on the
   dense 27B — *and* uses less VRAM. This is the opposite of the Muse Glimmer
   result, so drafter type is not a universal win either way.
2. **Your `--spec-draft-n-max 2` is close but not optimal.** The MoE peaks at 3,
   the dense 27B at 4. Worth +5.5 % and +7.5 % on the real production profiles.
3. **BeeLlama is now redundant.** Mainline serves MTP *and* DFlash for
   `qwen35`/`qwen35moe` on both backends, and the two engines are within 0.4 %.
4. **Vulkan wins everything here.** +38 % decode over ROCm on the MoE, and unlike
   Muse Glimmer, Vulkan wins prefill too.
5. **Your old Qwen DFlash sidecars were dead files** — arch `dflash-draft`,
   rejected by both engines. Replaced with current-format `dflash` sidecars.
6. **`--ubatch-size 192` was costing 77 % of your prefill throughput.** Raising it
   to 512 takes MoE prefill from 1319.6 to 2332.0 t/s at zero decode cost.
   **This is the single largest win in this document** — see §8, added after the
   first draft.

---

## 1. Audit findings

### Your DFlash sidecars did not work at all

Both pre-existing Qwen DFlash files declare `general.architecture = 'dflash-draft'`:

```
Qwen3.6-27B-DFlash-IQ4_XS.gguf      (Jun  8)  arch = dflash-draft
qwen36-35b-a3b-dflash-IQ4_XS.gguf   (Jun 10)  arch = dflash-draft
```

Neither engine registers that name — `llama-arch.cpp` has only
`{ LLM_ARCH_DFLASH, "dflash" }`. Both fail identically:

```
error loading model: unknown model architecture: 'dflash-draft'
```

So **any past Qwen DFlash benchmark on this rig either predates the format change
or never actually ran DFlash.** Replaced (old files left in place, not deleted):

| new file | size | arch | target_layers |
|---|---:|---|---|
| `Qwen3.6-27B-DFlash-Q8_0.gguf` | 1764 MB | `dflash` | [2,17,32,47,62] |
| `qwen36-35b-a3b-dflash-Q8_0.gguf` | 402 MB | `dflash` | [2,7,12,17,23,28,…] |

Sourced from `Anbeeld/Qwen3.6-27B-DFlash-GGUF` and
`Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF` (quantizations of the z-lab DFlash drafts).

### Model metadata

| model | arch | blocks | MTP layers | notes |
|---|---|---:|---:|---|
| `Qwen3.6-27B-MTP-Q4_K_M` | qwen35 | 65 | 1 | dense |
| `Qwen3.6-27B-IQ4_NL` | qwen35 | 64 | — | dense, no MTP |
| `Qwen3.6-35B-A3B-MTP-UD-IQ4_NL` | qwen35moe | 41 | 1 | 256 experts, 8 used |
| `Qwen3.6-35B-A3B-UD-IQ4_NL` | qwen35moe | 40 | — | no MTP |
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | qwen35moe | 41 | **1** | **has MTP** |

`nextn_predict_layers = 1` on every MTP model. `speculative.cpp` only chains
heads when `n_mtp_layers > 1`, so these run the single head autoregressively —
but `--spec-draft-n-max` is still live, confirmed empirically by acceptance
moving cleanly with depth (0.768 → 0.697 → 0.630 → 0.541 for n-max 1→4).

---

## 2. Raw kernel matrix (llama-bench, no speculation)

`-fa on -ctk q8_0 -ctv q8_0 -ub 192 -t 16 -r 2`. Decode (`tg128`, t/s):

| model | mainline-rocm | mainline-vulkan | bee-rocm | bee-vulkan |
|---|---:|---:|---:|---:|
| 27B-MTP Q4_K_M | 33.81 | 39.61 | 33.47 | **40.26** |
| 27B IQ4_NL | 36.07 | 40.30 | 34.94 | **40.83** |
| 35B-A3B-MTP IQ4_NL | 94.10 | 141.34 | 95.53 | **142.62** |
| 35B-A3B IQ4_NL | 95.39 | 140.14 | 96.73 | **142.78** |
| 35B-A3B Q4_K_M | 95.07 | 137.59 | 99.84 | **138.54** |

Three things fall out:

- **Vulkan > ROCm on decode for every model** (+13 % dense, +47 % MoE).
- **BeeLlama ≈ mainline** (+0.9 % to +1.9 %, i.e. noise).
- **The MoE decodes 3.5× the dense 27B** (141 vs 40) — 3B active params.
- `35B-A3B-UD-Q4_K_M`, the heaviest quant, is only **2.6 % slower** than IQ4_NL.
  Its real cost is VRAM/context, not throughput.

### Batch-cost curve (ms per forward pass) — why draft depth matters

| N | 27B Vulkan | 27B ROCm | MoE Vulkan | MoE ROCm |
|---:|---:|---:|---:|---:|
| 1 | 25.87 | 35.13 | 7.53 | 16.06 |
| 4 | 28.45 | 54.03 | 11.67 | 17.59 |
| 16 | 86.13 | 70.69 | 42.05 | 33.08 |
| **×N=1 at 16** | **3.33×** | 2.01× | **5.58×** | 2.06× |

The same Vulkan wide-batch cliff found on Muse Glimmer, and *steeper* on the MoE
(5.58×). This is the mechanism behind every draft-depth result below: on Vulkan a
wide verify pass is disproportionately expensive, so shallow drafts win.

### Anomaly

`bee-rocm` on `35B-A3B-UD-Q4_K_M` collapses at pp16: **232.98** t/s vs
mainline-rocm's **430.07**. BeeLlama-specific, ROCm-only, this model only. Another
reason to stay on mainline. Not investigated further.

---

## 3. MTP vs DFlash vs none

Server-level, mainline Vulkan, 32k context, 512 tokens, temp 0, **server
restarted between every run** (`cache_n=0` verified), medians.

### MoE — `Qwen3.6-35B-A3B-MTP-UD-IQ4_NL`

| config | pp t/s | tg t/s | accept | VRAM GiB |
|---|---:|---:|---:|---:|
| no speculation | **344.79** | 139.72 | — | 16.80 |
| MTP n-max 1 | 282.74 | 168.51 | 0.768 | 17.32 |
| MTP n-max 2 *(your default)* | 285.85 | 196.86 | 0.697 | 17.38 |
| **MTP n-max 3** | 283.30 | **204.87** | 0.630 | 17.45 |
| MTP n-max 4 | 281.67 | 199.88 | 0.541 | 17.51 |
| MTP n-max 5 | 281.53 | 179.52 | 0.432 | 17.57 |
| MTP n-max 6 | 280.19 | 177.37 | 0.409 | 17.64 |
| DFlash n-max 1 | 227.36 | 143.54 | 0.750 | 17.63 |
| DFlash n-max 2 | 229.48 | 164.56 | 0.590 | 17.70 |
| DFlash n-max 3 | 227.06 | 167.67 | 0.489 | 17.76 |
| DFlash n-max 4 | 227.22 | 167.15 | 0.411 | 17.82 |
| DFlash n-max 6 | 229.73 | 150.81 | 0.294 | 17.95 |
| DFlash n-max 8 | 231.61 | 63.49 | 0.229 | 18.08 |
| DFlash n-max 15 | 221.32 | 59.14 | 0.134 | 18.52 |

### Dense 27B — `Qwen3.6-27B-MTP-Q4_K_M`

| config | pp t/s | tg t/s | accept | VRAM GiB |
|---|---:|---:|---:|---:|
| no speculation | **194.54** | 39.97 | — | 16.27 |
| MTP n-max 1 | 161.33 | 64.23 | 0.845 | 16.81 |
| MTP n-max 2 *(your default)* | 158.87 | 72.85 | 0.670 | 16.95 |
| MTP n-max 3 | 158.61 | 75.93 | 0.581 | 17.10 |
| **MTP n-max 4** | 160.08 | **76.33** | 0.518 | 17.25 |
| DFlash n-max 2 | 136.94 | 65.83 | 0.597 | 18.62 |
| DFlash n-max 4 | 135.80 | 68.00 | 0.407 | 18.91 |
| DFlash n-max 8 | 134.90 | 31.62 | 0.261 | 19.51 |
| DFlash n-max 15 | 130.56 | 31.83 | 0.155 | 20.55 |

### Reading

- **MTP beats DFlash at every comparable depth**, on both models, on decode *and*
  prefill *and* VRAM. Best MTP vs best DFlash: MoE 204.87 vs 167.67 (**+22 %**),
  dense 76.33 vs 68.00 (**+12 %**).
- **DFlash at n-max 8+ is worse than no speculation at all** (MoE 63.5 vs 139.7;
  dense 31.6 vs 40.0) while costing up to 4.3 GiB extra VRAM. If you ever enable
  DFlash on Qwen, never leave it at the default depth.
- **Speculation costs prefill.** MTP −18 % (MoE), DFlash −34 %. Worth it for chat
  or agentic work; check it against your own prompt:output ratio for RAG.

---

## 4. Backend and engine, at server level

MoE, MTP n-max 3, 32k:

| config | pp t/s | tg t/s |
|---|---:|---:|
| **mainline Vulkan** | **283.30** | **204.87** |
| BeeLlama Vulkan | 262.83 | 206.01 |
| mainline ROCm | 115.17 | 148.92 |

- **Vulkan beats ROCm by 38 % on decode and 2.5× on prefill.** Note this differs
  from Muse Glimmer, where ROCm won prefill — do not generalise backend choice
  across model architectures.
- **BeeLlama vs mainline is a tie** (206.01 vs 204.87 decode, +0.6 %; mainline
  wins prefill by 7.8 %). At n-max 2 the same: 196.03 vs 196.86.

BeeLlama therefore has **no remaining advantage**: mainline is 2 days newer,
loads MTP and DFlash for both Qwen archs on both backends, and adds
`--spec-type draft-dspark` which BeeLlama lacks.

---

## 5. Production-context validation

The tuning above ran at 32k. Both changes were re-validated at the contexts the
profiles actually use, changing **only** the binary and `n-max`:

### `default.sh` — MoE at 261 120 context, `--cache-ram 24576`

| | pp t/s | tg t/s | VRAM GiB |
|---|---:|---:|---:|
| before (BeeLlama Vulkan, n-max 2) | 257.97 | 190.46 | 20.09 |
| **after (mainline Vulkan, n-max 3)** | **285.82** | **200.95** | 20.16 |
| | **+10.8 %** | **+5.5 %** | +0.07 |

### `preset-27b-q4km-q8-128k.sh` — dense 27B at 131 072 context

| | pp t/s | tg t/s | VRAM GiB |
|---|---:|---:|---:|
| before (BeeLlama Vulkan, n-max 2) | 153.66 | 72.58 | 20.39 |
| **after (mainline Vulkan, n-max 4)** | **156.34** | **78.03** | 20.69 |
| | +1.7 % | **+7.5 %** | +0.30 |

Gains are smaller at production context than at 32k (speculation economics shift
with KV depth), which is exactly why the 32k numbers alone were not used to
justify the change.

---

## 6. Changes made

**Modified — 4 files:**

1. `config/profiles/default.sh`
   - binary: BeeLlama Vulkan → **mainline Vulkan**
   - `LLAMA_SPEC_DRAFT_N_MAX`: `2` → **`3`** (+ explanatory comments)
2. `config/profiles/preset-35b-iq4-q8-256k.sh`
   - binary → mainline Vulkan; `n-max` `2` → **`3`**
3. `config/profiles/preset-27b-q4km-q8-128k.sh`
   - binary → mainline Vulkan; `n-max` `2` → **`4`** (+ comment)
4. `bin/llama-launch`
   - corrected the stale "BeeLlama is required for legacy DFlash or MTP-on-Vulkan"
     comment; now records that mainline covers everything, with the measurement.

**Added — 2 model files** (approved): `Qwen3.6-27B-DFlash-Q8_0.gguf`,
`qwen36-35b-a3b-dflash-Q8_0.gguf`.

**Created:** this document.

**Inspected only:** all four builds' binaries, `src/llama-arch.cpp` and
`common/speculative.cpp` in both trees, `src/server/common.sh`, every Qwen GGUF's
metadata.

**Deliberately not changed:**

- The two dead sidecars (`*-DFlash-IQ4_XS.gguf`, `qwen36-*-dflash-IQ4_XS.gguf`)
  are still on disk, ~1.2 GB total. They are unusable — safe to delete.
- `preset-27b-q4km-q5_1-176k.sh` and `preset-35b-q4km-q5_1-190k.sh` still use
  BeeLlama and n-max 2. They were removed from the launcher menu earlier and are
  now unreachable, so they were left alone.
- BeeLlama tree untouched. Nothing rebuilt.

---

## 7. Recommendations not yet acted on

1. **`Qwen3.6-35B-A3B-UD-Q4_K_M` deserves a profile.** It has MTP layers and
   decodes within 2.6 % of IQ4_NL, so the better quant is nearly free on speed.
   The constraint is VRAM (22.66 GiB file) and therefore max context — that needs
   a headroom scan before it can be recommended as a preset.
2. **The dense 27B is hard to justify on speed.** 78 t/s vs the MoE's 201 t/s at
   production context, 2.6×. Keep it only where its dense-model quality matters.
3. **DFlash on Qwen is available but not recommended** — the new sidecars work
   and are tuned above, but MTP wins on every axis. Revisit if a future DFlash
   drafter ships with higher acceptance.
4. **Consider retiring the BeeLlama tree** (~large checkout) now that nothing
   depends on it. Not deleted here.

---

## 8. Addendum — real prefill, and the ubatch finding

*Added after the sections above. This section supersedes every prefill number
earlier in this document.*

### The prefill numbers in §3–§5 were not real prefill measurements

Sections 3–5 measured prefill with the standard **77-token** benchmark prompt.
At that length `prompt_ms` is dominated by fixed per-request overhead, not kernel
throughput — the identical artifact that produced the false "Vulkan wins prefill"
conclusion in the Muse Glimmer audit. Re-measured with a **3080-token** prompt:

| config | pp @77 tok | pp @3080 tok | understated by |
|---|---:|---:|---:|
| MoE, Vulkan, MTP 3, ub 192 | 283.30 | **1319.6** | 4.7× |
| dense 27B, Vulkan, MTP 4, ub 192 | 160.08 | **585.3** | 3.7× |

The §3–§5 figures remain valid for *comparing configs against each other*, but
they are not throughput numbers. Likewise the "+10.8 % prefill" attributed to the
BeeLlama→mainline switch in §5 was a 77-token measurement; treat it as a
config-vs-config delta only.

### ubatch sweep (3080-token prompt, 64-token cap, 2 runs, medians)

**MoE `Qwen3.6-35B-A3B-MTP-UD-IQ4_NL`, Vulkan, MTP n-max 3, 32k ctx**

| ubatch | pp t/s | vs 192 | tg t/s |
|---:|---:|---:|---:|
| 192 | 1319.6 | — | 173.6 |
| 256 | 1670.0 | +26.6 % | 174.4 |
| 512 | 2343.4 | +77.6 % | 167.1 |
| 1024 | **2829.6** | **+114.4 %** | 174.1 |

**Dense 27B `Qwen3.6-27B-MTP-Q4_K_M`, Vulkan, MTP n-max 4, 32k ctx**

| ubatch | pp t/s | vs 192 | tg t/s |
|---:|---:|---:|---:|
| 192 | 585.3 | — | 82.4 |
| 512 | 779.6 | +33.2 % | 82.4 |
| 1024 | **806.4** | +37.8 % | 82.5 |

**ROCm cross-check (MoE, MTP n-max 3)**

| ubatch | ROCm pp | Vulkan pp |
|---:|---:|---:|
| 192 | 608.7 | 1319.6 |
| 512 | 1731.7 | 2343.4 |

ROCm gains more in relative terms (+185 %) but **loses to Vulkan at every
ubatch**. The §4 conclusion — Vulkan wins prefill on Qwen — survives proper
measurement. (Contrast Muse Glimmer, where ROCm won prefill; the two model
families genuinely differ.)

Decode is unaffected at every ubatch on both models, so this is close to free.

### VRAM at production context decides the setting

Re-run at the contexts the profiles actually use:

**MoE at 261 120 context** (`--cache-ram 24576`)

| ubatch | pp t/s | tg t/s | procVRAM | card free | procGTT |
|---:|---:|---:|---:|---:|---:|
| 256 | 1676.3 | 170.5 | 20.23 | 1.59 | 0.26 |
| **512** | **2332.0** | 167.0 | 20.53 | **1.40** | 0.52 |
| 1024 | 2789.3 | 171.6 | 21.13 | 0.83 | 1.03 |

**Dense 27B at 131 072 context**

| ubatch | pp t/s | tg t/s | procVRAM | card free | procGTT |
|---:|---:|---:|---:|---:|---:|
| **512** | **787.2** | 82.1 | 20.97 | **1.00** | 0.30 |
| 1024 | 808.2 | 82.3 | 21.42 | 0.56 | 0.59 |

**Chose 512, not 1024, for both.** On the MoE, 1024 buys a further ~20 % prefill
but halves free VRAM to 0.83 GiB and doubles GTT residency to 1.03 GiB — too thin
on a card that also drives the desktop, and the KV cache is fully allocated at
startup so this is already the steady-state figure. On the dense 27B, 1024 buys
only +2.7 % over 512 for half the headroom. 512 keeps ≥1.0 GiB free on both.

### Are the presets worth updating? Yes — this is the biggest single win

| profile | prefill before | prefill after | decode |
|---|---:|---:|---:|
| `default.sh` / 35B preset (MoE @256k) | 1319.6 | **2332.0** | unchanged |
| 27B preset (@128k) | 585.3 | **787.2** | unchanged |

That is **+77 %** and **+34 %** prefill for a one-line change, dwarfing the
+5.5 %/+7.5 % decode gains from the n-max and engine work in §5.

### Changes made in this addendum

1. `config/profiles/default.sh` — `LLAMA_UBATCH`: `192` → **`512`** (+ rationale
   comment recording the VRAM headroom that ruled out 1024).
   Both live presets inherit this: `preset-35b-iq4-q8-256k.sh` and
   `preset-27b-q4km-q8-128k.sh`. Both were validated at their own contexts above.
2. `config/profiles/preset-27b-q4km-q5_1-176k.sh` and
   `preset-35b-q4km-q5_1-190k.sh` — **pinned** to an explicit `LLAMA_UBATCH=192`.
   These inherit from `default.sh` but run q5_1 KV at 176k/190k, which was *not*
   re-validated at 512; pinning stops the default change from silently altering
   an untested VRAM profile. (They are also unreachable from the launcher menu.)

### Still untested

- **ubatch for the Muse Glimmer profiles.** They were set to 512 in the earlier
  audit, but 1024 was never tried there. Muse runs only 16k context with 2.44 GiB
  free at ub 512, so there may be room. Not investigated.
- **ubatch between 512 and 1024** (e.g. 768) on the MoE — could recover part of
  the remaining 20 % while keeping ≥1.2 GiB free.
- Long-prompt behaviour beyond 3080 tokens, and prefill at genuine 100k+ depths.

---

## 9. Per-backend presets and the 35B-A3B Q4_K_M sweet spot

The launcher now offers **"Presets ROCm"** and **"Presets Vulkan"** as separate
branches, each with one best-measured preset per model. All measured 2026-08-11
at ubatch 512 on mainline llama.cpp 10358; every context verified free of GTT
spill by loading it and reading the server process's `drm-memory-gtt`.

### The tuning table

| model | Vulkan | tg t/s | ROCm | tg t/s | Vulkan lead |
|---|---|---:|---|---:|---:|
| Qwen 35B-A3B MoE IQ4_NL | MTP n-max **3** @256k | **204.9** | MTP n-max **3** @256k | 145.0 | +41 % |
| Qwen 35B-A3B MoE Q4_K_M | MTP n-max **3** @96k | **189.9** | MTP n-max **3** @70k | 140.5 | +35 % |
| Qwen 27B dense Q4_K_M | MTP n-max **4** @128k | **76.3** | MTP n-max **2** @128k | 48.5 | +57 % |
| Muse Glimmer 30B Q5_K_M | DFlash n-max **4** @16k | **73.8** | DFlash n-max **15** @16k | 45.8 | +61 % |

**Draft depth is not portable across backends.** The dense 27B wants 2 on ROCm
but 4 on Vulkan; Muse Glimmer wants 15 on ROCm but 4 on Vulkan. Each preset
therefore carries its own tuned `n-max` rather than inheriting one. Vulkan wins
every model here, so the ROCm branch exists for fallback, not for performance.

### 35B-A3B-UD-Q4_K_M — newly characterised

Previously unprofiled. It has MTP layers (41 blocks, `nextn_predict_layers = 1`)
and is the best-quality quant in the collection.

**n-max sweep (32k, 512 tokens, medians):**

| n-max | Vulkan tg | accept | ROCm tg | accept |
|---:|---:|---:|---:|---:|
| 2 | 188.2 | 0.711 | **140.5** | 0.706 |
| **3** | **189.9** | 0.596 | 139.1 | 0.589 |
| 4 | 176.5 | 0.474 | — | — |
| 6 | 147.4 | 0.327 | 115.4 | 0.381 |
| 8 | — | — | 81.8 | 0.274 |

Both backends are effectively flat between 2 and 3; **3** chosen on both to match
the other MoE presets.

**Context ceiling (`ctx_headroom.py`, q8_0 KV, MTP on, ubatch 512, Vulkan):**

```
 16k -> fit    procVRAM 21.13  procGTT 0.05  free 0.81
256k -> spill  procVRAM 21.72  procGTT 3.13  free 0.37
136k -> spill  procVRAM 22.01  procGTT 1.00  free 0.07
 72k -> fit    procVRAM 21.86  procGTT 0.16  free 0.22
104k -> spill  procVRAM 22.04  procGTT 0.48  free 0.04
 88k -> spill  procVRAM 21.95  procGTT 0.32  free 0.13
 80k -> fit    procVRAM 21.85  procGTT 0.30  free 0.23
```

**=> scan reported 80k max fit, spills by 88k.** This supersedes the README's
"136k @ q8_0" entry, which predates MTP and ubatch 512 — both consume headroom.

**The scan is conservative.** Its 88k "spill" was procGTT 0.32, barely over the
0.30 threshold, and that reading proved noise-sensitive. Direct loads measured:

| ctx | backend | procVRAM | procGTT | card free | tg t/s |
|---:|---|---:|---:|---:|---:|
| 96k | Vulkan | 22.25 | **0.21** | 0.61 | 189.8 |
| 80k | Vulkan | 22.03 | 0.20 | — | — |
| 80k | ROCm | 22.50 | 0.00 | — | — |
| 70k | ROCm | 22.33 | 0.00 | — | — |

**Shipped: 96k on Vulkan, 70k on ROCm.** ROCm carries ~0.47 GiB more VRAM for the
same config (22.50 vs 22.03 at 80k), so it gets less context. 96k on Vulkan runs
at 189.8 t/s — identical to the 32k figure, no degradation from depth.

Trust the direct-load numbers over the binary search for this model; re-verify if
the desktop's GPU usage changes materially.

**Is it worth using over IQ4_NL?** 189.9 vs 204.9 t/s — only **7 % slower** for a
materially better quant. The real trade is context: **80k vs 256k**.

### Files created

| file | backend | model | n-max | ctx |
|---|---|---|---:|---:|
| `preset-35b-q4km-vulkan-q8-96k.sh` | Vulkan | 35B-A3B Q4_K_M | 3 | 96k |
| `preset-35b-q4km-rocm-q8-70k.sh` | ROCm | 35B-A3B Q4_K_M | 3 | 70k |
| `preset-35b-iq4-rocm-q8-256k.sh` | ROCm | 35B-A3B IQ4_NL | 3 | 256k |
| `preset-27b-q4km-rocm-q8-128k.sh` | ROCm | 27B dense | 2 | 128k |

ROCm context verification: MoE IQ4_NL @256k = 21.69 GiB / 0.00 GTT; dense 27B
@128k = 21.78 GiB / 0.00 GTT.

### Also changed

- `bin/llama-launch` — menu split into `3) Presets ROCm` / `4) Presets Vulkan`,
  four entries each, decode t/s shown inline. All eight paths dry-run verified to
  resolve to the correct binary, model, context, spec-type, n-max and ubatch, with
  `--device Vulkan0` present only on the Vulkan branch.
- `src/benchmarks/ctx_headroom.py:105` — was hardcoded to the BeeLlama Vulkan
  binary. Headroom must be scanned against the binary the profiles actually use,
  so it now points at mainline (`matrix.BIN_VULKAN`).
