# Optimization audit — 2026-08-16

Scope: the full `llama_server/` setup, both production llama.cpp trees, live server
logs, and current hardware state. Focus: **prompt processing (prefill) and decode**,
per your request. No model-quantization suggestions in here.

Method: read every profile, `src/server/*`, the launcher, all audit docs, the
10358→10450 commit range in both local trees, live `/slots` + log timings, GPU/PCIe/
sysfs state, and upstream (llama.cpp, ROCm, known issues).

Caveats:

- Several files were **edited mid-audit** (all four `preset-3[68]*-dual-*` profiles
  touched ~01:02, adding `--ctx-checkpoints 8` + larger `--cache-ram`). I read the
  current state, but anything you're still editing may have moved.
- The README is partly model-authored (you said Qwen3.6 did a recent pass). Where
  README/launcher/profile numbers disagree, I say which source I believe and why —
  don't treat the README's tables as ground truth.
- "Live" numbers below are from the running Qwen3.8 27B ROCm dual server at the time
  of the audit.

---

## TL;DR — the five things worth doing, ranked

| # | Action | Expected effect | Effort | Risk |
| --- | -------- | ----------------- | -------- | ------ |
| 1 | **Build b10450 + one-line unsafe-math patch, benchmark, unpin** | Reclaim the 10358 performance *and* get ~10 server/spec/ROCm fixes (incl. the media slot save/restore work that may fix your post-image regression) | 1 build + 1 bench | Low (keep 10358 as fallback) |
| 2 | **Fix the launcher-generated Qwen3.8 profiles** (missing `--ctx-checkpoints 8`, `MTMD_BACKEND_DEVICE`, `--batch-size 6144`, `--main-gpu 1` on the Vulkan path) | Stops silently regressing the exact cache-invalidation bug you just fixed, and the XTX headroom | ~15 min edit | None |
| 3 | **38-vulkan preset: add `--main-gpu 1` + `MTMD_BACKEND_DEVICE=Vulkan1`** | ~1 GiB permanent XTX headroom + vision transients off the constrained card (measured on the sibling ROCm profile) | 2 lines | Low |
| 4 | **`--cache-reuse 256` on both 38 profiles** | +3% prefill measured on the 35B ROCm profile; never tested on 38 | 1 line + 1 bench | None |
| 5 | **Post your post-image-regression data to upstream issue #26873** | Unblocks an actual fix; your ROCm measurements are the only non-CUDA reproduction and rule-outs are unusually thorough | 30 min | None |

Everything else in this doc is secondary: config hygiene, a ROCm-upgrade question,
deep-context behavior notes (your 152k observation), and cleanup.

---

## 1. The llama.cpp pin is breakable — and breaking it is the biggest lever

### What you have

- Production: `llama.cpp` @ **b10358** (`030ebb558`), ROCm build (`gfx1100;gfx1201`,
  Release) + Vulkan build. BeeLlama v0.3.2 retained for KVarN (superseded).
- `llama.cpp-next` @ `ece963f41` = **b10450**, the *current* upstream release (verified
  by fetching tags today; b10450 is literally the latest release, tagged Aug 15).
- Your measurement: b10450 on this box = **−4.8% prefill, −4.0% decode, MTP acceptance
  61.9% → 55.7%**, attributed to `ggml-hip: remove -funsafe-math-optimizations (#26696)`.

### What upstream actually did (verified in the local trees)

1. **`#26696` is the *only* ggml-hip change between b10358 and b10450.** The full
   10358→10450 diff touching `ggml/src/ggml-hip/` is exactly that one commit
   (`e79e4bf66`). Your attribution is almost certainly correct.
2. **b10358 has the flag unconditionally**:
   `ggml/src/ggml-hip/CMakeLists.txt:130` →
   `set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")`
3. **b10450 removed it entirely.** The merged PR *intended* an opt-in CMake option
   (`GGML_HIP_UNSAFE_MATH`, default OFF) — that option is **not** in the final tree;
   I grepped b10450's `ggml/CMakeLists.txt` and the hip CMakeLists: nothing. The flag
   is gone, no knob.
4. The removal's stated reason: `-funsafe-math-optimizations` enables
   `-fassociative-math`, which reassociates FP reductions and made MTP greedy output
   at temp 0 diverge from the non-speculative baseline on gfx1151 (AMD's
   AIESW-40114 determinism requirement). AMD engineer IMbackK has an **open issue
   #26982** ("check why the perturbation from -funsafe-math-optimizations is so
   large") — i.e. AMD knows the perturbation is anomalously large and is looking.
   A *surgical* upstream fix (reassociation only in the hot kernels, or per-op
   control) could one day give you both determinism and speed; until then, the
   pragmatic move is your own.
5. **What else 10358→10450 contains** (the reasons to unpin once perf is restored):
   - `server: support slot save/restore with media inputs (#26640)` — the prompt-cache
     state machinery now serializes **image chunks** into saved slot state. This is
     exactly the subsystem your post-image prefill regression lives in (see §6);
     it's the single most likely candidate for a fix or for cleaner reproduction.
   - `server: re-design yield_to_queue thread model (#27133)` — worker/main-thread
     swap; can change latency behavior under load.
   - `spec: auto-detect mtp draft model type (#27005)`, `spec: enable backend
     sampling for both dflash & dspark (#26958)`, `spec: update speculative-simple`.
   - `ggml: recurrent state rollback for ggml_ssm_scan (#26623)` — CPU/CUDA only
     today (HIP/Vulkan fall back to CPU), but relevant when you run the Nemotron
     3.5 Lightning candidate: on your ROCm dual build that path would silently
     execute the rollback on CPU.
   - `common: --mmap/--no-mmap → --load-mode (#26934)` — your `--no-mmap` becomes a
     deprecated warning (still works). Values: `auto | none | mmap | mlock | dio`.
     Plan to migrate profiles to `--load-mode none` when you unpin.
   - Vulkan changes in the range are minor (Intel coopmat gating, TQ2_0 support,
     ssm_scan rollback) — nothing that changes your Vulkan numbers either way.

### Concrete plan

```bash
# in llama.cpp-next (already at b10450 = ece963f41)
mkdir -p build-fast
cmake -B build-fast -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201"
# one-line local patch: restore fast math (identical to b10358's line)
#   ggml/src/ggml-hip/CMakeLists.txt, near the old line 130:
#     set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")
cmake --build build-fast -j16
```

Then run your existing `llama-benchmark-mtp` / Pi-fixed-seed decode harness against:

1. b10358 (baseline),
2. b10450 stock (your known −4.8% control),
3. b10450 + patch.

**Decision rule:** if (3) lands within noise of (1) on both pp and tg (and acceptance
is back near 61.9% on 38-27B), switch production to (3) and delete the pin. You keep
every server-side fix above at zero perf cost. If (3) does *not* match (1), you've
localized the regression to one of the ~90 non-hip commits (server thread-model,
spec code) and can bisect with far less uncertainty than today.

Note for the record: your own data shows the unsafe-math build also gets **better
MTP acceptance** (61.9% vs 55.7%). The determinism argument against the flag is a
temp-0-greedy-identity argument; at your production temps (0.6–1.0, random seed)
identity to a non-speculative baseline is not a requirement you have, and the
acceptance gain is worth more to you than bit-exactness.

### Secondary: ROCm 7.1 → 7.2+ (optional, lower priority)

Your ROCm is the **distro-shipped 7.1** (Ubuntu resolute packages, LLVM 21).

- AMD's ROCm 7.2 for Radeon/Ryzen (Jan 2026) lists the R9700 as supported and made
  RDNA4 first-class; ROCm 7.1.1 had a known RDNA4 bug where **HIP left the GPU pinned
  at 100% utilization after idle** (ROCm/ROCm#5706, closed) — I checked your GPUs
  live: 3%/17% busy at idle, so you're not hitting that one, but the fix lineage
  matters.
- Complication: AMD's official 7.2 install path for Radeon/Ryzen targets
  **Ubuntu 22.04.3**, not your distro, and ROCm userspace wants to match the kernel
  amdgpu driver. Upgrading on a desktop that drives a 5K display is a
  driver-stack surgery, not a package bump.
- Recommendation: **do this after** the llama.cpp unpin (so each variable is
  isolated), and only with a full backup plan (the 24200 MiB XTX ceiling means a
  broken driver = a frozen desktop until you can SSH in). Worth tracking, not worth
  doing tonight.

---

## 2. The interactive launcher silently un-does your fixes (concrete bugs)

`bin/llama-launch`'s `prepare_profile_output` (the generator behind
`interactive-profile.sh`) has drifted from the hand-tuned presets. Current state as
of the audit:

| Generated custom profile | Missing vs the tuned preset | Consequence |
| --- | --- | --- |
| Qwen3.8 27B, **ROCm** | `--ctx-checkpoints 8`; `export MTMD_BACKEND_DEVICE=ROCm1`; `--batch-size 6144` (uses 2048) | **Regresses the 119k-token full-reprocess bug you just fixed** (entries = 32×690 MiB ≫ cache-ram → `server_prompt_cache::alloc()` silently skips the context); vision tower lands on the constrained XTX (+1.1 GiB + transients); loses the +12.3% prefill batch-size win |
| Qwen3.8 27B, **Vulkan** | `--ctx-checkpoints 8`; `--main-gpu 1`; `MTMD_BACKEND_DEVICE=Vulkan1` | Same cache regression; same XTX headroom problem |
| Qwen3.6 35B (any custom path) | hardcoded `LLAMA_UBATCH=512` for 38 only, but custom 35B paths inherit `default.sh`'s 1024 — OK today, fragile: any future default change silently moves untested VRAM profiles (you made exactly this mistake-avoidance decision for the q5_1 presets in the 08-11 audit) | Latent |
| Muse Glimmer menu item (option 6) | points at `models/Muse-Glimmer-30B-UD-Q5_K_M.gguf`, **which no longer exists** (you have `Muse-Glimmer-30B-Q8_0.gguf`) | `llama_validate_profile` fails with "model is not readable" — broken menu entry |

Also note: the live `interactive-profile.sh` currently just `source`s the tuned
preset, so *today's* running server is fine — the breakage hits the moment someone
walks the custom decision tree for the 27B model.

Fix: make `prepare_profile_output` emit the same load-bearing values as the presets
(easiest: for `qwen38-27b`, source the matching preset file instead of re-deriving,
and apply only the user's actual overrides — the preset-sourcing pattern the tool
already uses for the Presets branch). And either point menu option 6 at the Q8_0
file or remove it.

---

## 3. `preset-38-27b-q8-vulkan-dual` is missing the settings its ROCm sibling earned

The 38-ROCm preset documents, with numbers, that `MTMD_BACKEND_DEVICE=ROCm1` +
`--main-gpu 1` keep ~1.11 GiB permanent + ~0.9 GiB of vision transients **off the
XTX** at zero text pp/tg cost. The Vulkan preset (2565 bytes, rewritten today) still
lacks the equivalents:

```bash
export MTMD_BACKEND_DEVICE=Vulkan1     # Vulkan1 = R9700 (verified: vulkaninfo GPU1)
LLAMA_EXTRA_ARGS=( ... --main-gpu 1 ... )
```

I verified the device mapping while I was in there (because the R9700 arrived mid-docs):

- **Vulkan:** GPU0 = XTX (RADV NAVI31), **GPU1 = R9700** (RADV GFX1201), GPU2 = iGPU,
  GPU3 = llvmpipe.
- **ROCm/HIP:** agent order from `rocminfo` BDFIDs → device 0 = XTX (`0000:03:00.0`),
  device 1 = R9700 (`0000:14:00.0`). Your `ROCm0/ROCm1` docs are correct.
- **DRM card numbers disagree with both** (card0 = R9700, card1 = XTX). Harmless for
  the backends, but anything reading `/sys/class/drm/card*` (your spill
  heuristics, `xtx_predict.py`'s `DESKTOP_MIB`) must not assume card0 = the compute
  partner.
- The old 2-device Vulkan block in `llama-server.log` (Vulkan1 = "Raphael
  Mendocino", 32 GiB shared-heap iGPU) is a pre-R9700 artifact — ignore it; current
  enumeration has three real GPUs.

While there: the 38-vulkan header says "786.6 pp / 40.3 tg" and the launcher calls it
**UNTUNED** — that's the only one of your four dual profiles without a measured
`n_max` sweep, `ubatch` sweep, or `batch-size` sweep on the *dual* topology. The
36-MoE numbers changed direction between single- and dual-GPU (n_max 3→2 became the
winner once `--tensor-split` went live), so single-GPU-era optima are not portable to
this profile. A one-evening sweep would fill the gap.

---

## 4. Untested micro-wins worth a bench each (cheap, additive)

1. **`--cache-reuse 256` on both 38 profiles.** Measured +3% prefill on the 35B ROCm
   dual profile (2623→2702); the 38 profiles don't carry it. One line, no VRAM cost.
2. **f16 KV on Qwen3.8 27B — only if XTX headroom appears.** At `ts 1,1` it won't
   fit (≈+4 GiB per card; XTX currently has ~2.6 GiB free of the ~24200 MiB ceiling).
   If the monitor ever moves to the iGPU, or if you're willing to skew the split
   toward the R9700 (e.g. `0.7,1.3`, which your 33,31 experiments say costs decode),
   f16 KV is worth testing: on the 35B it bought +17% prefill on Vulkan and ~+3%
   decode on ROCm. This is a KV-cache precision change, not a model quant — same
   class of decision you already made for the 35B.
3. **`n_max` re-sweep at deep context.** Your production session sits at ~150k
   tokens, where the verify pass reads a much bigger KV. Your n_max optima were all
   measured at shallow context (and you noted "speculation economics shift with KV
   depth"). At 150k, `n_max=3` on the 27B (or 2 on the MoE) may no longer be the
   optimum — or MTP itself may net less. One sweep per model at ~120k/160k would
   tell you whether the production decode profile is actually optimal where you live.
4. **`--spec-draft-p-min` is pinned at 0.0 in every profile.** It's a legitimate
   early-rejection lever (skip drafting when confidence is low); you've never swept
   it. Low priority given MTP acceptance is already good.

---

## 5. Config / hygiene (no perf, but will bite)

1. **`default.sh` is the wrong document now.** It claims (in a header comment) to
   "track the tuned Q8 MoE preset (preset-36-35b-a3b-q8-rocm-dual-q8-256k-mmproj.sh)"
   but is actually a single-GPU Vulkan config with q8_0 KV, `LLAMA_TIMEOUT=0`,
   `PROFILE_NAME=...255k` at a 262144 context, and an `n_max` comment that says
   "optimum is 3" directly above `LLAMA_SPEC_DRAFT_N_MAX=2`. Because every preset
   sources it, it is the *de facto* defaults file — which means:
   - `LLAMA_TIMEOUT=0` is the inherited default. You discovered this exact footgun
     (zero-second socket timeout → silent empty 400 on large bodies) and fixed it in
     every dual profile, but `default.sh` itself still ships 0. Any profile that
     forgets to set it (or a future custom profile) inherits the bug. **Set
     `LLAMA_TIMEOUT=3600` in `default.sh`** — you want "safe default", not "fixed in
     four places".
   - The stale comments will mislead the next editor (or the next model). Rewrite
     the header to say what it actually is: "shared defaults sourced by all
     profiles; single-GPU Vulkan fallback config."
2. **README tables are inconsistent with the sources of truth** (expected, given the
   model-authored pass). Specific contradictions found:
   - "Qwen3.8 27B … — / **81.5**" decode: 81.5 is the *35B MoE ROCm* decode from the
     row above; the launcher (which you trust for menu numbers) says **38.0 tg** for
     38-ROCm and **40.3 tg** for 38-Vulkan. Also the "27B Dense (Vulkan) n_max 3 →
     76.3" entry in the MTP table is the *Qwen3.6-27B Q4_K_M single-GPU* number.
   - README says the production default is "f16 KV · ubatch 512"; `default.sh` is
     q8_0 KV · ubatch 1024.
   - `docs/SETUP.md` is pre-R9700 (single XTX, old model list, "Ubuntu 26.04").
   - `preset-38-27b-q8-rocm-dual...sh` references `docs/QWEN38_DUAL_GPU_AUDIT.md`,
     which **does not exist** — the audit that justifies its load-bearing values
     (batch 6144, ts 1,1, n_max 3, main-gpu 1) is missing from the tree.
   Suggest: one human pass that reconciles README ↔ launcher ↔ profile headers, and
   either write the Qwen38 audit doc or remove the reference.
3. **Disk/cleanup (not perf):** the BeeLlama tree + two 1.3 GB builds are declared
   "superseded / legacy" in three places; the two dead `*-DFlash-IQ4_XS` sidecars
   (if still around) are ~1.2 GB of files that error on load; `tools/` is empty;
   `.hypothesis/`, `.pytest_cache/` are test litter. Optional, but it reduces the
   chance of someone pointing a profile at a legacy binary by accident.
4. **Watch out for the deprecated `--no-mmap`** when you unpin (see §1.5) — migrate
   profiles to `--load-mode none` in the same pass.

---

## 6. The post-image prefill regression — your data is an upstream contribution

Your documented bug (one image request → text prefill permanently −31% on 35B / −40%
on 27B until restart; batching, thermals, mmproj placement, prompt cache, and GTT
spill all ruled out) matches **open upstream issue ggml-org/llama.cpp#26873** —
filed Aug 10 by a llama.cpp maintainer on a CUDA box (2×5070 Ti, Muse Glimmer):
first mmproj use costs +1.1 GB on the mmproj card and drops prefill ~3000→~2000.
Label: `bug-unconfirmed`, no fix yet.

Why you should act:

- Your ROCm reproduction is the **only cross-backend data point** (theirs is CUDA),
  and your rule-outs go further than the issue thread. A comment with your
  measurements (both models, both backends, the before/after pp numbers, the ruled-
  out list) materially raises the chance it gets triaged.
- `#26640` (slot save/restore with media inputs) landed in b10450 — the same
  subsystem. Re-testing your exact repro on the §1 build (b10450 + fast-math patch)
  is a free experiment: if the regression shrinks or disappears, you've also
  localized it for the issue.
- Until it's fixed, your workarounds are correct and I'd keep them: text-only
  profiles for image-free sessions, restart after heavy image use. One refinement —
  since the regression is per-server-instance, a `llamactl restart` is the cheapest
  "reset"; if image sessions are frequent, consider an auto-restart hook keyed on
  the first `/v1/chat/completions` containing `image_url` (your `llama-proxy`
  already tees all traffic, so the hook could live there with one regex).

---

## 7. Deep-context reality check (your 152k observation, quantified)

Live session at audit time: **151,820 prompt tokens**, decode ≈ 17–21 t/s,
3.1k-token prompt eval ≈ 233–270 t/s, MTP acceptance 0.49–0.63 (mean drafted length
2.5–2.9). Your documented 38 tg / 1086 pp numbers are shallow-context numbers.
At ~150k the gap is expected physics, not a config regression:

- The 27B is hybrid (16 of 65 layers full-attention), so attention cost grows with
  context only on those 16 layers — but at 150k tokens that's ~16 × 150k ×
  (2×8×128) bytes of KV read per new token per layer on the decode path, on a card
  that's *also* the desktop's. 20 t/s at 150k is plausible for Q8 dense on this
  split; the Qwen3.6 MoE at the same depth would hold up much better (3B active
  params) — if you're choosing profiles by "feels slow at 150k", the MoE is the
  right answer there, full stop.
- Prefill at depth scales with existing context: 233 t/s for new tokens at 151k
  depth vs 1086 at shallow is the expected super-linear-ish drop.
- Practical levers at depth (no quantization involved):
  1. Keep the prompt cache hot — the 01:02 `--ctx-checkpoints 8` + `cache-ram 24576`
     fix is doing its job in the live log (f_sim 0.98–0.999, no full reprocesses in
     the last hour; previously 8× 119k-token reprocesses in one session).
  2. Re-sweep `n_max` at depth (§4.3) — the optimum can move.
  3. If a session is approaching 200k+, that's the point where the 35B MoE profile
     (256k headroom, f16 KV, ~2× decode of the 27B at shallow, far better scaling)
     wins on both speed and VRAM-per-token of *useful* work.

---

## 8. Verified-not-issues (so you don't chase these)

- **GPU device numbering** (Vulkan0/1, ROCm0/1): correct as documented *today*,
  post-R9700. The scary 2-GPU log block is stale.
- **`-t 16 -tb 32` on the 9950X3D**: governor is `performance` on all 32 threads,
  max 5998 MHz, single NUMA node. Nothing to gain here.
- **CPU-side model residency**: with `-ngl 99` the server's RSS is ~8 GiB (KV
  bookkeeping, compute buffers, prompt-cache RAM), *not* the 28 GiB file — the model
  lives in VRAM. The 6.9 GiB in swap is mostly the QEMU VM and browsers, not the
  server (server `VmSwap` = 0).
- **R9700 stuck-at-100%-idle** (ROCm 7.1.1 RDNA4 bug, ROCm/ROCm#5706): not present
  on your box (3%/17% busy at idle).
- **`GGML_HIP_RCCL=OFF`**: correct for inference multi-GPU (llama.cpp does its own
  P2P copies; RCCL is for training-style collectives).
- **`--flash-attn on` everywhere**: right call; your KVarN audit already established
  the FA whitelist traps (q3/q2/q6 KV silently drops FA on Vulkan).
- **`--parallel 1` on the hybrid models**: correct (recurrent-state buffers ×4 for
  nothing, measured in the 38-rocm header).

---

## 9. Suggested execution order

1. (15 min) `default.sh`: `LLAMA_TIMEOUT=3600`, fix the stale header + `n_max`
   comment; fix the 38-vulkan preset (§3: two lines); add `--cache-reuse 256` to
   both 38 profiles (§4.1).
2. (30 min) `llama-launch`: make the 38 paths source their presets (or emit
   `--ctx-checkpoints 8`, `MTMD_BACKEND_DEVICE`, `--batch-size 6144`, `--main-gpu 1`
   per backend); fix/remove the Muse menu entry (§2).
3. (1–2 h) Build b10450 + the one-line fast-math patch; run the 3-way bench (§1);
   if it matches 10358, switch production and migrate `--no-mmap` → `--load-mode none`.
4. (30 min) Comment on upstream #26873 with your ROCm repro; re-test the regression
   on the new build in the same pass.
5. (one evening, optional) Deep-context `n_max` sweep per model at ~120k/160k;
   38-vulkan dual-GPU tuning sweep; f16-KV-on-27B experiment if XTX headroom appears.
6. (whenever, careful) ROCm 7.2+ evaluation as a *separate* variable, after the
   llama.cpp unpin.
