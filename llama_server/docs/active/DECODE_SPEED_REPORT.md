# Decode speed deep dive - Qwen3.8-27B dual AMD (open-ended investigation)

Started 2026-08-2x (user away; keep working until told to stop).
Target: decode t/s of the *currently running* setup and of the profiles used
for interactive/agent work. Prefill is out of scope except where a knob trades
off (note it, do not optimize it).

## Setup (ground truth)

| | |
|---|---|
| GPU0 (ROCm0/Vulkan0) | RX 7900 XTX, gfx1100, 24,560 MiB. Drives desktop. HARD CEILING 24,200 MiB (GTT collapse, no crash). |
| GPU1 (ROCm1/Vulkan1) | AI PRO R9700, gfx1201, 32,624 MiB, Gen4 x4 via chipset. No P2P, no xGMI. |
| CPU/RAM | Ryzen 9 9950X3D 16C/32T, 64 GB |
| OS/kernel | Ubuntu 26.04, kernel 7.0.0-30-generic, in-tree amdgpu |
| Driver | Mesa 26.0.3 (RADV), ROCm 7.1 (Ubuntu packaging), RCCL broken (segfaults 2-GPU allreduce) |
| CPU governor | performance (already set). THP = always. |
| Model | Qwen3.8-27B dense hybrid (SSM/DeltaNet) + built-in MTP head + vision. GGUF: Q8_0 (26.9 GiB), Q6_K (21.3 GiB, imatrix) |

### Builds on disk

| Worktree | Commit | Date | Role |
|---|---|---|---|
| `llama.cpp` | 030ebb558 (master) | 2026-08-10 | pinned ROCm + stock Vulkan (b10358) |
| `llama.cpp-vulkan-tp` | 13ac90156 (PR #25051 head) | forked ~2026-06-29, **766 commits behind master** | the ONLY tensor-capable Vulkan build |
| `llama.cpp-next` | ece963f41 (master) | 2026-08-15 | 136 commits behind current master |

### Currently RUNNING (measured from /proc + live log)

Qwen3.8-27B **Q8_0**, **Vulkan tensor** (PR #25051 build), `--split-mode tensor
--tensor-split 50,50 --main-gpu 1`, f16 KV, MTP n3 (draft q8_0 KV), c=216064,
ub 1024, batch 2048, --no-mmap, mmproj, temp 1.0.
Live log: **32-41 t/s**, acceptance 0.51-0.70 (profile header: 40.29 @ 54.5%).

### Best decode measured on this box (own data, 2026-08-21)

ROCm tensor, **Q6_K**, ts **58/42**, f16 KV, MTP n3, c=192512,
`GGML_CUDA_DISABLE_GRAPHS=1` -> **53-54 t/s live** (49-54). Profile:
`preset-38-27b-q6k-rocm-tensor-decode-58-42-188k.sh`. Unspeculated tg 32.84.

## Findings (status: [measured] [online-verified] [candidate - untested])

### F1. [measured, local] The running profile is the prefill variant, not the decode variant
The live server is `preset-38-27b-q8-vulkan-tensor-decode-200k.sh`, whose own
header says: "WHEN TO PICK THIS OVER THE ROCm TENSOR PROFILE: prefill-dominated
work... For interactive/agent use, ROCm still wins". Measured decode 40.3 t/s
(Vulkan tensor Q8) vs 53-54 t/s (ROCm tensor Q6 58/42). **~+30% decode by
switching the daily profile to the ROCm Q6_K 58/42 one.** Tradeoffs: -27k
context (216k -> 192k), Q6_K vs Q8_0 quality, prefill drops 565 -> ~415 t/s.
User approval needed before switching anything.

### F2. [online-verified] Rebuild `llama.cpp-vulkan-tp` on current master.
PR #25051 head 13ac90156 is still the PR tip (no new commits since 2026-08-16),
but its BASE is ~766 commits behind master b21e4de74 (2026-08-22). Merged since
its base that touch the tensor decode path:
- **#24710** (merged 2026-07-10): "make tensor-split regex patterns static to
  avoid per-token recompilation". Author profiled `-sm tensor` on consumer
  PCIe GPUs: **~30% of the decode thread in std::regex compilation** under
  `llama_meta_device_get_split_state()`, GPUs idle-waiting. This is the exact
  shape of this box. The current Vulkan TP build predates the fix.
- **#24814** (merged 2026-07-05): "Fix stale tensor-split params for draft
  models" (Eagle3/DFlash + tensor crash; same meta-backend component).
- #25005, #26081 and others.
Action: rebase the 14 PR commits onto master, rebuild, re-run the 2026-08-21
decode matrix (llama-bench -r3, MTP off) + one live MTP sample. The PR is a
draft and unmerged; rebase may need conflict work (commits touch
ggml-backend-meta.cpp which changed in #26502/#27433 merge+revert).

### F3. [online-verified] New ROCm build: upstream now has `GGML_HIP_UNSAFE_MATH`
#26696 (merged) made `-funsafe-math-optimizations` an opt-in CMake flag
(default OFF). This is why b10450 measured -4.8% pp / -4.0% tg. The
TUNING_QUEUE work item "b10450 + one-line -funsafe-math patch" is now just:
rebuild `llama.cpp` at current master with `-DGGML_HIP_UNSAFE_MATH=ON`.
Current master is 136+ commits past llama.cpp-next, 12 days past the pin.
Gain unknown: must re-run matrix; b10450-with-flag should at least match pin.
Note: #26696 also added a gfx1151 CI check asserting MTP stays byte-identical
at temp 0 - suggests unsafe-math + MTP has a correctness caveat worth testing
(temperature-0 greedy compare) after any upgrade.

### F4. [candidate] Open PRs that touch MTP decode (not merged, cherry-pick candidates)
- **#27210** "spec: add adaptive MTP draft depth (draft-mtp-adaptive)" - open,
  updated 2026-08-22. Mainline already has `--adaptive-target` (off in prod).
  Their README: adaptive can vary effective depth per step within the n_max
  ceiling. On dense 27B, content dominates acceptance (code ~2x prose); an
  adaptive depth could in principle capture both ends. UNTESTED here.
- **#27489** "ggml: reuse compute buffers for MTP" - open. Compute-buffer
  allocation is per-step overhead; could cut decode latency. UNTESTED.
- **#26827** "fix(mtp): serialize multi-ubatch decode execution" - open, fix.
- **#27368** crash fix for draft-mtp under KV-cache saturation - relevant to
  the 216k profile stability (crash, not speed).
- **#25319** "ggml: asynchronous scheduler memory copies" - open draft, could
  matter for the CPU-proxy allreduce path (decode does ~2 small allreduces per
  layer per step over host memory).

### F5. [candidate] n_max > 8 under tensor is currently blocked by an upstream bug
Documented locally: `-sm tensor --spec-draft-n-max 10/12/16` hits
`ggml-backend-meta.cpp:1576 GGML_ASSERT(meta_buf_ctx->bufs[i])`. #24549 (open)
and #24566 (open draft) target the same component. If either lands, the n_max
ceiling on tensor opens up - but p_min/n8/n16 were already tested and REJECTED
on layer split (-6 to -12%), so value here is uncertain. Re-test only if the
fix merges and a quick matrix is cheap.

### F6. [tested, null/rejected - do not re-recommend]
~45 knobs, all measured locally: all GGML_VK_* envs, all GGML_CUDA_* AR envs,
RCCL (segfault), -t 1..32, -ub 64..2048, q8_0 KV for speed, PCIe ASPM
performance, n16 p0.8 / n8 p0.8 p_min (negative on this model), n_max 4 on
dense-27B Vulkan, b10450 without unsafe-math. See
docs/active/AMD_DUAL_GPU_TUNING.md for the full null table.

### F7. [candidate, system] Mesa 26.0.3 is an Ubuntu package; check upstream Mesa
26.1/27 for RADV RDNA3/RDNA4 decode fixes (flash attention, coopmat2, pipeline
capture). A driver upgrade is the only lever on the Vulkan path left after
F2. Needs user OK (system package upgrade).

### F8. [candidate, system] Kernel 7.0.0-30 in-tree amdgpu: check for RDNA3
GTT/spill and HMM fixes; also check if `amdgpu.gart_size` / `vramlimit` or
`cgroup` pinning changes the XTX ceiling behavior. Low priority: the ceiling is
a design constraint (desktop on XTX), not a driver bug.

### F9. [measured, local analysis] `llama.cpp-next` (ece963f41, 2026-08-15) sits BETWEEN pin and master
It is 92 commits over the pinned 030ebb558 and 136 under current master b21e4de74.
Critically it CONTAINS `e79e4bf66 ggml-hip: remove -funsafe-math-optimizations
(#26696)` - i.e. the next-line build is the SLOW one (same -4.8%/-4.0% regression
the box saw on b10450). Anything benchmarked against `llama.cpp-next` must carry
`-DGGML_HIP_UNSAFE_MATH=ON` (flag exists from #26696; default OFF) to be comparable.
Also in pin..next, decode-relevant: #27133 server yield_to_queue thread-model
re-design, #27041 metrics during decode, #27005 auto-detect MTP draft type,
#26623 ggml_ssm_scan recurrent-state rollback. All ride along for free with an
F3 master rebuild.

### F10. [online-verified] Upstream docs/multi-gpu.md (as of pin) confirms:
- tensor mode REQUIRES flash-attn (running server has --flash-attn on: good).
- tensor + quantized KV cache is officially "not implemented" - the Vulkan TP
  build accepting q8_0 KV under tensor is running undocumented behavior; keep f16.
- RCCL is deliberately off by default upstream ("not universally beneficial");
  matches the local segfault + null findings. No action.
- row split is now deprecated upstream in favor of tensor. Layer remains the
  prefill workhorse on this box (interconnect-bound tensor prefill).

### F11. [candidate, untested] PCIe link-width verification (read-only)
The docs assume the R9700 runs at Gen4 x4 via chipset. If it ever negotiated
lower (x2/x1 or Gen3) tensor decode drops with it. `lspci -vvv` LnkSta check is
untried on this box and costs nothing.

### F12. [candidate, untested] CPU thread placement for the decode loop
`-t 1..32` was swept (null) but CPU AFFINITY was never tested. The 9950X3D has
one 8-core X3D CCD; the meta-backend decode thread (allreduce staging via
pinned host memory) is a single-thread hot loop. taskset'ing the llama-server
process onto the X3D CCD is untried. Cheap, reversible, no GPU restart needed
for a quick A/B on the next normal restart.

### F13. [measured, local sysfs 2026-08-2x] PCIe TRUTH differs from the docs, twice
Negotiated link state per hop (sysfs current_link_speed/width, stable across 3 samples):

R9700 path (lspci -t): IOD -> [02:02.1 internal] -> 05:00.0 (1022:43f4 chipset) ->
0c:08.0 (1022:43f5 chipset) -> 12:00.0 (1002:1478) -> 13:00.0 (1002:1479) -> 14:00.0 (R9700)

| hop | id | negotiated |
|---|---|---|
| 05:00.0 | 1022:43f4 | 16.0 GT/s x4 (Gen4 x4) |
| 0c:08.0 | 1022:43f5 | 16.0 GT/s x4 |
| 12:00.0 | 1002:1478 | 16.0 GT/s x4 |
| 13:00.0 | 1002:1479 | **32.0 GT/s x16** |
| 14:00.0 | R9700 | **32.0 GT/s x16** |

Effective end-to-end R9700 link = **Gen4 x4 ~ 15.75 GB/s** (min hop). The docs'
"~7.9 GB/s" was half the real figure (7.9 is Gen3 x4 or Gen4 x2). The GPU's own
final segment is Gen5 x16 (~63 GB/s) but is starved upstream.
XTX (03:00.0): 16.0 GT/s x16 = **Gen4 x16 ~31.5 GB/s**. The docs' "PCIe 5.0 x16"
label was wrong (9950X3D is Gen4 only) - the 31.5 GB/s number was right.
pp_dpm both GPUs: power_dpm_state=performance. R9700 sclk 2019/2350 MHz, XTX
1501/2482 MHz at idle. No AER error counters readable without root.

Implications:
- Tensor DECODE (10 KB activations/layer): unaffected at 15.75 GB/s. The
  "sync-bound not bandwidth-bound" conclusion stands.
- Tensor PREFILL (10.5 MB allreduce chunks): ceiling is 2x higher than the
  analysis assumed. Existing nulls (chunk size, wire bytes) still say sync-count
  dominates, so no software change follows - but the analysis doc should be
  corrected.
- HARDWARE option for the user: the Gen5 x16 13:00.0 segment shows a faster
  route exists in the board. If the R9700 could be reseated on a direct
  CPU Gen4 x16 slot (bypassing the chipset x4 hops), tensor prefill ceiling
  doubles to ~31.5 GB/s. Decode still unaffected. Flag only; no action here.
- A3 (link-width check) is DONE - removed from queue.

### F14. [online-verified] PR #25051 thread mining (48 comments + 10 reviews, full dump in research/raw-pr25051-comments.md)
- The PR is STALLED: author pwilkin last active 2026-08-16 ("once I'm back from
  vacation I'll revisit"). Users (2026-07-27, 2026-08-16) are asking for progress;
  leapdragon: "it's down to running quite old builds at this point or just
  sticking with ROCm tensor split mode" - exactly this box's situation.
- The multi-turn decode-crawl bug (follow-up prompts at 2-3 t/s, reproduced on
  R9700 pairs and W7900s, June 27) WAS fixed in the final commit of our build
  (13ac90156 "fix -sm tensor decode crawl"), and 0cc4m flagged it as a RADV
  driver workaround, not a root fix: "It should be reported to RADV." This is a
  concrete lead for the Mesa investigation (A4): multi-GPU timeline/cross-device
  semaphore handling on RADV.
- D2D allreduce was tried and removed (measured -90% on R9700s, power collapse);
  ring allreduce is the default in our build. Confirmed active on the live server
  (zero "AllReduce init failed / butterfly fallback" warnings in the current run).
- cattivik66 (2x W7800, separate PCIe4 x16 root complexes, no P2P - closest
  topology to ours): at 88K prompt / 164K context, tensor tg = 9 t/s vs 37 for
  row/layer. Short-context tensor wins (+25% pp, +49% tg). WARNING for the
  running 216k profile: no-P2P tensor decode may have a deep-context cliff that
  the shallow live numbers hide. (Our own matrix left "tg d65536" TODO for
  vulkan-pr tensor - this is why it matters.)
- characharm (9070 XT + R9700, our exact GPU class): tensor tg128 @ d60000 =
  10.68 vs 21.79 non-tensor. Same warning, independent box.
- wizardeur (2x XTX, matched): tensor tg128 41.57 vs layer 34.50; ROCm tensor 45.35.
  Matched fast links are where tensor decode shines; asymmetric slow links are
  where it hurts.
- TheBlueMatt: a Vulkan COMMAND-BUFFER GRAPH REUSE patch measured +25% on top of
  tensor-parallel on his 4x B60. That is open DRAFT PR #24720 (vulkan: command
  buffer graph reuse, updated 2026-06-20, NOT merged). Not in any of our builds.
  Cherry-pick candidate IF it shows a mechanism for RDNA (his numbers are Intel).
- JohannesGaessler (core dev): the meta backend's 1:1 buffer-type-to-physical
  mapping is the architectural limiter; a proper refactor would also fix TP
  compute-buffer over-allocation. Out of scope for #25051; watch for a future
  upstream PR - would be the real fix for tensor-mode inefficiency.
- MTP under tensor: draft head runs with devices=[default] (NOT tensor-split);
  backend sampling is unavailable under tensor (see F15).

### F15. [measured, local log] MTP draft model uses a CPU sampler under -sm tensor
Current server log, model load:
  `common_speculative_impl_draft_mtp: - n_max=3, n_min=0, p_min=0.00, backend_sampling=1`
  `set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU`
  `common_speculative_impl_draft_mtp: backend offload failed for seq_id=0; using CPU sampler`
So every draft+target sample is done on CPU (151k vocab, top-k 20) on the
running profile, while the ROCm profile (best decode) gets backend (GPU)
sampling. That is a previously unisolated component of the Vulkan-vs-ROCm decode
gap and of A11 (sampler cost). Merging #24710 (regex) + this + backend sampling
support under tensor are the three decode-path deltas a rebase/upgrade could
plausibly unlock.

### F16. [online-verified] PR #27210 adaptive MTP: tested on the SAME model class, big code win
PR #27210 (`--spec-type draft-mtp-adaptive --spec-draft-n-max 12`) was benchmarked by
its author on **Qwen3.8-27B Q8_0, 2x AI PRO R9700 (gfx1201), ROCm** - same model,
same GPU class as the R9700 half of this box. temp 0.6 / top-k 20 / top-p 0.95
(matches the shipped profiles), ctx 8192, 2 seeds:

| config | coding | prose | hard prose |
|---|---:|---:|---:|
| fixed n2 | 67.2 | 53.3 | 51.2 |
| fixed n3 | 78.8 | **56.4** | **52.6** |
| **adaptive n12 (floor 2)** | **86.4 (+9.6%)** | 54.4 | 51.0 |
| **adaptive n12 (floor 3)** | **85.3 (+8.2%)** | 54.8 | 51.3 |

Adaptive reached depth 9-12 on coding with 3s-spike rates to 115-130 t/s during
content recall. Prose is a tie within noise. The local fixed-n3 optimum (measured
2026-08-16) was a mixed-content harness; for the box's actual coding/agent
workload, adaptive n12 is the strongest untested single knob found so far.
Cherry-pick surface: 9 files, +446/-25, ONLY common/ + server-context + tests
(no ggml/meta changes) -> LOW-MED cherry-pick difficulty onto the pinned build.
Caveats: tested at ctx 8192 (verify cost is depth-dependent; re-test at depth),
2x R9700 not the asymmetric pair (mechanism transfers, numbers won't), open PR
(unmerged, AI-assisted disclosure - review the state machine before trusting).

### F17. [online-verified] Discussion #21043 (RDNA4 R9700 experiments, 47 comments; full dump in research/raw-disc-21043.md)
OP = 50+ experiments on a single slotted R9700 (Gen5 x16 slot), Mesa 25.2.8, b8554.

New-for-this-box levers (NOT in the local ~45-knob null table):
- **`rm_kq=1` source patch** (one line in ggml-vulkan.cpp: `uint32_t rm_kq = 1; // was 2`,
  reduces VGPR pressure on RDNA4): +0.8-1.1% RADV dense-27B decode, +13% on AMDVLK
  dense. PPL/VRAM identical. Fold into the A1/A2 rebuilds; test with the matrix.
- **AMDVLK as alternate ICD**: OP measured AMDVLK+rm_kq=1 (NO gfx queue) = 32.73 vs
  RADV 29.30 on 27B dense (+11.7%), 35B MoE +3.7% (with gfx queue). GFX queue HURTS
  AMDVLK dense (-8%) but helps AMDVLK MoE (+4.7%). Not installed on this box.
  Caveat: the PR #25051 tensor allreduce path (cross-device semaphores / CPU proxy)
  is uncharted on AMDVLK - could break, not just be slower. A/B is a controlled
  experiment via VK_ICD_FILENAMES (system default untouched).
- **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`**: OP says zero on RADV; JohnTDI (comment 18,
  below) reports a small positive on RADV in his MTP config. Untested here; one env
  var, zero risk, test at next restart of the Vulkan TP profile.
- **power_dpm_force_performance_level**: both cards currently "auto".
  digitalscream (comment 3): forcing "high" made "a huge difference to TG" on dual
  R9700; JohnTDI (comment 18): "auto remained faster than forcing high".
  Conflicting reports -> cheap sudo A/B, keep expectations modest.
- **glslc regression (comment 18)**: old system glslc 2023.8 caused a SEVERE Vulkan
  shader perf regression on Qwen3.8-27B; fixed with Vulkan SDK 1.4.357.1.
  **CHECKED OUR BUILDS: they use /bin/glslc = shaderc 2026.1-1 (current). NOT affected.**
- **R9700 BAR = full 32 GiB** (resource0 = 0x800000000) and XTX = 24 GiB:
  amittell's 256MB-BAR ub deadband (comments 12/16) does not apply to slotted cards.
- **VectorAnvil (comments 14/15)**: HIP direct-P2P allreduce patch, 2x R9700:
  +12.5% tg (7B), +9.4% tg on Qwen3.6-27B Q6_K. **N/A here: no P2P on this box**
  (XTX CPU-direct, R9700 chipset-side, different roots). Recorded so it is not
  re-proposed.
- **JohnTDI comment 18 (2026-08-16) - Qwen3.8-27B Q8_0 + MTP, single R9700, our exact
  model**: best 64.95 t/s (2x 30-request runs) with MTP **n_max 7**, **q4_0 KV**,
  ub 288, parallel 3 (three 16k slots), GGML_VK_ALLOW_GRAPHICS_QUEUE=1,
  --spec-draft-ngl all, ctx 49152. Baseline (n5, ub 512, q4_0 KV, 2x 32k slots) 59.2.
  Convergence with F16 (adaptive n12, +9.6% coding): deeper MTP helps code workloads
  on this model class; our shipped n3 at 192k ctx is the conservative end.
- ASPM: current policy = **default** (sysfs). Local test was null; d-shehu also saw
  no boost. Stays out of the queue. BIOS items (ASPM off in BIOS, Global C-States
  off) are a user-side checklist, noted here, not an agent action.
- Mesa: OP's exhaustive table shows NO decode movement from any Mesa/RADV flag
  ("Nothing moves RADV 27B decode"; 29.3 t/s = 71% of 640 GB/s BW ceiling).
  Mesa's big wins (+21/+48%) are PREFILL. A4 re-scoped accordingly.

### F18. [measured, local] Clocks under load are fine; iowait is Docker, not the LLM
R9700 during the live server run: sclk 3254 MHz, mclk 1258 MHz (level 5/5, max).
No clock-boost headroom to chase. XTX sclk table top 2482.
System iowait was ~34% at 15:37 - traced to Docker Desktop's qemu VM
(qemu-system-x86_64, virtio-blk on Docker.raw, 104% CPU), NOT the llama-server.
The server (restarted 15:13 by the user, same Vulkan TP profile, now ~117% CPU)
is CPU-bound on the decode side - consistent with the un-#24710'd build + CPU
sampler (F15) + CPU-proxy allreduce staging; reinforces A1.
ASPM policy = default (not performance); local test already null; excluded.

### F19. [online-verified] Mesa 26.0 -> 26.2 RADV delta (766 radv commits; title scan)
Almost all GFX12 graphics-path changes (HiZ, VRS, RT, video). Two commits matter
for this box's compute/multi-GPU path:
- `937bd7e0` (2026-05-15) **radv: clear the follower gang semaphore when a cmdbuf
  is reset** - "follower gang" is RADV's multi-device construct; the closest
  upstream match to the driver bug behind PR #25051's decode-crawl workaround.
  (Whether Ubuntu 26.0.3 already cherry-picked it: unknown - check on upgrade.)
- `7fa1b487` (2026-06-30) **radv: disable AMD_device_coherent_memory on gfx12 due
  to out of order behavior** - gfx12 = the R9700; a memory-ordering fix that can
  touch CPU-GPU sync paths.
No commit explicitly claims decode performance. Keep A4 as: user-approved upgrade
to 26.1/26.2 + A/B decode matrix + prefill check. Expected: low decode, maybe
prefill, correctness watch on the tensor path.

### F20. [online-verified] ROCm: Ubuntu 26.04 ships only 7.1.0; newer needs AMD's repo
Installed: libamdhip64 7.1.0-0ubuntu2 (ROCm 7.1). Upstream: 7.2.0..7.2.4
(2026-01..05) and a 7.14.0 tag (2026-07). A #21043 commenter on ROCm 7.2.1
(kernel 7.0.0-15, same kernel family) reported ROCm "actually better" than Vulkan
for Qwen3.6-27B dense - consistent with F1 (ROCm already wins decode here).
Upgrading to 7.2.x = AMD repo + kernel/driver coupling risk (A9 stays
high-effort; only after the build-side items pay off).

## ACTION QUEUE (strictly untested-on-this-box items; cross-checked against
the ~45-knob null table in AMD_DUAL_GPU_TUNING.md)

| # | Action | Why untested | Value est. | Effort | Status |
|---|--------|--------------|-----------|--------|--------|
| A1 | Rebase #25051's 14 commits onto master b21e4de74, rebuild Vulkan TP, re-run decode matrix (-r3, MTP off) + 1 live MTP sample. PR is stalled (author vacated 2026-08-16) so a local rebase is the only path to the fixes | Base is 766 commits stale; #24710 (30% of decode thread in regex, merged 2026-07-10) + #24814 are missing from the RUNNING build; #24720 (vulkan graph reuse, +25% claim on Intel) stays a separate cherry-pick eval | med-high | med (rebase conflicts in ggml-backend-meta.cpp possible; #26502 merged+reverted in that window) | not started |
| A2 | Rebuild ROCm at master with `-DGGML_HIP_UNSAFE_MATH=ON` (replaces the "one-line patch" TUNING_QUEUE item); run 2026-08-16-style matrix vs pin | Only b10450-WITHOUT-flag was tested; the flag is now first-class; 228 fresh commits | med | low-med | not started |
| A3a | ~~link-width check~~ **DONE -> F13.** R9700 = Gen4 x4 chain (15.75 GB/s; docs said 7.9), final segment Gen5 x16; XTX Gen4 x16 (docs said 5.0) | - | - | - | closed |
| A3b | dmesg amdgpu errors/throttling history (needs sudo, user present) | Never checked | low | low (read-only) | queued (user present) |
| A4 | Mesa 26.0.3 -> latest: re-scoped to PREFILL interest (F17: no decode movement from any Mesa/RADV knob per OP exhaustive table). Still worth reading 26.1/26.2 RADV changelogs for the multi-GPU semaphore bug behind the PR #25051 decode-crawl workaround | Driver layer never touched | low (prefill) / med (multi-GPU bug watch) | med (system pkg, user approval) | main (inline) |
| A5 | ~~RADV_* env sweep~~ **MOSTLY RESOLVED by F17**: OP tested RADV_DEBUG=nocompute, RADV_PERFTEST=sam/bolist/localbos/dmashaders/nircache/hic/nogttspill, RADV_PROFILE_PSTATE - all zero on RDNA4. Only survivor is queue selection, which is A13b (llama.cpp-level env). | - | - | - | closed |
| A6 | taskset llama-server onto X3D CCD (A/B on next scheduled restart, no extra GPU cost). NOTE F17: CPU pinning was tested NULL on the OP's 9900X box; X3D V-cache is the only reason it might differ here - low expectations | CPU affinity never tested on THIS box | low | low | queued (needs restart window) |
| A7 | **Cherry-pick #27210 (draft-mtp-adaptive) onto the pinned ROCm build; test `--spec-type draft-mtp-adaptive --spec-draft-n-max 12` on the ROCm tensor Q6_K 58/42 profile, fixed 4-prompt protocol + one real coding session, at shallow AND ~60k depth** (F16: +9.6% coding on 2x R9700, prose tie; only common/ files touched) | Fixed-depth sweeps only; adaptive on dense 27B never tested; MoE test (negative) does not transfer | **med-high** | med (small build + protocol) | queued (needs GPU window) |
| A8 | ~~open-PR cherry-pick eval~~ **RESOLVED**: #27210 promoted to A7; #27489 = memory-only (saves ~1 GiB, auto-enable restricted to single-CUDA-device MTP path, no speed gain, N/A here); #25319 = watch-only (API consensus PR, two prior attempts reverted, high risk); #26827 N/A (np=1); #24720 (vulkan graph reuse, +25% claim on Intel 4xB60) = watch, not cherry-pickable without risk | - | - | - | closed |
| A9 | ROCm 7.2 (or latest) vs 7.1 on Ubuntu 26.04: hipBLAS/flash-attn small-batch decode deltas | Package never upgraded | low-med | high (risk) | main (inline) |
| A10 | Context-size decode check on the RUNNING profile: 216k (XTX near ceiling, possible GTT pressure) vs 131k decode t/s | "Context is free" was measured on ROCm Q6 163-196k; Vulkan Q8 @216k near-ceiling untested | low | low (needs GPU window) | queued |
| A11 | Sampling-cost check: (a) temp 1.0 top-k 20 vs temp 0 greedy live decode on same prompt; (b) Vulkan-tensor (CPU sampler per F15) vs ROCm (GPU sampler) on the SAME model/quant/split to isolate sampler+backend contribution | Never isolated the sampler cost; F15 shows the running profile samples on CPU | low-med | low (needs GPU window) | queued |

| A12 | Deep-context tensor decode cliff check (running profile, no restart needed beyond a probe): live probe at ~60k and ~120k KV depth (llama-bench -d 65536 on the PR build after llamakill, or a long live generation) to confirm/refute the no-P2P tensor tg collapse (9-11 t/s reported by 2 other boxes) | Our own matrix left vulkan-pr tensor tg@d65536 as TODO; two independent no-P2P boxes report catastrophic deep-ctx tensor decode on the PR build | **med-high (could invalidate the running profile for long sessions)** | low (one llama-bench run) | queued (needs GPU window) |

| A13 | Force `power_dpm_force_performance_level` (high/performance) on both cards, A/B the decode matrix (F17: conflicting reports, both cards currently "auto") | Never tested here | low | low (needs sudo, 2 writes) | queued (user present) |
| A13b | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` on the Vulkan TP profile at next restart (F17: OP=zero, JohnTDI=small positive; one env var, zero risk) | Never tested here | low | trivial | queued (needs restart window) |
| A14 | Install AMDVLK as a second ICD; A/B the PR-build decode matrix via VK_ICD_FILENAMES (F17: OP +11.7% dense with rm_kq=1). Risk: PR #25051 cross-device allreduce uncharted on AMDVLK - if it breaks, ICD file removal restores status quo | AMDVLK never installed here | low-med | med (pkg install, no system default touched) | queued (user approval) |

NOT on the queue (tested null/negative locally - kept out deliberately):
GGML_VK_* envs (whole list), GGML_CUDA AR envs (whole list), RCCL, -t, -ub,
batch on Vulkan, q8_0 KV for speed, PCIe ASPM, p_min/n8/n16, n_max 4 dense-VK,
main-gpu 0 on Qwen3.8, GGML_VK_COMM_RING (already default in the PR build),
--cache-reuse (inert on hybrids), Nemotron q8_0 (model deleted).

## STALE NEXT ACTIONS (superseded by the ACTION QUEUE above)
## NEXT ACTIONS (self-prompt queue - work through top-down, then requeue)
1. Mine discussion #21043 (RDNA4 R9700 experiments) comments end-to-end. [assigned: child driver-system]
2. Mine PR #25051 full comment thread (48 comments) for maintainer guidance + perf notes. [assigned: child llamacpp-deep-dive]
3. Check Mesa upstream releases 26.1+ for RADV perf/decode fixes; check Ubuntu update path. [assigned: child driver-system]
4. Read #27210 / #27489 / #26827 diffs; assess cherry-pick difficulty onto pinned build. [main]
5. Search issues/discussions for "R9700" and "gfx1201" (newer threads). [assigned: child llamacpp-deep-dive]
6. Check llama.cpp multi-gpu.md + new doc pages for tensor mode guidance changes. [main]
7. Online: other people's Qwen3.8-27B / Qwen3-Next class dense hybrid MTP numbers on RDNA3/4 (sanity check on 40-54 t/s). [assigned: child model-config]
8. Check llama.cpp-next worktree: what 136 commits did it add over the pin; is it already built? Any decode-relevant changes? [main]
9. ROCm 7.1.52801 vs 7.2: does Ubuntu offer newer; hipBLAS/flash-attn decode changes. [assigned: child driver-system]
10. CPU-side: verify -t 16 vs 32 at batch-1 (they swept, but on tensor+MTP?), check llama-server thread placement (taskset decode thread to X3D cores). [main]

## Safety rules (user away)
- Do NOT stop/restart the running server, do NOT kill the GPUs, do NOT modify
  profiles. Local work is read-only investigation + report writing.
- Benchmarks that need a free GPU go in the report as actions for when the user
  is back, tagged with exact commands.
- Record VRAM next to every speed claim; tag every number with ctx depth and
  MTP on/off.
