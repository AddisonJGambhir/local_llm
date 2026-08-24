# Three-lever validation, 2026-08-24

Scope: the three items that survived the adversarial pass on the tuning queue:
(A17) the "empty-messages client storm", (L2) the restart-window one-liners
(Vulkan env vars + `-tb`), and (M9) the chunked GDN prefill kernel. For each:
the mechanism from the local code (llama.cpp-tp-adaptive, the running build),
the evidence for or against value, a stated prior, a test protocol, and a
falsification criterion. Online sources are cited; per the standing rule,
NVIDIA/CUDA numbers are used only as algorithm references, never as expected
value on this box.

Status per item:

| Lever | Verdict | Expected value | Effort |
| --- | --- | --- | --- |
| A17 client storm | **Misdiagnosis. Close.** | none (it was never a client) | done today |
| L2 env one-liners | Valid, small | 0-3% aggregate, mostly null-closure | one restart window + bench cells |
| M9 chunked GDN | Fused multi-token path: present + enabled. True chunked scan: **absent** (see amendment below) | unknown until Step-0 | Step-0 gate (~30 min); if it passes, 1-2 wk chunked-scan kernel |

---

## A17: the "empty-messages client storm" is a server-internal template probe

### What F27 claimed

F27 (2026-08-22): a client was POSTing empty chat bodies at ~1000/s; task
counter 107,732 in 18 min; 22 requests failed with "grammar: exceeds sane
defaults"; the 50 MB log rotated every ~4 min under the churn. The item was
ranked "high (bench validity), trivial once found".

### What the code actually does

`common/chat-diff-analyzer.cpp` (upstream file, present in the running build)
contains `autoparser::detect_user_start_marker()` at line 359. It probes the
model's chat template with a **baseline of zero messages**:

```cpp
params.messages              = json::array({});   // empty baseline
...
auto comparison = compare_variants(tmpl, params, [&](template_params & p) {
    p.messages = json::array({ user_msg });
});
if (!comparison) {
    LOG_DBG("%s: Template application failed, unsupported empty messages? trying complex variant\n");
    // falls back to a 2-message vs 3-message probe, which succeeds
```

The Qwen3.8 template contains a guard:
`{{- raise_exception('No messages provided.') }}` when the message list is
empty. So on this model family the empty-baseline probe **always throws**, the
exception text is logged at D level (a 4-line block ending in
"Error: Jinja Exception: No messages provided."), and the code immediately
recovers with the complex variant. Every "No messages provided" event in the
log is one template analysis, not one client request.

### Evidence

1. **Timing shape.** In the current log file, events cluster at
   t=24.08-27.09 and t=120.38-121.16 of the ~2 h run that opened the file
   (wall clock ~10:47-10:50 and ~12:23-12:24), each pair of lines ~0.1 ms
   apart (the exception block + the short error line of one probe). Each
   cluster event sits ~50-100 ms after a slot launch / new-request template
   analysis (e.g. 24.08.726.479 slot release -> 24.08.778.352 "using
   differential autoparser" -> 24.08.784.811 probe failure). This is the
   per-slot template-analysis cadence, not a retry loop.
2. **Current run: zero events.** The run started 12:53 (UD-Q6_K_XL, n3,
   verified from /proc and the boot section of the log) shows 0 occurrences
   across its full lifetime as of writing.
3. **Live client capture.** `ss -tnp` sampled once per second for 10-15 min
   on :1234: the only non-listener peer was `pi` (pid 2807628). No mystery
   client.
4. **F27's task-counter figure (107,732 in 18 min) does not reproduce in
   today's logs** and is now treated as a separate, unresolved artifact. If
   it recurs, re-measure with the method below before attributing it to any
   client.

### Cost assessment

Each probe costs ~6.5 s of template analysis at slot launch (measured
24.08.778.352 -> 24.08.784.811 for the Phase 1-3c sequence), which is a
one-time per-template-analysis cost, and the empty-baseline failure adds two
D-level lines. Neither is a throughput or bench-validity problem.

### Action

- Close A17 as misdiagnosed. The log signature was a false friend.
- Optional upstream nit (do not patch the pinned tree for it): skip the
  empty-baseline probe when the template's capability set indicates an
  empty-messages guard, to cut D-level noise.
- **The real residual finding is a data-quality defect** (below).

### Data-quality defect found while doing this (fix before next bench)

`llama-server.log` is a **shared append target across processes**: the file
contains at least the 2 h server run, hundreds of short-lived benchmark
processes (each logging its own model loads from t=0.00), and the current
server run, with no truncation between them. Rotation backups are stale
(`.1` = Aug 9). Timestamps are per-process elapsed time (M.S.ms.us since
process start, `common/log.cpp:98-105`), so lines from different processes
interleave non-monotonically. Any log forensics done on this file (including
today's, and F27's) is contaminated.

Fix: give benchmark processes their own log files (they already stream to
`output/benchmarks/<stamp>/`; stop them from also writing the server log), or
add a run-id line at process start. Until then, treat the shared log as
unusable for rate/timing claims.

Corrections to the record:

- F27's "client POSTing empty messages at ~1000/s": retracted as stated.
- My "storm still live, last event 13:59" claim (made earlier today): wrong;
  13:59 was the file's tail timestamp, not an event timestamp. Retried with
  proper timestamp mapping before writing this section.

---

## L2: restart-window one-liners (Vulkan env vars + `-tb`)

Null-table gap check (what is genuinely untested here, from
`AMD_DUAL_GPU_TUNING.md`): `GGML_VK_DISABLE_FUSION`,
`GGML_VK_DISABLE_MULTI_ADD`, `GGML_VK_DISABLE_COOPMAT` (the KHR family),
`GGML_VK_ALLOW_GRAPHICS_QUEUE`, and `-tb` (the old sweep covered `-t` 1..32
only). Already tested and null/negative: `GGML_CUDA_DISABLE_FUSION`
(ROCm, -5.3% when disabled, i.e. fusion ON is good), `GGML_VK_DISABLE_COOPMAT2`
and `_DECODE_VECTOR` (NVIDIA extensions, RDNA does not have them, no-ops).

All tests: one variable per cell, baseline re-measured in the same session,
xtx-guard up, peak XTX beside every number, prompt file named.

### 1. `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` (the one with a real prior)

Mechanism (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:5926-5940`): llama.cpp picks
its compute queue family **excluding** the graphics bit by default, plus a
separate transfer family. With the env set, it takes the first
compute-capable family, which on RADV is the combined gfx+compute family.
Consequences per card:

- XTX (drives the desktop): default keeps llama.cpp off the compositor's
  queue; the env moves compute onto the shared gfx queue. VRAM is unaffected;
  this is a scheduling-contention change, watch display smoothness during the
  test.
- R9700 (no display): both queues idle; the env only changes which hardware
  pipe/scheduler the work lands on.

Evidence (AMD only): #21043 (R9700 experiments) OP measured zero on RADV;
JohnTDI (comment 18) reports a small positive on RADV in an MTP config.
Two data points, opposite directions, same GPU class.

Prior: |delta| < 2%. This is as much a **jitter experiment** as a mean-speed
experiment: the interesting output is tg variance (the 3 s windows), not just
the mean.

Test: next normal restart, env on, one session of real use. Compare against
the F24 baseline (31.56 t/s @49k, mean acceptance length 2.37, per-position
(0.641, 0.420, 0.277)). Record tg mean + 3 s-window sd + acceptance.
Falsify: mean within 1% and variance comparable -> close as null.

### 2. `GGML_VK_DISABLE_FUSION` (expected to confirm a negative)

Mechanism (`ggml-vulkan.cpp:6484`): `disable_fusion` gates
`add_rms_fusion` (6486), the multi-add chain fusion, `MUL_MAT_ADD_ADD`
(matmul + two residual adds in one dispatch, 16249-16258), rope/set-rows
fusion, and topk-moe fusion (the last is inert on dense 27B). On this model,
`MUL_MAT_ADD_ADD` covers the output-projection matmul + the two residual
adds per layer: fusing removes two VRAM round-trips and two dispatches per
layer per op. Decode batches (1-13 rows) and prefill ubatches (1024) both
benefit from the fusion on paper.

Evidence: the ROCm-side analog `GGML_CUDA_DISABLE_FUSION` measured **-5.3%
when disabled** (null table, i.e. fusion ON is worth 5.3% on HIP). The
Vulkan fusion is different code, but the op structure is the same.

Prior: disabling -> -3..-6% (harmful). The test is a null-table closure, not
a hunt. It would only be interesting if it comes out positive, which would
point at a mis-tuned fused pipeline on RADV or an interaction with the
meta-backend's per-GPU buffer mirroring under tensor split (untested
combination: every prior fusion measurement pre-dates the tensor profile).

Test: `GGML_VK_DISABLE_FUSION=1`, pp8192 at d0 and d65k, -r 5, tensor 50/50
profile. Falsify: within 1.5% at both depths -> null, close.

### 3. `GGML_VK_DISABLE_MULTI_ADD` (subset, test conditionally)

Mechanism (`ggml-vulkan.cpp:6104-6106`): `device->multi_add` requires
`shaderRoundingModeRTEFloat16` plus push-constant headroom; it fuses chains
of consecutive ADD ops (residual chains) into one dispatch. It is a strict
subset of what DISABLE_FUSION disables.

Ordering: test after DISABLE_FUSION. Only isolate multi-add if the full
fusion test comes out negative and we want to know which fusion pays.
No separate prior; no separate test unless triggered.

### 4. `GGML_VK_DISABLE_COOPMAT` (valid in this build, pp-only lever)

Build verification (done today): the tp-adaptive build uses `/bin/glslc`
(Vulkan 1.4.341 tools, `CMakeCache.txt`). The build's own feature test
(`vulkan-shaders/feature-tests/coopmat.comp`, compiled exactly as
`test_shader_extension_support` does) **succeeds on this glslc** (exit 0), so
`GGML_VULKAN_COOPMAT_GLSLC_SUPPORT` is ON and the env var is a live test
(unlike the NV coopmat2 extensions, which RDNA lacks entirely).

Mechanism: the flag gates `device->coopmat_support` (6537), which feeds:

- `mul_mat_l` (large-batch matmul pipeline, 6377: `= coopmat_support &&
  driver_id != AmdProprietary`, true on RADV), and
- the flash-attention pipeline selection `FA_COOPMAT1` (3415, 4073, 6126:
  `coopmat1_fa_support = coopmat_support && subgroup_require_full_support`).

For this model specifically (UD-Q6_K_XL = K-quant weights, f16 KV cache):
the weight matmuls run K-quant vec/mmvq pipelines regardless, so the
coopmat-sensitive surface is **flash-attention prefill (f16 Q/K/V) and the
f16xf16 ops**, not decode. Decode batches of 1-13 rows route to the
small-batch paths; tg should be ~unchanged.

Evidence: none found for RDNA on the FA-coopmat-vs-scalar choice. Community
data is NVIDIA-centric. Honest prior: pp +/-1-3% (either direction), tg ~0.
If the WMMA/FA path is well-tuned on RADV this is a small negative; if it is
not, a small positive.

Test: `GGML_VK_DISABLE_COOPMAT=1`, pp8192 at d0 and d65k, -r 5, tensor
profile; one tg sample to confirm the ~0 expectation. Falsify: within 1.5%
at both depths -> null, close.

### 5. `-tb` {16, 24, 32} (null closer)

Current run: `n_threads = 16 (n_threads_batch = 32) / 32` (boot log). The
old null sweep covered `-t` 1..32 but never `-tb`. Mechanism: batch threads
drive the CPU side of prompt processing; with `-ngl 99` nearly all ops are
on GPU, and the tensor allreduce staging runs on its dedicated proxy thread,
not the batch pool. On a 16C/32t part, 32 batch threads is 2 per core.

Prior: null (the CPU work is scheduling-dominated and small). Test:
pp8192 d0, -r 5, three cells. Falsify: all within 2% -> null, close.

### Batch plan

One free-GPU window: re-measure the baseline pp/tg in-session first, then
run the four env cells + the `-tb` cells (env vars each need their own
server start; `-tb` cells can share one start only if done as separate
bench invocations against a restarted server per value, since it is a
process arg). Total: ~2-3 h. Expected outcome: 2-4 closed nulls, at most one
small real win (ALLOW_GRAPHICS_QUEUE or DISABLE_COOPMAT).

---

## M9: chunked GDN prefill (the one structural prefill lever)

### What it is

Qwen3.8-27B: 65 layers, `full_attention_interval = 4`, so **48 of 65 layers
are Gated DeltaNet** (recurrent, linear attention) and 17 are full
attention. The GDN layers run the fused `GGML_OP_GATED_DELTA_NET` op, which
in every local build executes as a **token-sequential recurrence**.

### Code facts (verified in llama.cpp-tp-adaptive, the running build)

- Dispatch: `ggml_vk_gated_delta_net` (`ggml-vulkan.cpp:11909-11978`)
  dispatches workgroups `{ H, n_seqs, S_v }` with S_v = 128 (the d128
  pipeline family). The token dimension (`n_tokens`, a push constant) is
  **not** in the grid.
- Shader: `vulkan-shaders/gated_delta_net.comp`, `main()` at line 95:
  `head_id = grid.x`, `seq_id = grid.y`, `col = grid.z * COLS_PER_WG +
  thread-offset`; then `for (uint t = 0; t < n_tokens; t++) { ... }` (line
  122) with the recurrent state update inside. Each workgroup carries a
  shard of the per-head S_v x S_v state matrix (ROWS_PER_LANE = 4 rows per
  lane over 128 threads) and walks the whole ubatch token-by-token.
  Parallelism = heads x seqs x state-column shards. The sequential chain is
  n_tokens long per workgroup.
- The op is built by `src/models/delta-net-base.cpp` (called from the
  qwen35 model code); the Vulkan pipeline exists; the current run's boot
  section shows no CPU-fallback warning, so GDN runs on GPU in production.
- **The shader is byte-identical between tp-adaptive and tp-trial** (a
  rebase onto master b21e4de74, 2026-08-22): upstream has not changed this
  kernel in that window. We already run the newest upstream Vulkan GDN
  kernel as of 08-22.

### Why it is slow (the theory)

The op is **latency-bound, not compute-bound**. Two independent data points
agree on the mechanism (both non-AMD hardware, used for mechanism only):

1. khalidejaz's independent chunked implementation (gist, 2026-08-11, RTX
   5060 Ti): the isolated GDN op runs at ~**1% of the GPU's tensor peak**;
   the serial state scan is ~**78% of the op** and stalls on global loads
   and barriers, hidden only by block occupancy. Chunk size decides
   occupancy (16-token chunks fit ~3 blocks/SM; 64-token chunks fit 1),
   which alone was worth 6-17%.
2. PR #24561 (chunked MFMA GDN prefill, MI250X): the per-token kernel
   sustains 18-30 GB/s effective and degrades with length; the chunked
   kernel holds ~46 GB/s. Isolated speedup 1.63x (512 tok) to 2.44x
   (4096 tok).

The chunked algorithm (FLA line of work: Yang et al., gated delta rule;
WY-style per-chunk factors) does not remove the recurrence; it replaces a
long per-token latency chain (n_tokens dependent steps) with a short
per-chunk serial scan (n_tokens/C steps, each a parallel matmul) plus a
parallel over-chunks precompute. The win is dependent-step count and
occupancy, not FLOPs.

### A theory detail the prior docs missed: depth dependence

GDN cost is O(1) in context depth (the state is a fixed S_v x S_v per head;
nothing grows with the prompt). Flash-attention cost grows with depth.
Therefore **the GDN fraction of prefill time is largest at d0 (cold starts,
new files, first request after a checkpoint replay) and shrinks with
depth**. A chunked port helps short-to-medium prompts most, and the
d65k cells will show less than the d0 cells. Any test plan that only
measures at depth underestimates the win; any plan that only measures at d0
overstates the IRL average.

### Value validation (the adversarial part)

- The 08-23 doc's "+22% prefill" figure came from a **pinned PR set**
  (#26001 + #25635 + #26705) tested as a bundle. It is not an isolated GDN
  number. Retracted as an expectation.
- #24561's own end-to-end table (MI250X, Qwen3.6-27B): at the realistic
  default ubatch 512, **+2.4% to +6.2%** E2E; at ubatch 8192, +2.3% to
  +23.7% (the big number needs the kernel to see the full prompt at once).
  It also quotes #22587: "delta-net is ~2-3% of E2E at default ubatch, hard
  to measure".
- #24561 is **CDNA-only**: `__builtin_amdgcn_mfma_f32_16x16x16f16` intrinsics
  behind `amd_mfma_available()`, gated to gfx90a. It does not apply to
  gfx1100/gfx1201. It is an algorithm and structure reference, not a
  cherry-pick.
- Issue #20354 (2026-03, Strix Halo gfx1151): on that build Vulkan had **no**
  GDN shader at all (CPU fallback) and the HIP AR kernel ran at CPU parity
  (12 t/s). That pathology is not our case (we have the shader, verified
  above), but it is the warning that AR GDN kernels on RDNA can be far from
  bandwidth-bound, i.e. the RDNA AR number could be worse than the CUDA one
  relative to peak.
- **Fresh upstream signal**: a build newer than master b21e4de74 (our
  08-22 baseline) logs `resolve_fused_ops: fused Gated Delta Net (chunked)
  enabled` (observed in the shared log from a process built outside the
  local trees; neither tp-adaptive nor tp-trial contains the resolver
  string). A chunked fused GDN path has landed in upstream master very
  recently. Identify the commit before planning the port.
- Port topology note: a GDN kernel is a shader + ggml graph change. It does
  **not** touch the meta/tensor backend, so the rebase death that killed A1
  ("rebase onto new master breaks -sm tensor") does not apply to a GDN port
  into tp-adaptive. This is materially easier than A1 was.

Honest expectation for this box: **+2-6% E2E prefill at ubatch 1024,
d0-favored**, possibly worse than that if the RADV AR kernel is closer to
the #20354 pathology, possibly better if the upstream chunked path is a
near-clean port. Not a 20%+ lever.

### Step 0: the cheap gate (do this before any kernel work)

Isolated op benchmark on this box, no production impact, ~30 min in a
free-GPU window:

```
# tp-adaptive Vulkan build, model H from the gguf (Qwen3.8-27B GDN heads), d=128
./build/bin/test-backend-ops -b Vulkan -op GATED_DELTA_NET \
  -d H:48,d:128,n_tokens:1024  -r 5    # repeat at 512/2048/4096
```

(adjust the op-arg syntax to the build's test-backend-ops; the point is the
isolated per-token time of the AR kernel and its effective GB/s, per the

# 24561 table format). Then compute the GDN fraction of prefill

    GDN_fraction ~ 48_layers x (us per token, isolated) x n_tokens
                   / (measured pp time for the same n_tokens)

using measured pp8192 d0 (~8192 / 385 s on UD-Q6 50/50, 08-24 record) as the
denominator.

Gate: GDN fraction >= 5% -> proceed to port. < 2% -> drop M9 (the #24561
E2E math caps the win at ~1.2%). 2-5% -> only after the L2 one-liners and
any other queued items are exhausted.

### Port options, in order

1. **Port the fresh upstream chunked fused path** (first identify the
   commit; it post-dates b21e4de74). A shader + resolver change, no meta
   backend involvement. Lowest risk, and it keeps us in step with upstream.
2. **Vulkan GLSL port of the #24561/gist algorithm** if (1) stalls:
   small chunks for occupancy, state in registers/LDS, and pipeline the
   parallel precompute ahead of the serial scan. The gist's two production
   bugs are direct warnings for a tensor build: scratch must live on the op's
   own device (the 2-GPU offload bug), and the parallel precompute needs an
   explicit dependency on the producing stream (the NaN-logits race). Under
   tensor split the equivalent is per-device scratch + semaphore ordering
   with the allreduce path; get that right or the failure mode is silent
   corruption several kernels later.
3. **HIP WMMA port into the pinned ROCm build**: only benefits the ROCm
   profiles, which are no longer the daily driver. Deprioritized.

Quality gate before any ship: `test-backend-ops` GATED_DELTA_NET cases
including partial chunks (17, 65 tokens) and multi-seq; PPL delta vs the AR
kernel on the fixed prompt set. Reference tolerances: #24561 reports max abs
diff ~1.2e-4 (f32 round-off) vs the per-token kernel.

---

## Other methods found online (triaged, not promoted)

- **#27193** "vulkan: fuse GATED_DELTA_NET state write into the op" (open):
  saves the state round-trip out of the dst tail. Small; pair it with a GDN
  port if one happens, not standalone.
- **#22967** (RFC: chunked CUDA prefill GDN): the tracking issue; leonardHONG
  owns the CUDA side. Watch, do not duplicate.
- **NVlabs/GatedDeltaNet + flash-linear-attention**: the algorithm
  references (gated delta rule, parallel scan, WY factors). Citations for the
  port, not levers.
- **Vulkan GDN shader draft PR (AICrier writeup / r/LocalLLaMA)**: merged
  upstream; we already carry the newest shader (byte-identical to the 08-22
  master baseline). No action.
- Re-confirmed dead-ends (kept out of the queue per the standing rules):
  RCCL (segfault), D2D allreduce (-90% measured), P2P patches (no P2P on
  this box), CUBLAS compute envs (K-quant paths untouched), ngram-mod
  re-test.

## Priority order

1. Log-hygiene fix (separate bench processes from the server log): before
   the next benchmarking session, otherwise every future measurement
   inherits the forensics contamination documented in the A17 section.
2. L2 one-liner window: ALLOW_GRAPHICS_QUEUE (session A/B) + DISABLE_COOPMAT,
   DISABLE_FUSION, `-tb` (bench cells). ~2-3 h, mostly null-closure.
3. M9 Step-0 isolated GDN benchmark + fraction gate. ~30 min. Proceed to the
   upstream chunked-path port only if the gate passes and the upstream commit
   is identified.
4. A17: closed.

## Standing constraints honored in this doc

- temp 1.0 / top-p 0.95 / top-k 20 is the production sampler and is not a
  tuning target (vendor-recommended); no test plan below touches it.
- f16 KV and f16 draft KV remain the standing defaults; nothing here
  revisits KV precision.
- XTX ceiling 24,200 MiB; xtx-guard up for every bench cell; peak XTX
  recorded beside every speed.
- CUDA/NVIDIA numbers are algorithm references only; no expected value in
  this doc derives from non-AMD hardware.

---

# Verification pass (2026-08-24, after the doc was written)

Checked against the running build and logs. Most of the doc holds; one claim does
not, and it changes the M9 conclusion.

## Confirmed

- **A17 is a misdiagnosis.** `detect_user_start_marker` is at
  `chat-diff-analyzer.cpp:359` as stated, and the Qwen3.8 template really does
  contain `raise_exception('No messages provided.')` (one of 11 guards in it).
  The shared log holds **34** such events, not 107,732 requests, and the current
  run's own log has **zero**. Closing A17 is correct.
- **Code references accurate** to within a line or two: ALLOW_GRAPHICS_QUEUE 5926
  (exact), disable_fusion 6484 (exact), multi_add 6104 (exact),
  ggml_vk_gated_delta_net 11907 (doc said 11909), shader main() 95 (exact), token
  loop 121 (doc said 122).
- **Shader identity confirmed by SHA-256**: `gated_delta_net.comp` is byte-identical
  between tp-adaptive and tp-trial (2f4ccdd5a862e79d...).

## Refuted: M9 as scoped

The doc states the chunked path is absent locally ("neither tp-adaptive nor
tp-trial contains the resolver string") and scopes a 1-2 week port.

**Chunked GDN is already present AND enabled in the production build.**
`llama.cpp-tp-adaptive/src/llama-context.cpp:557-596` implements the chunked
capability check inline, with `fused_gdn_ar/ch = true` and `auto_fgdn = true` as
defaults (lines 200-202). Verified at runtime under the production tensor-split
config with `-lv 5`:

    sched_reserve: resolving fused Gated Delta Net support:
    sched_reserve: fused Gated Delta Net (autoregressive) enabled
    sched_reserve: fused Gated Delta Net (chunked) enabled

Both paths resolve, no device mismatch, no self-disable under `-sm tensor`. The
lines are absent from normal logs only because they sit above default verbosity 3.

The doc's grep searched for master's `resolve_fused_ops` symbol; this fork
predates that refactor and implements the same logic inline, so the search
returned a false negative.

**What survives of M9:** the Vulkan shader still walks tokens sequentially INSIDE
a workgroup (`gated_delta_net.comp:121`) even on the chunked path, so occupancy
and chunk-size tuning of that shader remains an open question. That is a shader
optimisation, not a missing feature, and is much smaller than the doc's framing.
The Step-0 isolated benchmark is still the right way to size it.

## Minor

- GDN layer split is **49 GDN / 16 attention**, not 48/17: full attention at
  `i % 4 == 3` (blk.3, 7, 11, ..., 63), verified from the GGUF tensors
  themselves (`blk.{i}.ssm_conv1d` present iff `i % 4 != 3`), not just the
  `full_attention_interval` scalar.

## Revised priority

1. Log-hygiene fix — still the top item. Both F27 and this doc drew wrong
   conclusions from the shared log.
2. L2 env one-liners — unchanged, valid, ~2-3 h of mostly null-closure.
3. M9 — Step-0 gate decides whether the real chunked-scan kernel (1-2 wk)
   is worth it; see amendment below.
4. A17 — closed.

## M9 amendment (2026-08-24, second pass): "chunked" is a fusion flag, not a scan algorithm

The refutation above is right about the narrow claim and wrong about the
framing. Code evidence, all in llama.cpp-tp-adaptive (the running build):

1. **One op, one name switch.** `src/models/delta-net-base.cpp:567-572`:
   `build_recurrent_attn` builds a **single** `ggml_gated_delta_net(q,k,v,g,b,s,K)`
   over the whole ubatch; `FGDN_CH` vs `FGDN_AR` is only the tensor *name*
   (`n_seq_tokens > 1` vs `== 1`), consumed solely by the capability check.
2. **The "chunked" check is device placement, not an algorithm.**
   `llama-context.cpp:519-598`, inside `sched_reserve()` (line 439 — matches
   the `-lv 5` runtime prefix): reserves a `16*n_seqs`-token graph and
   verifies each GDN node lands on the same device as its layer.
   "fused Gated Delta Net (chunked) enabled" therefore means *multi-token GDN
   ops are fused and GPU-resident*. It says nothing about how tokens are
   processed inside the kernel.
3. **The kernel has no token parallelism.** `gated_delta_net.comp:121`:
   `for (uint t = 0; t < n_tokens; t++)` is a dependent loop over *all* tokens
   of the op; the push constants carry no chunk parameter. `ggml_vk_gated_delta_net`
   (`ggml-vulkan.cpp:11909-11978`) dispatches one launch with grid
   `{H, n_seqs, S_v}` — no token/chunk dimension. There is exactly one GDN
   shader in the Vulkan backend.

So the FLA-style chunked scan (parallel within-chunk, serial between chunks,
WY representation — the algorithm in CUDA #24561 and in vLLM/SGLang GDN
prefill kernels) is **not implemented** in this build. The prefill critical
path per GDN layer is still a dependent scan over the full ubatch: up to
2048 sequential state-update iterations in the production ub/batch regime.

Consequences:

- The verification pass's "descoped to shader occupancy tuning" is the wrong
  frame: the kernel already runs H x S_v workgroups per sequence per layer
  (thousands — occupancy is not the limit), and no workgroup-count or loop
  tuning can shorten a 2048-step dependent chain. The only structural lever
  is fewer sequential steps, i.e. the real chunked algorithm.
- What **was** refuted: "the fused multi-token path is absent and needs a
  port." It is present and enabled (the original doc's grep was a false
  negative: it covered `ggml/src` only, while the check lives in
  `src/llama-context.cpp`; the binary strings live in `libllama.so`, not the
  18 KB launcher).
- The remaining work is a *new* kernel + graph change: chunk the ubatch in
  the graph, compute within-chunk contributions in parallel, serial scan
  between chunks. Still 1-2 weeks of kernel work — correctly named.
- The Step-0 gate is unchanged and is now the only decision that matters:
  isolated GDN fraction of pp time < ~5% -> drop the whole item (occupancy
  tuning included); >= ~5% -> the chunked scan is the single architectural
  pp lever on this box.
