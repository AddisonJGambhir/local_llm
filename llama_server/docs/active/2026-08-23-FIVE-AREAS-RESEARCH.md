# Research session 2026-08-23: the five areas (code + data, no GPU work done)

Research only, per instruction. No servers were started or touched (I am the
llama-server). Everything below is either (a) read from local code/logs/records
or (b) an exact measurement plan for the next free-GPU window. Numbers quoted
from prior sessions carry their original rep counts; nothing here is a new
measurement.

---

## Area 3 first, because it rewrites the problem: the 2.9x prefill gap is a

cross-backend comparison, and the shipped build mostly resolves it

The "unresolved tension" is stated as: tensor wins decode (81.67 tg),
Vulkan layer/row wins prefill (980 vs 337.7 t/s on a 71,449-token prompt, 2.9x).
But the 980 is VULKAN layer and the 337.7 is **ROCm** tensor. The build that
holds the decode win (llama.cpp-tp-adaptive, PR #25051) has its own measured
tensor prefill numbers in MASTER_RESULTS.csv (2026-08-21, llama-bench, same
binary, d0 pp8192):

| quant | layer 28/37 | row | tensor 50/50 | tensor/layer ratio |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 d0 | 1232.00 | 1240.47 | 940.78 | **1.31x** |
| Q6_K d0 | 1091.76 | 1082.92 | 885.56 | **1.23x** |
| Q8_0 d65536 | 468.75 | 487.41 | 574.23 | tensor **wins 1.22x** |
| Q6_K d65536 | 461.97 | 465.72 | 549.69 | tensor **wins 1.19x** |

So with the build actually shipped for decode, tensor prefill is within ~1.3x of
layer at empty KV and **wins by ~20% at depth** (this matches the tuning doc's
decay table: tensor decays -47% d0->d98k, layer -70%). The 2.9x only exists
between ROCm tensor (337.7) and Vulkan layer (980). Nobody needs the ROCm
tensor profile for prefill; it never won prefill.

Consequences:

1. **A hybrid single server (prefill mode A, decode mode B) is not possible
   anyway**: split mode is a context parameter fixed at creation
   (`llama_context::set_sampler`-era cparams; the meta device graph is built per
   context). There is no per-batch or per-phase split-mode switch in any tree
   here, and I found no open upstream PR adding one. Master's multi-gpu.md
   still documents exactly one mode per context.
2. **Two concurrent servers do not fit the XTX.** The adaptive tensor
   preset peaks 23,521 MiB at 256k (MASTER 2026-08-22); the layer server peaked
   17,575-17,866 MiB at 98k (MASTER 2026-08-22). Sum > 24,200. A second
   *small* layer server (c ~49k, ~14k MiB) plus the tensor server at 98k
   (20.4k MiB) also exceeds the measured cliff line once the desktop baseline
   is included. Dual-instance routing is not practical here; 24 GB on the XTX
   is otherwise a fine working size (see the correction note below Area 3).
   [correction applied 2026-08-23 per user: the XTX headroom is NOT a design
   constraint; only the measured ~24,180 MiB GTT cliff is a hard fact, and the
   box runs fine up to it]
3. The residual same-build gap to attack is **1.3x tensor-vs-layer prefill at
   d0**, and it only matters for cold long-prompt ingestion. Concrete
   untested levers, in order:
   - **ubatch 1024 -> 2048 at c=98304** (tensor, the 98k profile). Every
     prefill ub sweep on record was at 192k/245k/256k (where 1024 or 512 won on
     VRAM) or at 128k d65k (where 512 won at 128k ctx). At 98k the VRAM budget
     exists (20,388 MiB peak + ~1,171 MiB per the ub2048 delta measured at
     256k -> ~21.6k, under ceiling). Fewer ubatches = fewer allreduce
     barrier chains per prompt. One llama-bench A/B: `-ub 1024` vs `-ub 2048`,
     `-p 8192 -n 256 -r 3`, c=98304, tensor 50/50. See command block D below.
   - The allreduce path itself (ring, CPU proxy, single `proxy_thread`
     draining both bridge queues: `ggml-vulkan.cpp:17874-17890`). Prefill at
     ub 1024 over a 71k prompt issues thousands of allreduces, each with
     several cross-device timeline hops submitted by ONE CPU thread. This is
     the most plausible mechanistic owner of the 1.3x, and it is upstream-level
     work (JohannesGaessler's meta-backend refactor is the real fix, per the
     PR #25051 thread). Not something to patch today; flagged for the PR
     comment / MR.
   - F13 already corrected the link figure: the R9700 end-to-end link is
     **15.75 GB/s (Gen4 x4)**, not the 7.9 GB/s the docs assumed, so the
     interconnect ceiling for prefill is 2x higher than the earlier analysis
     used. The comm cost is latency/serialization-bound, not BW-bound.

**Bottom line for #3: the biggest measured ratio in the dataset is a
measurement artifact (ROCm tensor vs Vulkan layer). The real, same-build
residual is 1.3x at d0, inverting to a tensor win at depth. The only cheap
untested knob is ubatch at 98k.**

---

## Area 1: the depth cliff - what the code rules out, what remains

Code read (llama.cpp-tp-adaptive @13ac90156 + #27210 port, the running tree):

- Qwen3.8 is the "single trained MTP head" branch
  (`common/speculative.cpp:912-916`): own draft context, **not**
  `is_mem_shared`, **not** `chain_heads`.
- The draft context gets a **mirror decode of every target batch**
  (`process()`, `speculative.cpp:1118-1180`): prompt ubatches and verify
  batches are replayed into the draft context, so its KV holds the full
  sequence but only for **1 nextn layer** with q8_0 draft KV. At 120k depth
  that is roughly 1/65 of the target's f16 KV, i.e. ~0.3 GiB. Not a cliff.
- Draft steps are single-token decodes of that 1-layer context
  (`draft()`, `speculative.cpp:1252-1277`), seeded with the target's nextn
  hidden state. **The draft head runs on one device (not tensor-split), so
  draft steps pay no cross-GPU allreduce.** Per F14 the draft devices are
  `[default]`.
- **Rejected draft KV costs nothing**: the single-head path re-adds only the
  new token at `n_past + i + 1` each step; stale rejected cells from a longer
  previous round sit at HIGHER positions and are never read (attention reads
  only earlier cells), and the next round overwrites them in place. There is
  no per-depth write amplification of speculative KV. **Answer to "do deeper
  drafts write speculative KV that costs more once the cache is large?":
  no, per the code. The only depth-scaling draft cost is 1 layer of
  attention over the growing KV, ~1/65 of a target layer per step.**
- One structural detail the A/B will surface: Vulkan routes small-batch
  mul_mat to the vec (GEMV) pipeline only up to 4 columns
  (`ggml-vulkan.cpp:9395-9399`, `mul_mat_vec_max_cols`), else the matmul path.
  So **n3 verify (4 rows) runs vec GEMV; adaptive verify at depth >= 4 (5-13
  rows) runs the matmul/coopmat path.** The two configs' cost-per-depth curves
  are not guaranteed to scale identically - another reason the depth A/B
  cannot be extrapolated from d0 numbers.

What the existing data already says:

- This box has NO architectural tensor cliff at 50/50: raw tg d0->d65k is
  21.21 -> 19.74 (Q8, -7%, MASTER 2026-08-21) and 23.92 -> 21.79 (Q6).
- The dramatic collapse on record is **Q8 tensor 70/30 at d65k: 26.01 ->
  11.65 tg, with peak XTX 23,866 MiB** (within 350 MiB of the measured GTT
  cliff at 24,180, HOW_TO_OPTIMIZE). Same row, Q6 70/30 (peak 20,932 MiB)
  stays at 473.33 pp / 26.31 tg. **The "depth cliff" already reproduced on
  this box is the VRAM cliff, not a tensor-depth effect.** The external
  reports (9-11 t/s at 60-160k on no-P2P boxes, F14) are consistent with
  XTX-near-ceiling runs on Q8 heavy splits.
- The IRL 120k number (27 t/s) vs bench 81.67 is explained by content
  (F24: mean acceptance length 2.37 vs ~6-7) plus the raw -7% depth decay.

So the open question is exactly A16/A12: does adaptive still beat n3 at
60k/120k/200k, and does the adaptive controller lag (drafting deep into
low-acceptance content where each wasted verify row is ~10x more expensive in
compute than at d0)? Controller analysis
(`common/speculative-adaptive.h`): from floor 3, climbing to 4 needs 6
consecutive full accepts; a drop from N needs max(5N, 20) accumulated
missed tokens. At deep context with IRL per-position acceptances
(0.64/0.42/0.28, F24), full 3-accepts are rare, so the controller sits at the
floor - good. The risk is transient high-acceptance stretches (repetitive
tool output) that climb the controller, followed by a regime change where 12
deep drafts into a 13-row verify waste ~3x verify compute for 1-2 accepted
tokens. The A/B will show whether that lag is measurable.

**Decision experiment (server-side, because llama-bench cannot run the MTP
draft path):** block A below. 7+ reps per cell, per-run acceptance from the
log (the `draft acceptance = ...` print_timing line), permutation test.
llama-bench block B gives the raw (unspeculated) depth curve on the current
binary, which bounds what the MTP numbers can mean.

---

## Area 2: CPU sampling fallback - confirmed, costed, no upstream help

- `llama_context::set_sampler` (`src/llama-context.cpp:1194-1211`): the tensor
  exclusion is **unconditional and hardcoded**; it erases the backend sampler
  and returns false, independent of backend. Both target sampling and the MTP
  draft's own top_k(10) sampler fall to CPU (two separate log lines, both
  confirmed in the running server's history, F15).
- Web check (2026-08-23): discussion **#24048** ("SPLIT_MODE_TENSOR and
  backend sampling incompatible?") is still **Unanswered**; the latest reply
  (2026-06-17) points at the same hardcoded guard. A PR search of
  ggml-org/llama.cpp for meta-backend sampling work finds **nothing open and
  nothing merged**. No `ggml_backend_comm`/meta sampling implementation
  exists in any local tree. Confirms F15's verdict: a GPU sampler under
  tensor is original work, not a cherry-pick.
- Cost is already measured (F26): ~1-1.5 ms per ~75 ms round, i.e. **~1.5-2%**.
  The adaptive mode's extra draft samplings are cheap by comparison:
  top_k=10 over 151k f32 logits, one per draft token, CPU-side; even at n12
  that is well under the target's single full-chain sample. **Expected
  ceiling above 81.67 from this item alone: ~2%.** Real, but the smallest of
  the five areas.
- Cheap partial mitigation, untested: the target sampler chain under the
  IRL profile is temp 1.0/top-p 0.95/top-k 20. At temp 0 / top_k 1 the chain
  is trivial (argmax) and the CPU cost collapses. The bench harness already
  uses temp 0 + top_k 1, so the 81.67 number already pays the MINIMUM sampler
  cost; the 2% matters for the IRL temp-1.0 profile, not the bench number.

**No action today. If it is ever pursued: upstream PR (meta-backend
`backend_apply` for the sampler ops) is the only clean path.**

---

## Area 4: the "free 1.1+0.9 GiB VRAM" is already banked - the labels are wrong

This one resolved as a **false positive in the FACTS generator**:

- The four profiles tagged `tower NOT routed` (both q8/q6k rocm tensor-decode
  presets, both *-dual-f16-256k-mmproj-tensor-decode presets) each `source`
  a parent that **does** export the routing var:
  - `preset-38-27b-q6k-rocm-dual-f16-256k-mmproj.sh:176` ->
    `export MTMD_BACKEND_DEVICE=ROCm1`
  - `preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:74` -> same.
- Verified empirically today by sourcing each of the four profiles in a
  clean subshell: all four resolve to `MTMD_BACKEND_DEVICE=ROCm1`.
- The env var reaches the server: `llamactl` sources the profile in its own
  shell (`common.sh llama_load_profile`), the supervisor inherits it, and
  `supervisor.py` spawns `llama-server` with the inherited environment
  (no env sanitizing anywhere in the chain).
- `tools/mtmd/clip.cpp:184-199` honors the var in BOTH trees (ROCm pin and
  tp-adaptive), creating a separate `ROCm1`/`Vulkan1` backend for the tower
  and logging `CLIP using <backend>`.
- Why the label is wrong: `src/benchmarks/standardize_headers.py:48` greps
  for `MTMD_BACKEND_DEVICE` in **the profile file's own text only**, never
  following the `source` chain. Inherited exports are invisible to it.

So the ~1.1 GiB permanent (mmproj-BF16.gguf is 931 MiB on disk; the profile
comments put tower VRAM ~889 MiB + context ~0.2 GiB) plus the ~0.9 GiB
transient is already on the R9700, not the XTX, in all four presets.
**There is no remaining VRAM win to claim here.** Residual work:

1. Fix the generator to follow the source chain (one small change, see the
   note in the sweep section; not applied, see below).
2. Confirm on the next normal restart that the log contains
   `CLIP using ROCm1 backend` (or Vulkan1) for the active profile - the
   current log has rotated past model-load time (F27 churn), so it cannot be
   confirmed from disk right now.
3. Re-run `standardize_headers.py` after the fix so the FACTS blocks stop
   lying.

**Reported plainly: the item as stated ("fixing MTMD_BACKEND_DEVICE frees
~1.1 GiB") does not reproduce as a gap - the fix is already in place via
inheritance; only the labels are stale.**

---

## Area 5: rm_kq - located, and it does NOT touch the current model

- The knob is `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4712-4724`:
  `rm_kq = 2` by default, `4` only on the `AMD_GCN` branch. Our two GPUs do
  NOT hit that branch: detection (`get_device_architecture`,
  `ggml-vulkan.cpp:330-370`) classifies both gfx1100 and gfx1201 as
  `AMD_RDNA3` (64/32 subgroup range + 4x8 mixed-signedness int-dot-product,
  which RDNA4 also reports) - so **both cards run `rm_kq = 2` today**.
- The discussion's one-liner (`rm_kq = 1`) therefore changes both GPUs'
  K-quant GEMV pipelines (Q2_K..Q6_K).
- **But the current build is Q8_0.** The Q8_0 GEMV pipelines
  (`mul_mat_vec_q8_0_f32_f32` etc., lines 4762/4788/4840) are built with
  `rm_stdq`, which is already 1 on these GPUs. **rm_kq=1 cannot change a
  single pipeline in the running Q8_0 profile.** The +0.8-1.1% claim
  (#21043, RDNA4-only evidence) applies only to the Q6_K/Q6_K_P tensor
  presets (menu items 7-9) and possibly Q6 layer profiles.
- It is still a legitimate cheap A/B for the Q6_K presets: one line, one
  rebuild, Q6_K tensor 70/30 decode matrix (`-p 0 -n 512 -r 3` at d0 and
  d65536), plus a PPL check since it changes kernel shapes. Expectations:
  modest on the R9700 (RDNA4, the evidence base), unknown on the XTX
  (RDNA3, no evidence at all).

---

## The unrun harness, audited (src/benchmarks/preset_sweep.sh)

Findings, worst first:

1. **`kills` kills every `llama-server` by name - including the production
   server (i.e. me).** Running the sweep as-is stops the live service. It
   must target only the PID it spawned.
2. **No ceiling protection for the server it launches.** The 256k presets
   (49/51 at 23,521 MiB steady, `45/55` at 22,820) run within ~0.7 GiB of
   the 24,200 ceiling BEFORE the desktop's 1.7-3.0 GiB is even considered.
   `bin/xtx-guard` would not help: it kills only `llama-bench`, never
   `llama-server`. A sweep run of the 256k presets on a busy desktop is a
   real risk of GTT collapse or compositor eviction. The sweep needs its own
   per-run ceiling monitor that kills only its own server PID.
3. The header comment says "2 reps, median" but the code keeps the **max** of
   2 reps (upward bias), and 2 reps cannot rank MTP configs under the method
   rules (>= 7 reps + permutation test). Fine for menu labels, not fine for
   speculator A/B - the label column should say so.
4. Server output goes to `/dev/null` - the per-run draft-acceptance counters
   (required by the method rules for any MTP comparison, and the only
   validation channel since output hashes are not reproducible on this box)
   are lost. It also loses load-failure diagnostics.
5. The prompt is 24,000 chars of one repeated sentence (~4.9k tokens). That
   is exactly the repetitive-content regime F24 identified as inflating
   MTP tg ~2.5x. Uniform across presets (good for a menu), but any tg label
   derived from it is a repetitive-regime upper bound. Tag it, or swap in a
   diverse prompt (A18).
6. 1 Hz VRAM sampling: safe for the cliff (which needs ~8 s sustained), but
   peak values under-sampled; the CSV should carry the sampling caveat.
7. Minor: server launched with bare `eval "$CMD" &` (no setsid); a dead
   terminal reaps it. And the CSV's `sp` column correctly catches spec via
   the dry-run command (profiles express spec through `LLAMA_SPEC_MODE`,
   which `llama_build_command` translates to `--spec-type`), so that one
   is fine as-is.

**I did not run the sweep today** (research-only day; it also requires a free
GPU and, as written, would have killed me). It is the right first
measurement task for the free-GPU window once items 1-4 above are patched;
the patch is described in the handoff block below.

---

## Measurement plan for the next free-GPU window

Method rules applied throughout: one variable per cell, >= 7 reps with
permutation test for speculator A/Bs (4 reps with per-run acceptance for
ranking), llama-bench for raw ranking / server for absolute MTP numbers,
temp 0 + top_k 1 + ignore_eos, per-run acceptance recorded, idle box,
baseline re-measured in-session, peak XTX beside every speed, xtx-guard up
for any llama-bench sweep.

### A. Depth cliff: adaptive vs n3 at depth (THE decision experiment)

Server method, on llama.cpp-tp-adaptive (build 9834, current binary at
`llama.cpp-tp-adaptive/build/bin/llama-server`, which already carries the
F28 #24710 regex patch). Two configs differing in ONE flag:

- A: `--spec-type draft-mtp-adaptive --spec-draft-n-max 12 --spec-draft-n-min-adaptive 3`
- B: `--spec-type draft-mtp --spec-draft-n-max 3`

Identical otherwise: tensor 50/50, main-gpu 1, c=262144, f16 KV, ub 1024,
Q8_0, temp 0, top_k 1, ignore_eos. Depth created by sending a ~60k / ~120k /
~200k token prompt first (cache_prompt=true so it is not re-processed), then
the timed 2048-token generation with cache_prompt=false. 7 reps per cell per
depth; record per-run `draft acceptance` + per-position counts from the log.

Per-run acceptance is the gate (F25): if A's p4..p12 marginals are ~0 at a
depth, A and B are the same config and the comparison is void - report the
acceptance, not the t/s.

Rough wall time: 3 depths x 2 configs x 7 reps x (~35 s decode + overhead)
plus 60-200k prompt encodes. Plan for ~3-4 h; run depths in ascending order
so the machine stays at the same context allocation (c=262144 constant).

Expected outcomes and what they mean:

- A >= B within noise at all depths with comparable acceptance -> adaptive
  is depth-safe; the IRL 27 t/s at 120k is content, not depth.
- B wins at 120k/200k while A's acceptance at p4+ stays > 0 -> the
  controller lags at depth; fix = depth-aware cap (e.g. n_max 6 above 100k)
  as a server-side policy, not a build change.
- Both collapse together below ~20 t/s above ~120k -> raw decode depth
  effect (compare against block B); check peak XTX: if > 23.5k, it is the
  VRAM cliff, tag it and size down.

### B. Raw depth curve on the current binary (bounds for A)

llama-bench, no speculation (method rule), xtx-guard up:

```bash
cd ~/Desktop/local_llm/llama.cpp-tp-adaptive
export LD_LIBRARY_PATH=$PWD/build/bin
M=~/Desktop/local_llm/llama_server/models/Qwen3.8-27B-Q8_0.gguf
llama_server/bin/xtx-guard &
for d in 0 65536 131072 196608; do
  timeout 600 ./build/bin/llama-bench -m "$M" -ngl 99 -c 216064 -b 2048 -ub 1024 \
    -fa 1 -ctk f16 -ctv f16 --no-mmap --split-mode tensor --main-gpu 1 \
    --tensor-split 50,50 -p 1024 -n 256 -d $d -r 3
done
kill %1
```

This extends A12's 49k-clear result to 131k/196k on the CURRENT binary
(build 9834, which postdates the F14-era probe binary). Record peak XTX per
row; a 70/30-style collapse appearing at 50/50 would be the first evidence
of an architectural (not VRAM) cliff.

### C. Preset sweep (menu labels), after the harness patch

Patch preset_sweep.sh: (1) `kills` -> kill only `$MY_PID` (the PID captured
after `eval`); (2) redirect server stdout to a per-preset log file instead of
/dev/null and grep the last `draft acceptance =` line into a new CSV column;
(3) per-run ceiling monitor: sample the XTX at 1 Hz and `kill $MY_PID` after
4 consecutive samples > 24,000 MiB (mirrors xtx-guard's sustained-overage
rule); (4) keep the max-of-2 but add a `reps=max-of-2, repetitive-prompt`
column note; (5) launch with `setsid`. Then run it as a single task in a
free-GPU window; output to `output/benchmarks/`.

### D. Cheap single-variable A/Bs, bundle into the same window

1. **ub 1024 vs 2048 at c=98304** (tensor prefill residual, Area 3):
   `llama-bench ... -c 98304 -p 8192 -n 256 -r 3 -ub 1024` then `-ub 2048`,
   everything else identical. Watch peak XTX (expect ~21.5k).
2. **rm_kq 2->1 rebuild** (Area 5), Q6_K presets only: edit
   `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4713` (`uint32_t rm_kq = 2;` -> `1`),
   rebuild in a copy of the tree, then Q6_K tensor 70/30:
   `-p 0 -n 512 -r 3` at d0 and d65536, plus PPL on a fixed set. Do NOT mix
   this into the same binary as block A/B runs.
3. **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`** (A13b): one env var on the Vulkan TP
   profile, one session of normal use, compare IRL tg + acceptance against
   the F24 baseline (31.56 t/s @49k, mean len 2.37).

### E. Verification for Area 4 (no GPU cost)

On the next normal restart of ANY profile: grep the fresh log for
`CLIP using`. Expected: `CLIP using ROCm1 backend` (ROCm presets) or
`Vulkan1` (Vulkan presets). If it says the XTX/ROCm0/Vulkan0, the
inheritance analysis above is wrong and the 1.1 GiB question reopens - tell
me and I will re-derive it.

---

## Deep research pass (2026-08-23, online): 20 untried mechanisms

Method for this section: (1) inventoried every `getenv` in both live trees and
diffed against the ~45-knob null table, so nothing below was already tested
locally; (2) mined llama.cpp PRs/issues/discussions created since 2026-06
(vulkan, hip/rdna, ssm/gdn, flash-attn, qwen35, speculators); (3) read the
Mesa 26.1/26.2 RADV release notes line-by-line; (4) read the two most relevant
community threads found (#27290, #27154). Each item: source, expected effect,
risk/effort, test, and why it is untried here. Order = expected value.

### Decode (tg)

**M1. PR #27173 - chained MTP draft + meta output mirror + sched pool.
OPEN, unmerged, AI-disclosed. +12.8% mixed on Qwen3.8-27B Q8_0, 2x RTX 5090,
`-sm tensor` - same model, same quant, same split mode as this box.**
The PR (11 files, ~+1000/-117) changes draft mechanics, not sampling:

- `LLAMA_SPEC_CHAIN=1`: all draft tokens in ONE decode call, token picks done
  on-GPU. Before: one GPU call per draft token + a CPU round-trip after each.
  Author: +2-6 t/s.
- Sub-vocab scoring: GPU sends back only picked token + prob (not 151k
  scores); draft scores only the 32k most-common tokens; target still verifies
  against the full vocab, so output is unchanged. Author: +5-11 t/s.
- `LLAMA_META_MIRROR_OUTPUT=1`: keep a full copy of the output (lm-head)
  layer on each GPU instead of splitting it - every verify step currently
  pays GPU-to-GPU traffic to combine the halves. Costs ~0.8 GiB/GPU
  (151k x 5120 Q8_0). Author: +5-8 t/s. **This one is topology-specific to
  us**: the lm-head allreduce is the single largest per-verify transfer in
  tensor mode (151k f32 per row, x4-13 rows) and it round-trips our 15.75
  GB/s host-staging path every round.
- `LLAMA_SCHED_POOL=N`: the meta backend re-plans the work split almost every
  step because batch sizes repeat; pool the plans. Author: +5 t/s.
- Plus always-on: remember the work split per op; remove extra per-chunk sync
  events from the GPU-GPU add (the kernels already wait on each other).
- Plus a bugfix: wrong state after token rollback (short batches overwrote
  the saved states). Relevant to our long agentic sessions.
Author's numbers: control n3 142.2 t/s -> chain n5 160.4 t/s (+12.8%),
+28% on echo-style content, +1.9% prose. AI usage disclosed; per F16's rule,
review the state machine before trusting. Cherry-pick difficulty MED-HIGH:
`common/speculative.cpp` was already modified by our #27210 port (conflicts
likely, same shape as the six #27210 conflicts), `ggml-backend-meta.cpp`
+364/-67, `src/models/qwen35.cpp` +206 (model-specific chain graph).
Test: port onto llama.cpp-tp-adaptive, then at c=98304 A/B `adaptive n12`
(current) vs `chain n5` vs `chain n12`, 7+ reps each, per-run acceptance,
permutation test (block A protocol). If the port fights us, the two
always-on fixes (split memoization + sync removal) are the cheapest slice.
Untested here: nothing from this PR is in any local tree (verified by grep).

**M2. DFlash2 sidecar for Qwen3.8-27B - the checkpoint EXISTS.
`z-lab/Qwen3.8-27B-DFlash2` (mirror `incoai/...`), block-diffusion drafter,
llama.cpp support in open PR #27342.**
Author's table (Qwen3.8-27B Q4_K_M target, M5 Pro, 2048 tok, xhigh
reasoning): autoregressive 10.42 t/s; DFlash2 Q8_0 draft, block 8: 18.43
(1.77x, acceptance 5.08); BF16 draft: 19.31 (1.85x, acceptance 4.92).
Block-diffusion drafts a WHOLE BLOCK in one shot - no sequential draft steps,
no per-step latency, unlike MTP's n sequential 1-layer passes. Requirements:
port #27342 (open) and #26636 (open - "add backends of the other model to the
context", fixes the draft-model abort under `-sm tensor` we hit with DFlash
v1/DSpark); convert the HF draft to GGUF. DFlash v1 measured here lost to MTP
(26.68 vs 37.56) but that was v1 against an FP8-trained sidecar on our
quantization; DFlash2 is a different, model-matched draft.
Test: LAYER profile first (draft-model spec works there), block 5-8,
7+ reps with acceptance counters vs the same profile's adaptive MTP; then
tensor if #26636 ports cleanly. Quality gate: PPL delta vs target-only.
Untested: no DFlash2 artifacts in any local tree; checkpoint not downloaded.

**M3. f16 draft KV for MTP - one flag, untried.**
Ours ships `--spec-draft-type-k/v q8_0`. The #27290 production profile runs
`--spec-draft-type-k f16 --spec-draft-type-v f16` at 48.3% acceptance on the
same model; our IRL acceptance is 44.2% (F24). The DFlash note in
`common.sh` already documents KV precision moving acceptance 43.6 -> 58.5% on
another model. Cost: ~0.15-0.3 GiB more draft KV (1-layer q8_0->f16 at 256k),
zero speed cost expected. Test: A/B q8_0 vs f16 draft KV, same preset,
acceptance counters + tg, 7+ reps, at depth (block A depths). If acceptance
rises >2 points at any depth, this is free tg.

**M4. n_max kernel-shape cells (8, 10) on the adaptive path.**
#27290's n_max sweep on the same model is jagged: n3 43.11, n4 37.79, n7
29.10, **n8 49.31**, n9 49.02, n12 43.31 - "eight candidates hit a favorable
batch and kernel shape". We have never measured n8 or n10 as adaptive
ceilings (n8/n16 were rejected on LAYER with p_min>0; tensor n>8 was blocked
on the STOCK build's draft-mtp path, but the adaptive path demonstrably runs
n_max 12 under tensor on this build - the shipped profile). Combined with
Vulkan's vec->matmul routing change at >4 verify rows (this doc, Area 1), the
per-n cost curve is not monotone and the controller's climb/drop dynamics
interact with it. Test: adaptive with n_max 8 vs 12 vs 16 (one flag each),
c=98304 and at one deep point, 7+ reps, acceptance counters. Cheap: no
rebuild.

**M5. The three genuinely-untested Vulkan env vars.
`GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_COOPMAT` (the whole coopmat
family, not just coopmat2), `GGML_VK_DISABLE_MULTI_ADD`.**
From the getenv inventory diff: the null table covered COMM_TREE/PROXY,
GRAPH_OPTIMIZE, MMVQ, COOPMAT2(_DECODE_VECTOR), INTEGER_DOT_PRODUCT,
MAX_NODES_PER_SUBMIT, ASYNC_USE_TRANSFER_QUEUE, ALLOW_GRAPHICS_QUEUE - but
NOT these three. Fusion-off is the analog of the ROCm `GGML_CUDA_DISABLE_
FUSION` (-5.3%, tested) and could go either way on Vulkan; coopmat-off
forces the non-WMMA matmul path, which at batch 1-13 decode could in
principle beat a mis-tuned WMMA path. Each is one env var, zero risk, ~5 min
of matrix per var. Expectation: probably null (the other ~15 VK envs were),
but the null table is a promise only if the list was complete - it wasn't.

**M6. Cherry-pick #26284 (RDNA3 + RDNA4 mmq retune) into the pinned ROCm
build. OPEN PR; parent #26199 is already in the pin (verified: 60bccc376 is
an ancestor of 030ebb558).**
mmq = the small-batch matrix-matrix decode path (verify batches, n>1). The
PR's own table is measured on a **7900 XTX** (RDNA3) and RDNA4 - our exact
cards. Rebuild with the F3 one-line `-funsafe-math-optimizations` restore
(any post-08-13 ROCm rebuild needs it), A/B the decode matrix (-p 0 -n 512
-r 5 at 58/42) plus one MTP session for acceptance stability. Expectation:
low-single-digit % decode on the ROCm profiles; zero for the Vulkan
profiles (HIP-only change).

**M7. Drafter-quant alignment: rebuild only the 8 nextn tensors.
#27290's counter-intuitive result: an NVFP4 MTP head beat a Q5_K head by
+69% acceptance (48.3% vs 33.1%) against a mixed-quant target, because the
head's error pattern aligned with the target's. For our Q8_0 target the
head is also Q8_0 (max alignment already) - the experiment here is the
REVERSE direction: quantize just the nextn tensors down (Q5_K/Q4_K) and
measure acceptance, per the thesis that the drafter and target are one
quantized system. Effort: one `gguf-split`/re-quant of 8 tensors (~1 hour),
851 other tensors byte-identical (their verification method).
Expectation: genuinely unknown - could be +acceptance, could be -; one
GGUF, one A/B with acceptance counters. Cheap if it works, worthless if not;
rank it below M1-M4.

**M8. Single-device MTP draft (custom 10-line patch).**
The MTP self-spec draft context inherits the target's cparams wholesale
(`server-context.cpp` spec_mtp branch: `cparams_mtp =
common_context_params_to_llama(params_base)`; `--spec-draft-device` is only
wired to the separate-draft-model path - verified in code). So the 1-nextn-
block draft runs tensor-split and pays 2-4 tiny allreduces per draft step
and per mirror batch. Patch: force `split_mode=NONE, main device=XTX` on
`cparams_mtp`. The h-row handoff (5120 f32 = 20 KB) moves over PCIe once per
step - negligible. Expectation: <2% (the draft is 1/65 of a pass); listed
because it is a one-afternoon patch that also de-risks the F23-style
"draft under tensor" question.

### Prefill (pp)

**M9. Chunked GDN (SSM) prefill kernel - the structural prefill lever.**
Qwen3.8-27B is 3 Gated DeltaNet blocks + 1 full-attention block
(full_attn_interval=4, verified in `src/models/qwen35.cpp:22-25` and in
#27290): **48 of 65 layers are recurrent SSM**. Our trees run the token-
by-token recurrent GDN kernel for prefill - `ggml/src/ggml-cuda/
gated_delta_net.cu:181` literally says `//TODO: Add chunked kernel for even
faster pre-fill`, and the Vulkan path has the same fused-recurrent pipelines
(`gated_delta_net_f32_d16..d128`). #26001 added a 3-stage chunked GDN for
CUDA (NVIDIA Ampere tensor-core gated; vLLM/FLA-inspired) - it is NOT in our
trees and its gate excludes HIP/RDNA. In the #27290 A/B, the pinned-PR set
led by #26001 moved 45.42 -> 55.40 t/s (+22%) on this exact model - much of
that is the GDN chunking. Mechanism to pursue upstream (port to HIP WMMA /
Vulkan), or cherry-pick if it ever lands for HIP. This is the biggest
structural prefill item found; the rest of this section is smaller.

**M10. Cherry-pick #25635 (XOR-swizzle flash-attn K/V smem fp16 tiles)
into the pinned ROCm build. OPEN PR; +7.41% prefill at 32K (759.38 ->
815.64) and +2.61% hot decode on Qwen3.8-27B measured in the #27290 A/B.**
The CUDA source compiles for HIP, so the pinned ROCm build is the target.
The Vulkan TP profile is unaffected (different backend). Test: rebuild with
F3 flag, pp8192 d0 + d32k A/B on the layer profile, -r 5.

**M11. Quant drop to Q5_K_M (or Q4_K_M) + #26705 cherry-pick.
Unsloth Dynamic-v3 GGUFs for Qwen3.8-27B exist in Q4-Q8 (verified on HF).
Decode is bandwidth-bound (Q6_K measured 13% faster than Q8_0, profile
headers): Q5_K_M (~18.3 GiB vs 27.7) buys both decode bandwidth and context
headroom at a quality cost gated by PPL. #26705 (open) makes Q4_K/Q5_K
mmvq branchless - "improving perf at batch sizes > 1", i.e. exactly the
speculative verify batches. Test: download Q5_K_M, vram-probe the 58/42 and
50/50 splits, raw decode matrix -r 5 (MTP OFF per method rules), PPL vs
Q6_K/Q8_0 references, then one MTP session. This also re-opens the 256k+
context profiles at full f16 KV.

**M12. ubatch 2048 at c=98304 (tensor profile).** Already covered in the
measurement plan (block D.1) - included here so the mechanism list is
complete: fewer ubatches = fewer allreduce barrier chains per prompt;
untested at 98k because the existing ub sweeps were at 128k-256k where VRAM
forced smaller ubatch. The XTX has room at 98k (peak 20.4k; +1.2 GiB for
ub2048 is fine well below the cliff).

**M13. `-tb` (batch threads) 16/24 vs the default 32. The ~45-knob null
table swept `-t` (1..32) but never `-tb`; `llama_load_profile` defaults
`LLAMA_BATCH_THREADS` to `nproc` = 32 logical threads on a 16-core 9950X3D.
Batch threads drive prefill ubatch compute. Test: pp8192 d0 + d65k, -r 5,
tb in {16, 24, 32} on the layer and tensor profiles. One flag, zero risk.

**M14. Cherry-pick #24891 (server: fix checkpoint invalidation after tool
requests in multi-turn conversations). OPEN PR; IRL prefill mechanism.**
Our workload IS multi-turn tool conversation. If checkpoints are being
invalidated after tool requests, every tool round-trip re-encodes the whole
accumulated context - the documented failure mode was 119,389 tokens / 96 s
reprocessed ~8x in one session (base profile comment). The fix is server-
side (prompt-cache code), small, and model-agnostic. Test: port to
llama.cpp-tp-adaptive, one normal agentic session, count re-encode events in
the log before/after (they are logged).

**M15. `iommu=pt` (GRUB + reboot, needs user approval).**
Current state: IOMMU active (ivhd0) in Translated mode, no `iommu=` on the
kernel cmdline (verified: `/proc/cmdline`, `HOW_TO_OPTIMIZE` dead-ends note
the RCCL warning). The tensor allreduce CPU-proxy path DMAs pinned staging
buffers continuously (2 transfers x 2 directions per allreduce, thousands
per prefill); passthrough removes IOMMU page-walk overhead from every one
of them. This is why RCCL's own warning names the flag. Risk: reboot + GRUB
change + DMA isolation disabled (single-user desktop, low practical risk);
revert = one cmdline token. Expectation: small, comm-bound, mostly prefill;
honest prior is 0-3%. Do NOT bundle with any other change when testing.

### Stability / ops (affect measured speed)

**M16. Stop the empty-messages client storm (A17, F27).** ~1000 req/s of
empty chat bodies, 107k task-counter ticks in 18 min, oversized-grammar
rejects, 50 MB log rotating every ~4 min. This is CPU + slot-lock
contamination under every benchmark and destroys the log evidence the method
rules require. Find the retry-looping client in llama_server's clients/harness
(read-only search is possible today; the kill is a user action). This is the
highest-leverage "mechanism" for bench validity.

**M17. Mesa 26.0.3 -> 26.2.1 (A4, re-scoped after reading the release
notes line-by-line today).** 26.1: 243 RADV commits; 26.2: 304. **Zero
compute-performance commits** (verified by scan - all graphics/RT/video/
HiZ). Two items touch OUR multi-queue path: 26.1 "fix potential GPU hangs
with secondaries on transfer queue" (our allreduce uses secondary command
buffers on a separate transfer queue) and 26.2 "clear gang cache flushes
when the command buffer is reset" (follower gang = RADV's multi-device
sync, the construct behind the PR #25051 decode-crawl workaround). So the
upgrade is a STABILITY play for the tensor path, not a speed play - expected
pp/tg delta ~0, risk = usual driver churn (needs user approval per
constraints).

**M18. `power_dpm_force_performance_level` A/B (A13) and AMDVLK via
`VK_ICD_FILENAMES` (A14).** Already queued from F17 with the evidence
(conflicting reports on DPM: "huge difference" vs "auto remained faster";
AMDVLK +11.7% dense on RDNA4 with rm_kq=1, tensor path uncharted on AMDVLK).
No change from today's research except ordering: run A13 first (two sysfs
writes, fully reversible) because AMDVLK is the one that could break the
#25051 cross-device path and needs a clean A/B window.

**M19. `GGML_VK_SUBALLOCATION_BLOCK_SIZE` (untried allocator knob).**
From the inventory diff. Changes Vulkan sub-allocation granularity -
candidate for VRAM-fragmentation relief at 256k (the 49/51 profile sits at
23,521 MiB; fragmentation, not raw size, is what eats the margin). Perf
effect expected null. One env var, one load, read the peak.

**M20. #24720 (vulkan: command buffer graph reuse, +25% claim on 4x B60)
stays a cherry-pick candidate, not a plan.** Still open, last updated
2026-06-20 (stalled 2 months). The claim is Intel-specific; the mechanism
(CPU-side graph rebuild per step) is real for prefill. If the author
revives it, port and test; if not, it is subsumed by the upstream meta-
backend refactor (JohannesGaessler). No action today.

### Verified-negative from this pass (do NOT try)

- **#26079** (mvq->MMQ crossover tuning, merged 08-20, in master but NOT in
  the pin): the #27290 A/B tested it standalone on the same model: prefill
  -1.96%, hot decode -0.70% -> rejected. Do not cherry-pick.
- **#27140**: same A/B, prefill -1.59% -> rejected.
- **#26048** (fused nvfp4 MMQ epilogue): nvfp4-only; we are Q8_0/Q6_K.
- **ngram-mod re-test** (thread #27154, gfx1151 single GPU): our local
  measurement stands (best +1.5% at sd 3.77, acceptance 0.762 -> 0.613);
  the thread could not be fully retrieved (JS-rendered) and local data
  beats single-GPU anecdotes. Stays a dead end per the method rules.
- **DSpark sidecar** re-test: #27290 independently confirmed it loses to
  embedded MTP even with a dedicated GPU (26.49 vs 49.61 t/s). Dead.
- **CUBLAS_COMPUTE_TYPE / FORCE_CUBLAS_COMPUTE_16F**: cublas/hipBLAS only
  serves F16/BF16 weight GEMMs; our weights are K-quants (custom mmvq/mmq
  kernels). The env vars cannot touch our decode path. Dead by code read.

---

## What did not reproduce / what I will not re-claim

- "Four presets load the vision tower on the XTX" - not reproducible from the
  current files; they inherit `MTMD_BACKEND_DEVICE=ROCm1`. The FACTS labels
  are generator artifacts.
- "2.9x prefill tension between our decode config and our prefill config" -
  the 2.9x pairs ROCm tensor with Vulkan layer. Same-build (Vulkan TP)
  tensor-vs-layer is 1.3x at d0 and inverts at depth.
- "rm_kq=1 is a free +1% for the current setup" - the current Q8_0 build does
  not use the rm_kq pipelines at all; the win (if real here) is Q6_K-only.
- The 81.67 tg bench number remains a repetitive-content-regime number (F24).
  Nothing today changes that; block A is what converts it into a depth-safe
  IRL prediction.
