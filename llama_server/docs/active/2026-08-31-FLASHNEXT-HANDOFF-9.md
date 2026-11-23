> **SUPERSEDED by `2026-09-01-FLASHNEXT-HANDOFF-10.md`.**

# Handoff 9: actual 199K residency is solved; deep speed work is active

Supersedes `2026-08-30-FLASHNEXT-HANDOFF-8.md`. Earlier shallow results are not
valid evidence of high-context performance. The user requires Q8 KV or better,
all experts in VRAM, and roughly 200K of context that has actually been read.

## 1. Immediate state: a benchmark may still be running

At handoff, the exact same-prompt baseline A/B was running:

- server PID reported by the harness: `279932`
- unified exec session was `70853` (a new agent cannot rely on inheriting it)
- run directory:
  `/tmp/fn-sweep/baseline-fusions-layer-depth199k-sameprompt-ctx200704-ts400-ub128`
- last observed depth before this handoff: about 164K actual tokens
- binary:
  `llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server`

First inspect, do not start another model:

```bash
pgrep -a -x llama-server
tail -10 /tmp/fn-sweep/baseline-fusions-layer-depth199k-sameprompt-ctx200704-ts400-ub128/gpu-memory.tsv
tail -20 /tmp/fn-sweep/baseline-fusions-layer-depth199k-sameprompt-ctx200704-ts400-ub128/guard.log
ls -l /tmp/fn-sweep/baseline-fusions-layer-depth199k-sameprompt-ctx200704-ts400-ub128/out1.json
```

If `out1.json` exists, extract the final timings. If the process is still
running, let it finish. Never launch two copies of this model. Different ports
do not make the RAM or VRAM allocations coexist.

## 2. Non-negotiable methodology and safety

- A performance claim requires at least 10K actual live prompt tokens.
- A 200K-context claim requires actually reading roughly 200K tokens. Allocating
  a 200K/262K slot and testing at 9K does not reveal Vulkan migration.
- Vulkan startup GTT/system RAM around 2GB can be normal. Spill means correlated
  VRAM loss and GTT growth while the prompt is populated.
- ROCm OOM can reset the desktop or freeze/log out the user. Start small and use
  the guard. Do not jump ROCm directly to deep context.
- Guard both physical cards after readiness: R9700 card0 at 32,500 MiB and XTX
  card1 at 24,200 MiB, three strikes. Load-time limits are physical capacity.
- Port 8080 is normally searxng. Do not kill it.
- Commit root records after every material measurement. Do not commit inside a
  llama.cpp worktree without explicit user approval; its `AGENTS.md` forbids it.

Hardware naming:

- sysfs card0 = R9700 32,624 MiB; headless, about 57 MiB idle
- sysfs card1 = 7900 XTX 24,560 MiB; desktop card
- ROCm enumeration is reversed: ROCm0 = XTX, ROCm1 = R9700
- Vulkan tensor ratios and ROCm ratios are XTX,R9700, so `40,60` puts more on
  the R9700.

## 3. Correct shipped preset and best verified actual-199K result

Root commit `081b3a2` changed Vulkan preset 12 to the genuinely populated
configuration:

```text
ctx 200704
split mode layer
tensor split 400,600
ubatch 128
batch 2048
K/V q8_0/q8_0
flash attention on
no speculation
CPU only: per_layer_token_embd.weight and token_embd.weight
cache-ram 8192
mmap + lazy tensor reads
all experts on GPU
```

Menu text is now:

```text
12) layer, populated 200k — no spec — q8_0 — 200704
    (348 pp / 13.5 tg @d199k) [L 400,600 · txt]
```

Authoritative measurement on the earlier 199,320-token prompt:

- pp `347.8360`
- tg `13.5271`
- R9700 `30304..30310 MiB`
- XTX `23524..23620 MiB`
- no correlated migration, zero guard strikes

This is about 8.1x faster than the user's original spilling 220K run at 1.67
t/s, but it is still far below the goal of 50+ tg and 600+ pp at depth.

Earlier frontier facts:

- `ctx212992 split375,625 ub128`, 199,320 tokens: 338.99 pp / 13.41 tg,
  resident.
- ub256 at split375 migrated late around 160-172K twice.
- `ctx200704 split375 ub192`: 347.26 pp / 13.09 tg, resident but no win.
- split410 moves another whole layer to XTX and is unsafe.

## 4. New online research: the best lead is PR 27977 and its split-outs

Live GitHub API was queried on 2026-08-31. Relevant current PRs:

- https://github.com/ggml-org/llama.cpp/pull/27977
  `qwen4exp: reduce the generation slowdown as context grows`
  Reports a real 149K conversation improving 30 -> 56 tg and 1617 -> 2148 pp.
  Its biggest finding is an O(context * 256 sequence slots) CPU predecessor
  scan: about 20 million iterations per token at 88K. It also adds QSA gather,
  indexer head slices, a windowed scan, and a bitmap used-cell set.
- https://github.com/ggml-org/llama.cpp/pull/27992
  Adds a `(seq,pos)` index; reports 8.92 -> 24.25 tg at 240K on 2x L40S.
  It is complementary but more invasive and adds overhead to every model.
- https://github.com/ggml-org/llama.cpp/pull/28023
  Indexer-head slice summation; prompt-only improvement, reported 2170 -> 2366
  pp at 55K.
- https://github.com/ggml-org/llama.cpp/pull/28104
  New qwen4exp MTP port; reports +50% decode at 70K on Vulkan.
- https://github.com/ggml-org/llama.cpp/pull/28118
  Keeps speculative recurrent-state checkpoints on device. This explains the
  earlier MTP disaster: host checkpoints cost about 600ms per round. Reported
  no-draft 32.4, fixed MTP 41.5, and high-acceptance code 56.6 tg.
- https://github.com/ggml-org/llama.cpp/pull/28123
  Alternative recurrent-state rollback implementation for qwen4exp MTP.
- https://github.com/ggml-org/llama.cpp/pull/27973
  Vulkan GDN cache-write fusion, avoiding 48 CPY nodes/token. Draft and not yet
  hardware-validated by its author.
- https://github.com/ggml-org/llama.cpp/pull/28102
  ROCm gfx1201 flash-attention tuning; improves deep prefill on 27B dense, not
  decode. May matter later but is not the first Flash-Next lever.

PR 27739 was inspected. It is the older closed model-add PR, not a missed
backend optimization. Its important configuration note was CPU PLE offload.

## 5. PR 27977 actual test and the live A/B

Isolated worktree:

```text
llama.cpp-flashnext-deepopt
branch pr27977
HEAD db40b22d69f0
build build-vulkan
```

The test prompt is the first 611,800 characters of:

```text
llama.cpp-flashnext-deepopt/ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

Measured independently with `llama-tokenize`: 199,339 tokens without BOS;
server `prompt_n` was 199,340. Do not infer token count from characters.

PR 27977 combined result at exactly 199,340 actual tokens:

- pp `167.96057070413508`
- tg `14.260419018645496`
- R9700 `30212 MiB`, XTX `23734 MiB`
- GTT at end `142/732 MiB`; VRAM remained flat, so no migration
- result directory:
  `/tmp/fn-sweep/pr27977-layer-depth199k-ctx200704-ts400-ub128`

This is only +5.4% tg versus the older different-prompt 13.527 result and the
prompt speed looks much worse, but that comparison is not valid because the
prompt changed. The same-prompt baseline described in section 1 was launched
to settle it. Record its result before doing anything else.

PR 27977 commit order:

```text
618450d07ca8 qwen4exp: sum indexer heads by slices
39efb6390e13 kv-cells: stop the sequence scan once all sequences are seen
d7ca8ea59435 qwen4exp: attend selected cells instead of masking the window
28e37926cecf kv-cache: scan only the n-gram window in get_prev_tokens
b88668b47d06 kv-cells: keep the used-cell set as a bitmap
db40b22d69f0 nits
```

The QSA gather commit `d7ca8ea` is the main suspect for the poor Vulkan pp.
The next highest-value experiment is an isolated build with the CPU scan wins
(`39efb639`, `28e37926`, optionally `b88668b`) and indexer pp win (`618450d`),
but without `d7ca8ea`. Test at 10K first for correctness, then actual 199K.
Do not combine PR 27992 at the same time; its review thread reports that
stacking both can regress TG.

## 6. Tensor-parallel experiments: do not repeat

Vulkan TP experimental worktree:

```text
llama.cpp-flashnext-tp
```

It ports PR 25051 multi-buffer/all-reduce support. At `ctx16384`, Q8 KV,
tensor `40,60`, ub128, 11,547 actual tokens:

- 351.69 pp / 19.26 tg
- safe allocation about 29.3GB R9700 / 22.1GB XTX

Vulkan layer mode at the same depth is roughly 810 pp / 39 tg, so TP loses
about 2x. A local uncommitted change forces the small-reduction ring threshold
to zero; it built but was not benchmarked after the user redirected to ROCm.
Do not take it deep unless it unexpectedly approaches layer speed at 10K.

Stock ROCm TP on the model does work despite a newer CUDA issue report:

- exact 11,547-token prompt, ctx16K, tensor40,60, ub128
- 314.42 pp / 21.826 tg
- stable 29.77GB R9700 / 22.71GB XTX
- run `/tmp/fn-sweep/rocm-stock-tp-depth11k-ctx16k-ts40-ub128`

Still far behind Vulkan layer mode, so no deep run.

ROCm P2P worktree:

```text
llama.cpp-flashnext-rocm-p2p
```

It applies the bounded Aug 30 hybrid HIP all-reduce patch from
`stew675/llama-cpp-rdna-boosts`. The published gate rejected the gfx1100 XTX,
so the first 326.49/21.855 run was actually stock fallback. In the isolated
tree only, the gate was relaxed to this machine's gfx1100+gfx1201 pair while
retaining the 20ms bounded-spin poison recovery. The real heterogeneous run:

- 292.89 pp / 19.925 tg
- repeated 20ms peer-arrival timeouts and butterfly recovery
- safe completion, no machine crash
- run `/tmp/fn-sweep/rocm-hybrid-hetero-tp-depth11k-ctx16k-ts40-ub128`

This is a clear regression. Do not run the older direct-P2P patch: it has an
unbounded GPU spin and can wedge the machine.

## 7. MTP is reopened, not closed

Earlier MTP testing showed about a 2x loss even at 99% acceptance. The new
PR 28118 identifies the reason: qwen4exp speculative rounds serialized full
recurrent state to host memory. The eight-line ON_DEVICE checkpoint fix turns
MTP into a win in reported Vulkan tests. Therefore the earlier result does not
close MTP.

After the no-spec CPU-scan subset is measured, evaluate PR 28104 plus either
28118 or the newer rollback alternative 28123. Requirements:

- first verify which community MTP sidecar layout exists locally; earlier
  handoffs found the then-tested sidecar incompatible, so do not assume it now
  loads
- Q8 target KV remains mandatory
- minimum 10K real depth for any claim; ideally test 70K and then 199K
- monitor draft acceptance and report raw target/draft costs
- Vulkan speculative livelock fixes referenced by PR 28104 must be included
  before unattended testing

## 8. Root commits and dirty-tree preservation

Relevant root commits:

```text
d7a04b4 flashnext: require populated-context validation
7a6c61c flashnext: validate 200K residency frontier
081b3a2 flashnext: ship populated 200k Vulkan preset
```

The root worktree contains many unrelated user changes. Do not stage them.
The only uncommitted root change from this continuation before this handoff was
`llama_server/bin/flashnext_sweep.sh`, adding `FN_SPLIT_MODE` so the harness can
test tensor and layer modes. Include only that file plus this handoff and the
superseded banner if committing.

The known-good fusion worktree intentionally has seven dirty files. Preserve
them exactly:

```text
llama.cpp-qwen4exp-vulkan-fusions
```

Do not clean/reset/checkout any llama.cpp worktree. All experimental work was
kept in separate worktrees. No server should be left running after the active
same-prompt baseline completes.

## 9. Exact next actions

1. Recover the final same-prompt baseline result from section 1 and write it to
   a new root record. This decides the real value of combined PR 27977.
2. Build/test a PR 27977 subset without QSA gather commit `d7ca8ea`; start at
   10K, then run the calibrated 199,340-token prompt.
3. If CPU-scan-only materially improves deep tg without hurting pp, ship it in
   an isolated maintained binary/profile and update preset 12.
4. Then test the corrected MTP stack (28104 + 28118 or 28123). This is the only
   currently identified path with credible 50%+ deep decode upside.
5. Only after those, consider Vulkan GDN fusion 27973 or ROCm FA tuning 28102.

Be honest about the target: before these new PRs, verified actual-199K speed is
about 13.5-14.3 tg, not the shallow 49-50 tg previously advertised.
