# Handoff 7: independently verified Vulkan gains; MTP sidecar is incompatible

Supersedes `2026-08-30-FLASHNEXT-HANDOFF-6.md`. Treat all earlier handoffs as
leads only. The user explicitly warned that Claude's report contained major
errors. The authoritative measurements from this continuation are in
`llama_server/records/2026-08-30-flashnext-independent/RESULTS.md`.

## 1. Stop and inspect before doing anything

At handoff, no `llama-server` should be running. Verify it; never assume:

```bash
pgrep -a -x llama-server
ss -ltnp | grep -E ':(1234|8080|1825[0-9])[[:space:]]' || true
awk '/MemAvailable/ {print $2/1048576 " GiB available"}' /proc/meminfo
```

Port 8080 normally belongs to Docker/searxng. Do not kill it. If a user
`llama-server` is running, stop and ask. Different ports do not make two
50-GiB model allocations coexist safely.

Hardware naming is reversed between Vulkan and sysfs:

| physical card | Vulkan name | sysfs | reported total |
|---|---|---|---|
| RX 7900 XTX | Vulkan0 | card1 | 24,560 MiB |
| AI PRO R9700 | Vulkan1 | card0 | 32,624 MiB |
| CPU/iGPU | Vulkan2 | n/a | exclude explicitly |

The user explicitly raised the R9700 guard during this continuation. Current
checked-in defaults are:

- R9700/card0: 32,200 MiB, leaving about 424 MiB physical margin.
- XTX/card1: 24,200 MiB, leaving about 360 MiB physical margin.
- MemAvailable: at least 6 GiB while running; prelaunch harness minimum 30 GiB.
- Kill only after three consecutive five-second breaches.

Use `llama_server/bin/vram_guard.sh` through the harness. Do not weaken either
card's guard or use `pkill`. Kill exact PIDs only.

## 2. Current goal and honest status

The active goal is 50+ shallow token generation, 600+ prompt processing, and
well over 128K allocated context with Q8_0-or-better KV. The body currently
used is UD-Q2_K_XL because it is the only all-GPU body with enough context
headroom. The user rejects Q4 KV; every result below uses Q8_0 K and V.

| requirement | independently measured best | status |
|---|---:|---|
| shallow tg, sustained | 49.1566 t/s (3-pass median) | close, not met |
| shallow tg, 131K sample | 49.6960 t/s (one sample) | not a sustained claim |
| pp at 8,948 tokens | 1,044.05 t/s | met |
| tg at 8,948-token live depth | 38.5933 t/s | +28.75% over control |
| allocated Q8 context | 131,072 safely on patched build | met only literally, not “well over” |

Do not round 49.16 or the single 49.70 sample to 50. The next agent still has
two open tasks: sustain 50+, and step guarded context above 131,072.

## 3. The winning source tree — preserve it

The active experimental worktree is:

```text
/home/addison-gambhir/Desktop/local_llm/llama.cpp-qwen4exp-vulkan-fusions
```

It is detached at `aa1b06d0c` and intentionally dirty. **Do not reset, clean,
checkout, or discard it.** Its current binary is:

```text
llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server
```

Current source changes:

- `3c82e945`: Vulkan mat-vec elementwise epilog fusion.
- `a4c5df63`: fused REPEAT+MUL+ADD Vulkan kernel.
- Main-model hyper-connection graph wiring for those fusions; MTP-only graph
  hunks were deliberately excluded.
- Main-model portions of `b6d995d5`: avoid mean-collapse copies and use the
  skinny inject mat-vec path for batched prefill.
- Vulkan backend portion only of `233c0d9c`: radix TOP_K for large k. The
  regressive branch's QSA gather/cache model rewrites were excluded.

`git -C llama.cpp-qwen4exp-vulkan-fusions status --short` should show five
modified files and two new shaders. The worktree's `AGENTS.md` prohibits
commits without explicit user approval, so these changes were not committed
inside that llama.cpp clone. Root-repository records and harness work were
committed normally.

Rebuild incrementally with:

```bash
CCACHE_DIR=/tmp/flashnext-ccache \
cmake --build llama.cpp-qwen4exp-vulkan-fusions/build-vulkan \
  --target llama-server -j 16
```

The offline UI download warning is harmless.

## 4. Verified results from this continuation

### Baseline and rejected full branch

Exact Q2, Q8 KV, 49,152 context, split 400/600, ubatch 512 control on
`aa1b06d0c`:

- measured 44.7901, 45.0315, 46.4274 t/s; median 45.0315.
- R9700/XTX 28,755/22,675 MiB.

The full apepojken `qwen4exp-spec-mtp` branch at `843d5750`, with MTP disabled,
regressed to median 39.3025 t/s versus an identical ordinary-mmap upstream
control at 45.3681. Do not adopt that full branch. It lacks
`--tensor-read-lazy`, so comparisons used `FN_LAZY=skip` on both sides.

### Isolated fusion win

At the original 400/600 placement:

- 48.0208, 48.6790, 48.4513 t/s; median 48.4513.
- +7.59% over the exact 45.0315 control.
- R9700/XTX 28,755/22,688 MiB.

Moving one discrete layer to the XTX with split 425/575:

- 48.9480, 49.4649, 49.1566 t/s; median 49.1566.
- +9.16% over the original control.
- R9700/XTX 27,776/23,656 MiB.

The current full patched binary (fusions + skinny/copy changes + radix) was
48.6658 t/s median in a shallow regression check. This is about 1% below the
fusion-only placement result and may be noise or one of the later changes.
Isolate before removing anything because the same full stack wins enormously
at real prompt depth.

Sampler changes did not close the gap. `top_p;min_p;temperature` fell to
46.3281 because it filtered the full 150K vocabulary. `top_k;temperature`
measured 49.0762. The request is already greedy (`temperature:0, top_k:1`).

### Real-prompt stack-level A/B

Both runs used a 35,000-character slice of `/tmp/bigprompt-200k.txt`, exactly
8,948 prompt tokens, context 16,384, ubatch 1,024, split 425/575, Q2 body,
Q8_0 K/V, and 128 forced decode tokens:

| build | pp | tg at depth |
|---|---:|---:|
| current isolated patched build | 1,044.0525 | 38.5933 |
| untouched `aa1b06d0c` | 725.9201 | 29.9749 |

This is +43.83% pp and +28.75% deep tg. The patched continuation was coherent
and recognized the synthetic repetition; upstream degenerated into repetition.
This is a one-pass exact A/B, large enough to establish a stack-level win but
not attribution among fusions, skinny/copy changes, and radix TOP_K.

Run directories:

```text
/tmp/fn-sweep/vulkan-fusions-skinny-radix-q2-depth9k-ignoreeos
/tmp/fn-sweep/upstream-q2-depth9k-ignoreeos-control
```

### Safe 131,072 placement step

Current full patched binary, split 400/600, ubatch 256, Q8 KV:

- warmup 47.7522 t/s; one measured sample 49.6960 t/s.
- R9700/XTX 29,775/23,466 MiB.
- run: `/tmp/fn-sweep/vulkan-patched-q2-131k-placement`.

This leaves useful guard margin. Step to 147,456 and then 163,840; do not jump
straight to a huge context. Start with one warmup and one measured 128-token
placement probe at each step.

## 5. MTP is open, but the available sidecar does not load

The corrected native-MTP branch and binary are:

```text
llama.cpp-qwen4exp-vulkan-mtp                 # branch qwen4exp-spec-mtp
llama.cpp-qwen4exp-vulkan-mtp/build-vulkan/bin/llama-server
commit 843d5750
```

Available sidecar:

```text
llama_server/models/Qwen3.8-Flash-Next/MTP-Q4_K_M/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf
size 2,622,313,344 bytes
```

The first corrected-branch MTP launch used native `draft-mtp`, three draft
tokens, Q8 draft KV, and placed the full sidecar on Vulkan1/R9700 at 16K. It
failed before draft allocation:

```text
check_tensor_dims: tensor 'blk.48.nextn.fc_embd.weight' not found
```

Run directory:

```text
/tmp/fn-sweep/vulkan-mtpbranch-q2-mtp3-r9700-16k
```

This is a sidecar schema mismatch, not an OOM and not an MTP speed result. The
old WIP MTP loss must not be treated as evidence against this corrected
implementation. Next MTP work:

1. Dump the sidecar tensor names and inspect the corrected branch's converter
   or model expectations for `blk.48.nextn.fc_embd.weight`.
2. Find or convert a compatible MTP sidecar. Do not rename blindly; verify all
   required tensor shapes.
3. Retry at 16K with target `--fit off`, full draft on Vulkan1/R9700, draft
   Q8 KV, and the checked-in guard.
4. If it crosses 32,200 MiB, try draft tensor overrides or
   `--spec-draft-device none --spec-draft-ngl 0` for CPU draft placement.
5. Always compare MTP on/off on the same `843d5750` binary. That branch's
   no-MTP baseline is slower than upstream, so MTP must overcome its own base
   regression before it is useful.

The exact failed extra arguments were:

```text
--spec-type draft-mtp
--spec-draft-model .../Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf
--spec-draft-n-max 3 --spec-draft-n-min 1
--spec-draft-device Vulkan1 --spec-draft-ngl all
--spec-draft-type-k q8_0 --spec-draft-type-v q8_0
```

## 6. Closed dead ends and corrected false claims

- KVarN K6/V5 is closed: native route verified, but only 29.42 t/s at 16K
  and 30.72 t/s at 64K. IQ3 + all experts + 180K cannot physically fit with
  its measured 26.6 KiB/token growth.
- IQ3_XXS safe 49K median is 43.9283 t/s. Q2 is 45.0315 in the exact same
  config and saves 2,904 MiB aggregate VRAM. Q2 is the capacity path.
- The IQ3_S register-spill shader patch did not improve end-to-end speed.
- Claude's “89% host overhead / 2.7 ms real GPU work” claim is false. Vulkan
  timestamp profiling measured about 9.095 ms/token on XTX plus 15.054
  ms/token on R9700, the same order as whole-token time. Instrumentation itself
  slowed throughput, so use it for attribution, not headline speed.
- PR 27739 was voluntarily replaced by merged PR 27742; there is no reason to
  build 27739 as a separate performance candidate.
- PLE “SSD offload” is a false model. The 27,466 MiB table remains CPU mmap;
  measured disk traffic during decode was essentially zero. `--load-mode none`
  duplicates it into anonymous RAM and is not the answer on this 64-GiB host.

## 7. Harness and exact next commands

Use `llama_server/bin/flashnext_sweep.sh`. It now supports:

- `FN_PROMPT_FILE` and `FN_PROMPT_CHARS` for real prompts.
- `FN_WARMUP_PASSES=0` for one expensive deep-prompt pass.
- `FN_CURL_TIMEOUT` for long prefills.
- per-pass pp, prompt token count, and tg output.

Suggested next context step:

```bash
FN_TAG=vulkan-patched-q2-147k-placement \
FN_BIN=/home/addison-gambhir/Desktop/local_llm/llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server \
FN_CTX=147456 FN_TS=400,600 FN_UB=256 FN_PORT=18254 \
FN_NPRED=128 FN_PASSES=1 \
./llama_server/bin/flashnext_sweep.sh
```

Run with direct GPU access as required by the environment. Before every launch,
verify no server, RAM, VRAM, and port. Commit the record after every result.

## 8. Repository state and checkpoints

Root repository commits from this continuation:

```text
c76509c records: reject bundled Vulkan Qwen branch
3e61893 records: validate isolated Vulkan fusion gain
e081872 records: measure fused one-layer placement
19b48da bench: support real Flash-Next prompts
747eee5 records: validate deep Vulkan Qwen gains
1e15eb2 bench: raise guarded R9700 ceiling to 32GB
9798d42 bench: set R9700 guard ceiling to 32.2GB
```

The root worktree contains unrelated user edits and untracked models/docs.
Do not stage or alter them. Only add the exact Flash-Next files you change.

Primary detailed record:

```text
llama_server/records/2026-08-30-flashnext-independent/RESULTS.md
```
