# How to optimize a model on this box

Written 2026-08-18 after tuning Qwen3.8 27B Q6_K. Audience: future agents and
future me. Read this **before** starting a sweep — most of it exists because
something went wrong the first time.

`TUNING_QUEUE.md` is the worklist (what to try). This is the method (how to try
it without wrecking the machine or shipping a number that is noise).

---

## 0. The rules that are not negotiable

- **The 7900 XTX (card1) must stay under 24,200 MiB.** It drives the desktop.
  Confirmed by the owner 2026-08-20; docs saying 24,300 are superseded.
  **This is not an OOM boundary** — llama.cpp allocates straight through it and
  takes the desktop with it. Nothing else stops you. Two distinct failures sit
  just above: throughput collapses from ~24,180 MiB (see the GTT cliff below),
  and the compositor is evicted at 24,388 MiB.
- **Start `bin/xtx-guard` before any sweep.** `xtx-guard &` (defaults to 24,200),
  kill it after. It kills `llama-bench` only, never `llama-server`.
- **The guard needs SUSTAINED overage, and it now requires 4 consecutive samples
  (~8s).** A single sample over the limit is a transient allocation spike during
  graph setup, not real usage. The single-sample version produced FALSE KILLS
  that made 163840 and 200704 both look like they failed at "24,245 MiB" — the
  identical peak at two very different context sizes, which is the tell: memory
  that does not scale with context is not context memory. Steady state was 23.3.
- **Only ever run ONE guard.** Stale guards from earlier sweeps silently kill
  later runs at their own (lower) limit; that corrupted an entire probe round.
  Kill them with `bin/kill-guards`, which scans `/proc` and skips its
  own PID — `pgrep -f xtx-guard` matches your own shell and is useless here.
- **Never use `pkill -f` or `pgrep -f`.** `-f` matches the full command line,
  including *your own shell's*, because your shell's cmdline contains the
  pattern you just typed. This killed the agent's own session four times in one
  day and once took down `llama-server` and an unrelated gateway with it. Use
  `pgrep -x <exact-name>` and kill explicit numeric PIDs.
- **Do not change GPU power, voltage, or firmware without asking.** Power
  capping was tested and rejected (see §7).
- **Set a timeout on every command.** 10 minutes or less. Use `until <cond>; do
  sleep N; done` to wait on things, not a long bare `sleep`.

## 1. Know your hardware mapping

| llama.cpp | GPU | VRAM | sysfs | PCI | role |
|-----------|-----|------|-------|-----|------|
| `ROCm0` | RX 7900 XTX (gfx1100) | 24,560 MiB | `card1` | 03:00.0 | drives the desktop — the constrained card |
| `ROCm1` | AI PRO R9700 (gfx1201) | 32,624 MiB | `card0` | 14:00.0 | headless, roomy, but slower under load |

**The sysfs numbering is inverted relative to llama.cpp's.** ROCm0 is card1.
Read XTX usage from `/sys/class/drm/card1/device/mem_info_vram_used` (bytes).

Verify before trusting it: `llama.cpp/build/bin/llama-bench --list-devices`.

### The `-ts` separator trap

```
llama-bench   -ts 36/29      # SLASH
llama-server  -ts 36,29      # COMMA
```
Mixing them up does not error. It silently gives you a different split, and you
will spend an hour explaining a result that never happened.

## 2. What each tool can and cannot measure

| tool | measures | **cannot** measure |
|------|----------|--------------------|
| `llama-bench` | raw pp/tg, VRAM, split & ubatch scaling | **speculative decode (MTP/DFlash) — at all** |
| live server + `/completion` timings | real end-to-end tg, draft acceptance, prefill | clean isolation; noisy unless the box is idle |
| `bin/vram-probe` | whether a (split, ctx) pair fits | speed |

**This is the single most important line in the document:** `llama-bench` runs
no speculation, so every `tg` it prints is **unspeculated** and roughly 2x below
what the server actually delivers. Q6_K measured `tg 18.13` in llama-bench and
**35.21 t/s** on the server. Use llama-bench for *ranking* configs, and the live
server for *absolute* numbers. Never quote a llama-bench tg as "the speed".

## 3. The order to tune in

Cheapest and highest-leverage first. Stop when gains fall inside noise.

1. **VRAM probe first, always.** `bin/vram-probe <model> /tmp/p.txt <ts> <ctx> ...`
   Find what fits before benchmarking. A model 3 GB larger simply cannot use the
   XTX-heavy splits, and knowing that up front saves the whole sweep.
2. **Parallel topology.** Compare `--split-mode layer` with `tensor` when the
   architecture supports it. This is now the largest decode lever, with a large
   prefill tradeoff; verify tensor initialization in the server log.
3. **Per-GPU split ratio.** Sweep 5+ points and look for a *shape*, not a winner
   (see §5). Under layer mode this previously gave +11.5% Q6_K prefill.
4. **ubatch.** Next lever, smaller. 256 has won every time; 128 can collapse.
5. **Context.** Trade only after 2–4 are settled; it is pure VRAM, no speed cost.
6. **KV precision, draft depth, batch size.** Model-specific, do not transfer
   between architectures.

## 4. Why splits matter here (the mechanism)

Layer split is **pipeline parallel**: the two GPUs run **sequentially** per
token, so the slower card's time adds directly to every single token. It is not
load balancing.

Tensor split (`--split-mode tensor`) instead creates Meta-device tensors and
splits work within tensors across both GPUs. On this host it cuts dense-model
decode latency, but the inter-GPU collectives roughly halve marginal
long-context prefill. It is a workload choice, not a universal upgrade: keep
layer split for ingestion-heavy requests and use tensor split when generation
dominates or the existing KV is reused. Confirm the server log contains the
Meta-device/tensor-parallel initialization lines; `/health` alone does not prove
that tensor mode was actually selected.

The R9700 pins at its 300 W cap during prefill, hits the power wall, and
downclocks. So **moving layers onto the XTX routes work around the constrained
card** and speeds up prefill, right up until the XTX runs out of VRAM.

That is the entire story of this machine's tuning: *push layers onto the XTX
until you hit the memory ceiling.* Which split wins is therefore a function of
model size — a bigger quant hits the ceiling sooner and loses the best splits.

## 5. Statistics: how not to fool yourself

- **`-r 2` with `-d` CRASHES.** It trips a llama.cpp bug in
  `llama_context::state_seq_set_data → ggml_backend_cuda_buffer_set_tensor`
  (SDMA page fault) when restoring a cached depth state on these hybrid models.
  Reproduced many times. **You must use `-r 1` at depth**, which means single
  unrepeated samples. `llama-server` never calls that path and is unaffected.
- **Getting repeats anyway:** list the same config multiple times in one sweep
  (`-ts 36/29,38/27,36/29,38/27`) at `-r 1`. Repeats without the state restore.
- **Known noise floors** (measure yours, do not assume): pp ~0.3% between
  independent runs; tg ±0.30 (~1.3%) at `-r 2`; live-server MTP decode ~6%
  pass-to-pass. A 3% tg difference from `-r 1` is about 2.5σ — real, but thin.
- **Trust shape over spikes.** The Q6_K decode curve went
  17.41 → 17.96 → 17.56 → **18.13** → 18.00 across five splits. Monotone-ish
  structure survives noise; one isolated high sample does not.
- **Generate enough tokens.** For raw ranking, time at least `pp10240` and
  `tg2048`; short `tg32` results are references, not shipping evidence. For live
  MTP, 400/prompt × 4 prompts × 2 passes = 3,200 tokens per config was enough to
  call a 2% gap a tie. A ~100-token sample (what you get if the model hits a
  stop token early) is worthless.

### The contamination trap — read this one twice

An A/B once showed the challenger **12% faster**. It was entirely fake: the
baseline had been measured while a 24 GiB download saturated the machine. On an
idle box the gap vanished to 2%.

**The tell:** a sample with *high* draft acceptance but *low* throughput
(24.18 t/s at 66.2% acceptance). Acceptance and throughput should move together.
When they diverge, you are measuring contention, not the model.

**Rule: A/B on an idle machine, and re-measure the baseline in the same session
as the challenger.** Never compare against a number from hours ago.

## 6. Recipe: benchmark a new quant end to end

```bash
cd ~/Desktop/local_llm
B=llama.cpp/build/bin/llama-bench
M=llama_server/models/YOUR-MODEL.gguf

# 0. verify the download before it ever touches the GPU
sha256sum "$M"          # compare against the publisher's manifest

# 1. guard up
llama_server/bin/xtx-guard 23400 & GUARD=$!

# 2. what fits?  (expect ~300 MiB per 8192 ctx on a 27B dense at f16)
llama_server/bin/vram-probe "$M" /tmp/probe.txt \
    32/33 245760  34/31 245760  36/29 245760  38/27 245760

# 3. one split at depth, -r 1 (see §5). Repeat this command per candidate.
#    One 10-minute timeout applies to each candidate, not to a multi-config run.
timeout 600 "$B" -m "$M" -d 65536 -p 10240 -n 2048 -r 1 \
   -ub 256 -b 6144 -ngl 99 -sm layer -mg 1 \
   -ctk f16 -ctv f16 -fa on -lm none \
   -ts 38/27 --progress -o md

# 4. shallow reference numbers, -r 2 is safe with no -d
$B -m "$M" -p 2048,8192 -n 32 -r 2 \
   -ub 256 -b 6144 -ngl 99 -sm layer -mg 1 \
   -ctk f16 -ctv f16 -fa on -lm none -ts <winner> --progress -o md

kill $GUARD

# 5. THE REAL NUMBER: launch and measure MTP on the live server, idle box
llama_server/bin/llamactl start --profile <profile>
#    then 4+ natural prompts x 400 tokens, 2 passes; read `timings` from
#    /completion — predicted_per_second, draft_n, draft_n_accepted.
```

Always launch long-running servers **detached** so your shell cannot take them
down: `setsid nohup llamactl start --profile P > log 2>&1 < /dev/null &`.
Confirm with `ps -o sid= -p <pid>` — it must differ from your shell's SID.

## 7. Dead ends — do not re-run these

| tried | result |
|-------|--------|
| R9700 at 330 W (firmware max) | pp 358.95 vs 354.52 = +1.2%, **noise**. Reverted. |
| Blaming the 330 W cap for a crash | Disproved by control: crashed identically at 300 W. |
| Blaming a concurrent download for a GPU page fault | Wrong. Page cache is reclaimable, DMA buffers are pinned. `PERMISSION_FAULTS: 0x3` with `MAPPING_ERROR: 0x0` = valid mapping, wrong permissions — not a missing page. |
| `-r 2` at depth | Crashes, see §5. Not a tuning result. |
| llama.cpp b10450 | −4.8% pp / −4.0% tg vs b10358 (loss of `-funsafe-math-optimizations`, upstream #26696). **Stay pinned at b10358 / `030ebb558`.** |
| ik_llama.cpp on AMD | Dead end. |
| mainline `b062ba735` (163 commits past the pin) | pp 269.91 vs pinned 274.96, tg flat. Pinned drifts 1.0% against itself, so this is NO measurable difference. The community claim of "+10-15% on a current build" is CUDA and does not reproduce on the ROCm tensor path. |
| DSpark drafter (`draft-dspark`) | **29% SLOWER than MTP**: 26.68 t/s @ 19.2% acceptance vs MTP's 37.56 @ 61.3%, identical config. It also ABORTS under `-sm tensor` (`ggml-backend.cpp:930`, draft `output.weight` lands in the `Meta()` buffer). Upstream's numbers are FP8 target + BF16 draft; against a Q8_0 GGUF the drafter predicts the wrong distribution. Fixed by open PR #26636 if you want to retest. |
| RCCL (`-DGGML_HIP_RCCL=ON`) | Builds and links, works with `-sm layer`/`none`, **SEGFAULTS under `-sm tensor`**. RCCL 2.27.7 builds the communicator, negotiates "P2P/direct pointer", then dies. Warns `Missing "iommu=pt" from kernel command line`; IOMMU here is active (ivhd0) in Translated mode. Needs a GRUB change + reboot to test. |
| `GGML_CUDA_P2P=1` | pp byte-identical to baseline (274.79) — the flag does NOTHING on this chipset topology. |
| `HSA_ENABLE_PEER_SDMA=0` | +0.6% = noise. |
| `--batch-size` under tensor mode | 2048/6144/12288 span 0.24% pp, 0.13% tg. `ubatch` drives per-reduction work; outer batch is just a loop. Do not re-sweep. |
| PRs #21170 + #27405 (ROCm multi-GPU state restore) | Both apply to the pin and build clean, but UNVALIDATABLE: the crash would not reproduce, 0/4 attempts on the unpatched control, including the exact geometry that failed hours earlier. PR #27405 itself calls the fault "intermittent". |
| KVarN on Vulkan | 47%/42% slower on the XTX. Use q8_0 KV instead. |
| Tensor Q6_K split 34/31 at c245760 | Fastest raw decode tested (27.92 tg), but live MTP allocation reached **24,226 MiB**, above the 24,200 hard ceiling. Rejected; do not retry at this context. |
| Tensor `main-gpu 0` | No speed gain: Q6_K 294.35 pp / 27.29 tg vs mg1 294.53 / 27.33; Q8_0 263.21 / 22.40 vs mg1 repeat 263.22 / 22.64. It also moves pressure to the desktop XTX. Rejected. |
| Raising MTP acceptance with high `p-min` | Q6_K n2/p-min 0.8 reached 87.0% acceptance but fell to 34.98 t/s vs n3/p-min 0 at 43.27 t/s. Acceptance alone is not the objective. |
| Q8_0 layer server near 256k on the current desktop | At c260096, request two raised the XTX to 24,197 MiB. The watcher stopped the exact server PID. Use c245760 for live A/B or reduce desktop VRAM; do not treat `/health` load-time usage as the peak. |

## 7b. Vulkan vs ROCm, settled 2026-08-19

**Verdict: stay on ROCm.** Vulkan doubles prefill and loses a fifth of decode.

`-sm tensor` is BROKEN on the stock Vulkan backend: `--tensor-split` is silently
ignored and every weight lands on device 0. Measured at ts 45/55 — Vulkan put
23,865 MiB on the XTX and **57 MiB** on the R9700, then died. ROCm at the same
settings: 14,703 / 14,819. A 27 GB model can therefore never fit under Vulkan
tensor mode, and any "Vulkan tensor works" result at tiny `-p`/`-n` is really a
single-card run.

PR #25051 (`vulkan: add allreduce function with cross-device CPU proxy and fix
Tensor Parallel crash`, draft, base 2026-06-29) FIXES the split: 14,417 / 14,778,
matching ROCm. With it, and after re-tuning Vulkan's own optimums:

| metric (Q8_0, tensor)      | Vulkan + PR | ROCm pinned | winner |
|----------------------------|------------:|------------:|--------|
| pp8192 @ 0 ctx             |  **946.74** |      435.62 | Vulkan +117% |
| pp2048 @ d65536            |  **561.74** |      282.31 | Vulkan +99% |
| pp2048 @ d200704           |      324.10 |           - | - |
| tg256 raw @ 0 ctx          |       20.27 |   **24.63** | ROCm |
| **MTP decode, 4x400 tok**  |       40.29 |   **48.83** | **ROCm +21%** |
| MTP draft acceptance       |       54.5% |       55.1% | tie |

The prefill ratio is flat at 1.99-2.02x across 0 / 32k / 65k, so it is a real
backend property, not noise. But MTP decode is what you feel, and ROCm wins it by
21% at IDENTICAL acceptance — i.e. Vulkan drafts fine, it is just slower per
token.

**Vulkan's flags do NOT match ROCm's.** Re-derived from scratch:
  ubatch: 256 -> 768.27, 512 -> 877.50, **1024 -> 950.13**, 2048 -> 948.29
          (ROCm's optimum is 512; Vulkan wants 1024)
  split @ ub1024, pp8192 / tg256:
          30/70 838.92/17.58   40/60 902.23/19.45   45/55 936.86/20.31
          **50/50 946.74/21.37 (pp peak)**   60/40 873.63/24.01
          65/35 834.37/25.80   **70/30 801.09/25.90 (tg peak)**   75/25 OOM
  Decode climbs toward the XTX for the same reason as ROCm: the R9700 is behind
  a Gen4 x4 chipset link. But XTX-weighted splits cannot hold context (65/35
  exceeds the ceiling by c=98304), so 50/50 is the only usable point.

FP16 context ceiling on Vulkan at ts 50/50 + ub 1024, measured with REAL work
(`-p 2048 -n 128`, not a trivial probe):
    200704 -> 23,538 MiB, exit 0   <- practical max, 762 MiB headroom
    225280 -> 24,122 MiB, exit 0   <- technical max, only 178 MiB headroom
Quantized KV is documented as unsupported under `-sm tensor`, but the Vulkan PR
build ACCEPTS `-ctk q8_0 -ctv q8_0` and runs. Untested for quality/regression.

Adopting it is still premature: it is an unmerged draft on a base six weeks
behind the pin. Re-test when it lands in master.

## 8. Vendor performance claims

Verify provenance, then verify the claim; they are different jobs.

The HauhauCS "Q6_K_P / FastMTP, up to 3.02x TG" build checked out on
provenance — SHA256 matched, Ed25519 manifest signature verified — and still
failed on merit. Its byte size landed within **960 bytes** of unsloth's
UD-Q6_K_XL, i.e. the same recipe relabeled, and it measured 8.7% slower decode
with *no* draft-acceptance advantage (55.8% vs 57.6%).

**A file size that matches a known quant to within a kilobyte is the same
quant.** Check that first; it costs one `stat` and can end the investigation.

## 9. Reference numbers, Qwen3.8 27B (2026-08-18)

llama-bench, `d131072 · ub256 · f16 KV · r=1` — **unspeculated**:

| quant | best split | pp2048 | tg32 | shallow pp8192 |
|-------|-----------|-------:|-----:|---------------:|
| Q8_0 | 1/1 | 315.79 | 15.26 | 1099.06 |
| **Q6_K** | **38/27** | **353.30** | **18.13** | **1207.99** |
| Q6_K_P (Hauhau) | 34/31 | 323.44 | 16.68 | 1207.90 |

Live server, real MTP, 4 natural prompts × 400 tokens × 2 passes, idle:

| quant | decode | draft acceptance | XTX @ c=245760 |
|-------|-------:|-----------------:|---------------:|
| **Q6_K @ 38/27** | **35.21 t/s** | 57.6% | 23,021 MiB |
| Q6_K_P @ 34/31 | 34.51 t/s | 55.8% | 22,616 MiB |

MTP multiplier is roughly **1.5x** over unspeculated (Q6_K: 23.31 → 35.21 at
shallow depth). Acceptance is content-dependent and swings 45–83% by prompt.

VRAM scaling, Q6_K at ts 38/27: **~304 MiB per 8,192 tokens** of context.
229,376 → 22,413 · 245,760 → 23,021 · 262,144 → 23,631 MiB.

### Tensor-parallel decode profiles (2026-08-19)

llama-bench b10358, `d65536 · p10240 · n2048 · f16 KV · r=1` —
**unspeculated**. Repeated cells show both independent runs:

| quant / topology | split | ubatch | pp10240 | tg2048 |
|------------------|-------|-------:|--------:|-------:|
| Q6_K layer | 38/27 | 256 | 565.85 / 566.55 | 20.04 / 19.99 |
| **Q6_K tensor** | **1/1** | **256** | **294.53** | **27.33** |
| Q8_0 layer | 1/1 | 256 | 509.40 / 508.56 | 18.44 / 17.63 |
| Q8_0 tensor | 45/55 | 256 | 263.93 / 263.22 | 22.48 / 22.64 |
| **Q8_0 tensor** | **45/55** | **512** | **275.20** | **22.63** |

Tensor mode gains 36.4% raw Q6_K decode and 25.5% raw Q8_0 decode against
the layer means, while losing 48.0% and 45.9% prefill respectively. The Q8_0
tensor ubatch sweep was 128=235.14/22.57, 256=263.58/22.56,
288=225.98/22.37, and 512=275.20/22.63 (pp/tg); the warned discontinuity at
288 is real, and ub512 ships. With live
MTP at c245760, fixed four-prompt/two-pass harness:

| quant / topology | samples | decode | acceptance | XTX peak |
|------------------|--------:|-------:|-----------:|---------:|
| Q6_K layer 38/27 n3 | 8 | 32.231 t/s | 51.62% | 23,907 MiB |
| **Q6_K tensor 1/1 n3** | **8** | **43.576 t/s** | **51.01%** | **22,211 MiB** |
| Q8_0 layer 1/1 n3 | 8 | 34.352 t/s | 53.46% | 23,225 MiB |
| **Q8_0 tensor 45/55 n3, ub512** | **8** | **44.813 t/s** | **53.98%** | **21,908 MiB** |

The paired tensor gains are +35.2% for Q6_K (95% CI +9.35 to +13.34 t/s)
and +30.5% for Q8_0 (95% CI +8.77 to +12.15 t/s). Q6_K n2 versus n3 was a
2.1% statistical tie across 16 samples each; Q8_0 n2 was 7.2% slower and n4
was a 1.4% tie/loss. Both tensor profiles therefore keep n3. These are separate
decode profiles because the prefill regression makes tensor mode worse for
one-shot long-document ingestion with short answers.

### Vulkan tensor-parallel, Q6_K (2026-08-20)

PR #25051 build (`llama.cpp-vulkan-tp`), unsloth `Qwen3.8-27B-Q6_K.gguf`,
llama-bench `-sm tensor -fa 1 -ctk/-ctv f16 -p 8192 -n 256 -r 2`, ub 1024:

| ts (XTX/R9700) | pp8192 | tg256 |
|---------------:|-------:|------:|
| 50/50 | **889.60** | 23.17 |
| 60/40 | 811.35 | 25.63 |
| 65/35 | 770.92 | 27.47 |
| **70/30** | 729.82 | **29.15** |
| 75/25 | 701.73 | 29.14 |
| 80/20 | 670.47 | 28.09 |
| 85/15 | 642.86 | 26.23 |

**Decode has an interior maximum.** It is not "more XTX is better": tg peaks at
70/30 and falls away on both sides. Prefill, by contrast, falls monotonically as
work moves to the XTX. 70/30 dominates 75/25 (equal tg, 4% more pp), so the
useful range ends there regardless of memory.

The three profiles that ship from this sweep, **unspeculated** (`-r 3`):

| profile | ts | ub | KV | ctx | pp | tg |
|---------|----|---:|----|----:|---:|---:|
| q6k decode q8_0 | 70/30 | 512 | q8_0 | 196,608 | 696.78 | 28.65 |
| q6k decode f16 | 68/32 | 512 | f16 | 147,456 | 717.81 | 28.37 |
| q6k context | 50/50 | 1024 | f16 | 262,144 | 889.60 | 23.17 |

Q6_K versus Q8_0 at the same split is roughly −6% prefill / +8% decode. With MTP
on, the live server runs these at ~50–56 t/s, which is past the Q8_0 ROCm tensor
profile (48.83) and the Q8_0 Vulkan one (40.29) — but see "Never compare configs
on MTP decode" before using those figures for anything but a sanity check.

### Size from SERVER-ONLY VRAM, not the whole-card number

The XTX drives the desktop, so `mem_info_vram_used` on card1 is server +
desktop, and **the desktop is not a constant**. It moved 1,696 -> 2,332 MiB
during a single tuning session (Firefox, VS Code, a qemu VM), and this repo has
recorded it as high as 3.0 GiB.

That is enough to make identical configs look irreproducible. Measured
whole-card peaks for two configs, each run more than once:

| config | run 1 | run 2 | run 3 |
|--------|------:|------:|------:|
| ts 68/32, c 163840 | 23,763 | 24,247 | — |
| ts 70/30, c 147456 | 23,092 | 24,034 | 24,282 |

Nothing about the server changed between those runs. The desktop grew
underneath it. An entire round of "this config is unstable, reject it"
conclusions came out of that before the baseline was checked.

**The rule:**

```
server_only + worst_case_desktop <= 24,200
server_only <= 21,200 MiB          # allowing 3.0 GiB of desktop
```

Measure `server_only` as (peak whole-card under sustained load) minus (baseline
sampled with no server running, immediately before launch). Record the baseline
in the result row; a peak without its baseline is not a measurement you can
compare to anything.

### q8_0 KV is a context multiplier, not a speed win

PR #25051's build accepts `-ctk/-ctv q8_0` under `-sm tensor` even though
quantized KV is documented as unsupported there. What it buys is memory.

**Unspeculated** llama-bench, ts 70/30, ub 512, `-r 3`:

| KV | depth | pp | tg |
|----|------:|---:|---:|
| f16 | 0 | 710.56 ± 0.43 | 29.11 ± 0.28 |
| q8_0 | 0 | 696.78 ± 1.16 | 28.65 ± 0.28 |
| f16 | 32768 | 572.20 ± 0.72 | 27.26 ± 0.27 |
| q8_0 | 32768 | **511.02** ± 2.66 | 27.38 ± 0.17 |

q8_0 KV is **decode-neutral** — a tie at depth, −1.6% at d0 — and costs **10.7%
of prefill at 32k depth**. What it actually buys is `server_only` memory:

| KV | ctx | server_only |
|----|----:|------------:|
| f16 | 131,072 | 20,839 |
| f16 | 147,456 | 21,564 |
| f16 | 163,840 | killed at 24,290 whole-card |
| q8_0 | 196,608 | 20,024 / 20,114 |
| q8_0 | 229,376 | 20,810 |

**q8_0 holds 192k in less memory than f16 holds 128k.** f16 could not reach 160k
at any decode-friendly split. Both ship as separate profiles; the choice is
context length versus KV precision and prefill, not speed.

An earlier revision of this section claimed q8_0 won on decode. That was MTP
acceptance noise — see below — and it was wrong.

### Never compare configs on MTP decode

**Turn speculation OFF for any config-vs-config decode measurement.** Draft
acceptance is stochastic at temp 1.0 and ranged **47% to 68% across repeat runs
of identical configs**. Decode t/s tracks acceptance closely, so one config
produced 49.3 and 56.7 t/s on consecutive runs — a 15% spread from sampling
alone, which is larger than nearly every real effect in this document.

With MTP on you are measuring the sampler, not the configuration. Two ship
decisions in this session were made on MTP numbers and both had to be reversed
once the same configs were re-run unspeculated.

Use `llama-bench` with speculation off and `-r 3` for every comparison. Measure
MTP once, at the end, on the config you already chose — it is a multiplier
(roughly 2x here, ~50–56 t/s observed) and a sanity check that acceptance is in
its normal band, not a tiebreaker.

### The GTT cliff: throughput dies silently, just above the limit

The single most important finding of this sweep. At c=262144, moving five split
points toward the XTX:

| ts | peak XTX | pp | tg |
|----|---------:|---:|---:|
| 50/50 | 22,782 MiB | 671 | 43.41 |
| 55/45 | **24,180 MiB** | **140** | **27.00** |

Prefill lost 79% and decode 38%. There was **no OOM and no crash** — the server
started, served, and answered correctly the whole time. It was simply running at
a fifth of its prefill speed.

The margin here is thin. Measured either side of the line at c=262144:

| peak XTX | outcome |
|---------:|---------|
| 23,953 MiB | clean — tg 44.76, the best decode measured at this context |
| 24,180 MiB | collapsed — pp 140, tg 27.00 |

**227 MiB separates full speed from a fifth of it**, and the cliff starts within
20 MiB of the 24,200 ceiling. Budget real slack below it: the desktop baseline
moves by more than 200 MiB on its own between an idle and a busy session.

Two consequences:

1. **Two different failure points sit above the ceiling.** Throughput collapse
   (~24,180 MiB) arrives before compositor eviction (24,388 MiB), and it is
   silent. Do not treat a passing health check as evidence a config is sized
   correctly — the collapsed config served correct answers the entire time.
2. **A config that got slower may be out of memory, not badly tuned.** Always
   read peak VRAM next to the speed number. Tuning by t/s alone reads this as
   "tensor split is slow at high context" and produces a wrong conclusion that
   then gets written into a profile header.

### ubatch is a memory knob, not only a speed knob

Same model, ts 50/50, c=262144, peak XTX under load:

| ub | pp8192 | peak XTX | verdict |
|---:|-------:|---------:|---------|
| 512 | 823.91 | — | ships in the decode profile |
| 1024 | 882.46 | 22,782 MiB | ships in the context profile |
| 2048 | 910.88 | 23,953 MiB | **rejected** — +3.2% pp for +1,171 MiB |

Dropping 1024 → 512 at ts 70/30 *gained* decode (54.52 vs 53.55) and freed
948 MiB. The two shipped Q6_K profiles therefore use **different** ubatches on
purpose; do not "unify" them.

Related: `--main-gpu 1` does **not** park the whole ubatch compute buffer on the
other card. Roughly a GiB of it still lands on the XTX at ub 2048. Measured, not
assumed.

### Two smaller results

- **MTP n_max 4 loses on Q6_K**: 51.39 t/s at 51.0% acceptance versus 53.55 at
  58.2% for n3. The longer chain lands a smaller fraction of itself. n3 has now
  been confirmed independently on Q8_0 and Q6_K — stop re-testing n4.
- **The launcher's `** SPILL: N GiB of this server is in system RAM **` warning
  is a false positive whenever `--cache-ram` is set.** That memory is the prompt
  cache, which lives in system RAM by design. It appears only after the first
  generation fills it. The decode profile shows 1.27 GiB "spilled" while
  simultaneously posting the fastest decode ever measured here.

## 10. Checklist before shipping a config

- [ ] Peak XTX measured **under sustained request load**, not just at the
      load-time plateau. These differ by a lot: Q6_K ts 38/27 plateaus at 23,021
      MiB on load and peaks at **23,907 MiB while serving** (+886 MiB). Sizing
      off the plateau alone once implied 1,179 MiB of headroom where only ~293
      MiB existed. The desktop baseline also moves 1.1 → 3.0 GiB within a
      session, so leave real slack.
- [ ] Peak XTX is under 24,200 MiB **with real slack**, not merely under it.
      Throughput collapses within ~20 MiB of that line while every health check
      still passes, and the desktop baseline swings by more than 200 MiB on its
      own. See "The GTT cliff".
- [ ] Winner re-measured in the same session as the runner-up, idle box.
- [ ] Real MTP decode measured on the live server, not quoted from llama-bench.
- [ ] Profile header records the sweep, the mechanism, **and what was not
      measured**.
- [ ] Menu/catalog figures are measured, not estimated. If you have not measured
      it, write `NOT YET BENCHMARKED` — never invent a plausible number.
