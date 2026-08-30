# Handoff 6: READ THE INCIDENT FIRST. Decode is overhead-bound; tg 50 unproven.

Supersedes `2026-08-30-FLASHNEXT-HANDOFF-5.md`. That document's technical
content (presets, split map, placement laws, community numbers) is still
valid -- read it second. This file is the current state and the new rules.

---

## 1. INCIDENT: I CRASHED THE USER'S MACHINE. DO NOT REPEAT THIS.

The user came back to a dead desktop and had to log back into Ubuntu.

**Root cause on my side: every run I launched parked card0 (R9700) at
31,921 MiB of 32,590 MiB -- 669 MiB of headroom, 98% occupancy -- and held it
there for hours, unattended, while my VRAM watchdog guarded ONLY card1 (XTX)
and system RAM. I exempted card0 on purpose.** The reasoning I wrote down was
"the real risk on card0 is an allocator failure, which just exits the server,
not a PC crash." That was wrong and it is the direct cause of the exposure.

Forensics, for accuracy (the log did NOT capture a trigger line):
- No reboot; uptime was continuous. No kernel OOM-kill fired at any point.
- amdgpu `0000:14:00.0` (= card0, the R9700) re-initialized twice, 11:16:24
  and 11:17:28: PSP resume, SMU resume, GART re-enable.
- No ring-timeout / "GPU reset" / job-timeout message was recorded.
- Every run this session was **Vulkan** (`--device Vulkan0,Vulkan1`). No ROCm
  server was launched by me this session. Record this so you do not chase a
  ROCm-specific bug that may not exist; the user attributes the crash to a
  ROCm-class OOM, and the safety rules below apply either way.

### HARD RULES NOW IN FORCE

1. **Guard BOTH cards.** `/tmp/vram_guard.sh` now enforces
   `R9700_MAX=31600` and `XTX_MAX=24300`, plus `MEM_MIN=6` GiB, killing the
   test PID after 3 consecutive breaches (never on a single load spike).
   Never run an unattended test without it attached.
2. **A config needing >31,600 MiB on card0 is not allowed to run unattended.**
   This retires the 224K "hi-pp" and 256K builds as overnight configs -- they
   sit at 32,1xx-32,5xx on card0 by design, i.e. 65-444 MiB of margin.
3. ROCm additionally needs `--load-mode none` (mmap hangs its loader) = 26.8
   GiB anonymous RAM. Require MemAvailable >= 35 GiB before any ROCm launch,
   and step context UP from small. ROCm already OOM'd at 224K on the R9700
   compute buffer (handoff 5).
4. Kill exact PIDs only. Never `pkill -f`, `pgrep -f`, or `pkill -x bash`.
5. Never touch the user's own server; precheck `pgrep -x llama-server` and
   ports 1234/8080 before every launch. Port isolation is NOT sufficient
   protection -- two 50 GiB models cannot coexist on this box regardless.

### THE TENSION THE USER MUST DECIDE

Every config that meets the 180K+ context target requires card0 at 98-99%
occupancy. **"180K+ context" and "safe unattended" do not currently coexist.**
Do not silently pick one. Ask.

---

## 2. THE OTHER FAILURE: THE SESSION DIED AND NOTHING NOTICED

Work started 02:05 and **every process was dead by 02:19**. Last write to
every run dir and log is 02:18-02:19. Nothing ran from 02:19 to 11:19 -- nine
hours of an overnight window, wasted, because the agent session was killed and
had no heartbeat or restart path.

**If you are given an unattended window: commit after every single measurement,
and assume you will be killed at any moment.** Do not batch results in memory
or in /tmp and write them up "at the end." There was no end.

---

## 3. THE GOAL

User, verbatim: *"the most context, encode and decode speed possible with
optimal configs for ROCm and vulkan. do not stop until I get builds with 50+
tg, 500+ pp and 180k+ context at Q8 or better"* -- on UD-Q2_K_XL.

| target | goal | best measured | status |
|---|---|---|---|
| context | 180K+ | 229,376 (256K at ub 256) | MET |
| pp @d9K | 500+ | 613 | MET |
| KV | Q8+ | q8_0 / q8_0 | MET |
| **tg** | **50+** | **42.1** | **OPEN -- needs +19%** |

Three of four were already met by the preset shipped before this session.
**Decode is the entire remaining job.**

---

## 4. THE GOVERNING CONSTRAINT (most important finding; do not re-derive)

**Decode is ~89% overhead-bound, not bandwidth-bound.**

Per token: ~2 GB of weights read (6B active at Q2_K_XL).
- XTX   0.77 GB @ ~960 GB/s = 0.80 ms
- R9700 1.23 GB @ ~640 GB/s = 1.92 ms
- Layer split runs the cards **sequentially** (XTX layers 0-17, then R9700
  18-47), so the floor is 0.80 + 1.92 = **2.7 ms/token = ~367 t/s**.

Measured 42 t/s = 23.9 ms/token. **~21 ms is dispatch + sync overhead.**

Corroboration: tg is pinned near 42 at EVERY context (82K, 128K, 224K, 256K)
and on BOTH quants (IQ3_XXS and Q2_K_XL). It is a structural per-token cost,
independent of KV size and of weight size.

### What this rules out a priori -- do not spend time here

- Lowering body quant (Q2_K_S, IQ2) -- bandwidth is not the constraint.
- Rebalancing `--tensor-split` for bandwidth -- layer split is sequential, so
  total time is the sum regardless of ratio.
- Anything justified by "less data to read."

### What it points at

Reducing per-token overhead: node count, submission count, kernel launch cost,
and above all the **sequential two-card execution**, which wastes roughly half
the available GPU-seconds. `-sm row` is the structural fix and is
unimplemented on RADV ("does not support split buffers").

---

## 5. RULED OUT THIS SESSION, WITH DATA

### 5a. Upstream master -- CLOSED

**This model is now merged upstream.** `6c84c7d5d model: add Qwen3.8-Flash-Next
(qwen4exp) (#27742)` and `6fe749801 (#27880)` are both in master; our branch
`pr27880 @ aa1b06d0c` was 23 commits behind. A worktree already exists at
`llama.cpp-master` (built, working, `build-vulkan`).

Same-session head-to-head, identical 224K config, back to back:

| build | tg best | card0 | card1 |
|---|---|---|---|
| pr27880 @ aa1b06d0c | **42.10** | 31,921 | 23,463 |
| origin/master @ b8b743c3c | 41.70 | 31,921 | 23,461 |

Master is 0.9% slower = equal within noise. The MoE shader commits I expected
to win (#26686 row-ID/expert-count hoisting, #27925 mul_mat_id pad K) do
nothing for this model on RADV. **Rebasing is not a decode lever.** Master is
a valid maintenance fallback, not an upgrade.

### 5b. The GDN fusion commit -- NO-OP HERE

`866322481` flips `auto_fgdn`/`auto_flid` to false. I read the source: `auto_*`
only controls whether a load-time *probe* runs that can DISABLE fusion on a
device mismatch. Setting it false merely skips the probe. A mismatch logs at
WARN level, W lines ARE visible in our logs, and there are none -- **fused GDN
and fused LID were already active.** Saves two graph reserves at load, nothing
at decode.

### 5c. Dispatch knobs -- ALL INSIDE THE NOISE FLOOR

| config | tg best |
|---|---|
| baseline (`max_nodes_per_submit` default 100) | 42.10 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=500` | 43.13 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=2000` | 42.46 |
| `GGML_VK_MAX_NODES_PER_SUBMIT=100000` | 42.80 |
| `GGML_VK_FORCE_MMVQ=1` | 42.20 |

**There is no winner in this table.** Within a single run, pass1 -> pass2
varies by up to 1.3 t/s (41.80 -> 43.13). Between-config spread is smaller
than within-config noise. I initially reported the 43.13 as a win; that was
over-reading a best-of-two, and it was wrong.

### 5d. Carried over from handoff 5 (still closed)

MTP (tg 18.8 vs 28.3 at 77% acceptance), `-sm tensor` (aborts on the meta
backend whenever any tensor is CPU-resident, and the 26.8 GiB PLE always is),
`-sm row` (unimplemented), ROCm 10 (37% decode loss at 128K, 224K OOM),
KVarN (15.2 vs 28 tg), cross-card expert moves (-8.5%), whole-layer moves
(-32%).

---

## 6. MEASUREMENT METHODOLOGY -- FIX BEFORE YOU MEASURE ANYTHING

**The noise floor is ~1.5 t/s.** Any change smaller than that is not real.

Required protocol from here: `n_predict` 256+, **3 passes, discard pass 1 as
warmup, report the median**, and state the spread. `/tmp/fn_sweep.sh` currently
does 2 passes at 96 tokens and reports best-of -- **fix it before trusting it.**
A run takes only ~1.5 min (load ~30 s + generations), so a 3-pass protocol is
cheap. There is no excuse for best-of-two.

---

## 7. OPEN LEVERS, IN PRIORITY ORDER

1. **Per-op timing -- DO THIS FIRST.** `GGML_VK_PERF_LOGGER=1` (with
   `GGML_VK_PERF_LOGGER_FREQUENCY`) gives a per-op breakdown and will locate
   the 21 ms directly instead of guessing at knobs, which is how this session
   wasted its measurements. A partial CPU/GPU sample I captured before dying:
   `proc_cpu=237-281%` (2.4-2.8 cores busy) with `gpu0_busy=60-63%`,
   `gpu1_busy=23-32%` during decode. **Neither GPU is saturated and the CPU is
   using ~2.5 cores** -- consistent with submission/sync overhead. Confirm and
   localise this properly; it is the most informative unfinished measurement.
2. **`--fit on` + `--fit-target`.** Still never tested. `-ngl 99` silently
   kills it -- the live log line is `common_fit_params: failed to fit params to
   free device memory: n_gpu_layers already set by user to 99, abort`. **You
   must REMOVE `-ngl 99`** to test it. HF community reports automatic fit
   beating manual offload on CUDA rigs. This is also the most likely route to a
   config with SAFE card0 margins, which now matters more than before.
3. **`--load-mode none` (RAM-resident PLE).** llama.cpp itself prints
   `consider using --load-mode none for better performance` whenever CPU tensor
   overrides are combined with mmap. Never measured on Vulkan. Costs 26.8 GiB
   anonymous RAM -- check MemAvailable >= 35 GiB first. A safer variant that
   gets most of the benefit with no OOM risk: pre-warm the page cache
   (`cat <gguf> > /dev/null`) so PLE pages are resident but still evictable.
4. **Thread count sweep (`-t`).** The PLE gather is CPU work in the per-token
   critical path and the diagnostic above shows ~2.5 cores busy. Default is 16.
   Try 4 / 8 / 32. Cheap, never tested.
5. **ROCm optimal config.** Required by the goal ("optimal configs for ROCm and
   vulkan") but currently 37% behind Vulkan. One specific untried idea:
   `GGML_OP_OFFLOAD_MIN_BATCH` (default 32) is read by the CUDA/HIP backend but
   NOT by Vulkan. At decode batch 1 < 32, ops on CPU-resident tensors stay on
   CPU. Setting it to 1 may move the PLE path onto the GPU on ROCm. Also check
   whether HIP graphs are active -- graph capture is the one thing that could
   plausibly beat Vulkan on dispatch overhead. **Observe every safety rule in
   section 1; this is the path that can take the machine down.**

---

## 8. HONEST ASSESSMENT OF THE GOAL

tg 50 requires +19% on a workload that is 89% overhead-bound, where the
structural fix (`-sm row`, concurrent rather than sequential card execution) is
unimplemented on this driver, and where upstream's newest MoE shader work
produced nothing. **It may not be reachable on this hardware and driver.**

Community calibration (handoff 5): RTX 6000 PRO 96GB gets 59.7-70; 2x 4090D
48GB gets 44.7; dual 16GB consumer cards get 23.3. Our 42 at 128-256K is
already at the top of comparable hardware.

Do not quietly redefine the target. If the levers in section 7 plateau below
50, report the number reached, the spread, and the remaining gap, and let the
user decide.

---

## 9. FILES

- Live progress log (has the incident + all raw findings):
  `llama_server/docs/active/2026-08-30-OVERNIGHT-PROGRESS.md`
- Shipped preset:
  `llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh`
- Prior measurements: `llama_server/records/2026-08-30-flashnext-ctx/RESULTS.md`
- Prior handoffs: `docs/active/2026-08-29-FLASHNEXT-HANDOFF-{,2,3}.md`,
  `2026-08-30-FLASHNEXT-HANDOFF-{4,5}.md`
- Builds: `llama.cpp-qwen4exp/build-vulkan` (aa1b06d0c, current),
  `llama.cpp-master/build-vulkan` (b8b743c3c, equal speed),
  `llama.cpp-qwen4exp/build-rocm`
- Harnesses: `/tmp/fn_sweep.sh` (FIX TO 3-PASS MEDIAN FIRST),
  `/tmp/vram_guard.sh` (now guards both cards), `/tmp/fn_diag.sh`,
  `/tmp/flashnext_mmap_verify.sh`, `/tmp/flashnext_ctx_probe.sh`
- 200K prompt: `/tmp/bigprompt-200k.txt`
- Unfinished: deep-context tg at ~199K depth. Prefill decay measured to 72K
  (507 -> 190 t/s cumulative); the tg-at-depth number was never obtained.
