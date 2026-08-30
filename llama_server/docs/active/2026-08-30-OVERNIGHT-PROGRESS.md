# Overnight optimization run -- LIVE PROGRESS (updated continuously)

Goal (user, 2026-08-30, before sleep): Q2_K_XL Qwen3.8-Flash-Next.
**tg 50+, pp 500+, ctx 180K+, KV Q8 or better. Optimal config for BOTH
Vulkan and ROCm.** Do not stop until met.

## STATUS: IN PROGRESS

### Scoreboard vs goal

| target | goal | best so far | met? |
|---|---|---|---|
| context | 180K+ | 229,376 (and 262,144 at ub256) | YES |
| pp @depth | 500+ | 613 @d9K | YES |
| KV | Q8+ | q8_0/q8_0 | YES |
| **tg** | **50+** | **42.1 shallow / 41.8 (224K build)** | **NO -- +19% needed** |

The single open gap is decode speed.

### Key reframing (this session)

Decode is NOT bandwidth-bound. 6B active params at Q2 = ~2 GB read/token;
the two cards can do that in ~2 ms. Measured is 24 ms/token. We are ~12x off
the bandwidth limit, so **decode is overhead/latency-bound** (48 layers x
per-layer dispatch + sync, 512-expert routing, CPU-resident PLE forcing
graph splits). Optimization must target per-layer overhead, not bandwidth.

## SAFETY RULES IN FORCE (user: a crash = total failure)

- XTX (card1) drives the desktop. Sustained > 24,300 MiB = kill the test.
- ROCm needs `--load-mode none` (mmap hangs its loader) = 26.8 GiB anonymous
  RAM. Require MemAvailable >= 35 GiB before ANY ROCm launch.
- ROCm already OOM'd at 224K on the R9700 compute buffer. ROCm context must
  be stepped UP from small, never jumped to.
- Watchdog `/tmp/vram_guard.sh` kills the test PID on sustained breach only
  (3 consecutive samples), never on a load spike.
- Never `pkill -f`, `pgrep -f`, or `pkill -x bash`. Kill exact PIDs only.
- Never touch the user's own server. Precheck before every launch.

## WORK LOG

### 1. Deep-context measurement (the pending item from handoff 5) -- RUNNING

224K hi-pp build, UD-Q2_K_XL, ub 1024, ot = PLE + blk.30 gate/up to CPU,
199K-token prompt, n_predict 64, port 18252, run dir /tmp/flashnext-deep200k.

Confirmed at load: RSS 923 MB, RssFile 477 MB -> mmap is working exactly as
intended, PLE is file-backed and NOT anonymous. d0 tg = 41.2 / 41.8.

Prefill decay (cumulative t/s at depth), this is the real answer to "what
happens deep":

| depth | cumulative pp |
|---|---|
| 30,720 | 506.9 |
| 36,864 | 482.6 |
| 49,152 | 443.0 |
| 53,248 | 376.5 |
| 59,392 | 291.8 |
| 63,488 | 259.7 |

Instantaneous prefill is ~100 t/s by 63K depth and still falling. NOTE: a
`nice -19 -j6` master build was running concurrently from ~18% onward, so
this curve is slightly pessimistic. Direction is unambiguous regardless.

### 2. Upstream master now carries this model -- BUILDING

`6c84c7d5d model: add Qwen3.8-Flash-Next (qwen4exp) (#27742)` and
`6fe749801 (#27880)` are both merged. Our branch pr27880 @ aa1b06d0c is 23
commits behind. **No cherry-picking needed; master supports the model.**
Worktree: `llama.cpp-master`, build dir `build-vulkan`.

Commits in that range that touch this model's hot path:

- `90c26fcd4` Vulkan: hoist row IDs + expert count in shaders (#26686)
  -- 512-expert MoE decode. BEST CANDIDATE.
- `77f132cb1` vulkan: mul_mat_id pad K rather than N (#27925)
  -- mul_mat_id IS the expert matmul op.
- `57291f264` ggml: alloc dependencies in graph_optimize (#27301)
- `b387ddfd8` vulkan: fix missing view-alias deps in ggml_vk_graph_optimize

### 3. GDN fusion -- INVESTIGATED, NOT THE WIN (ruled out)

`866322481` flips `auto_fgdn`/`auto_flid` to false. Read the source: `auto_*`
only controls whether a load-time *probe* runs that can DISABLE fusion on a
device mismatch. Setting it false just skips the probe and leaves fusion on.

Checked our live log: a device mismatch logs at WARN level, and W lines ARE
visible in our log. There are none. **Fused GDN + fused LID are already
enabled here.** That commit saves two graph reserves at load, not decode time.

(Aside: all libllama INFO lines are filtered from the server log during load,
which is why `load_tensors:` buffer lines never appear. Only W and above.)

### 4. Confirmed: --fit is dead while -ngl 99 is set

Live log line: `common_fit_params: failed to fit params to free device
memory: n_gpu_layers already set by user to 99, abort`. To test --fit,
-ngl 99 must be REMOVED. Still untried.

### 5. Vulkan master build: NO DECODE WIN (ruled out)

Same-session head-to-head, identical 224K config, back-to-back:

| build | tg best | card0 | card1 |
|---|---|---|---|
| pr27880 @ aa1b06d0c (current) | **42.10** | 31,921 | 23,463 |
| origin/master @ b8b743c3c | 41.70 | 31,921 | 23,461 |

Master is 0.9% SLOWER, i.e. equal within noise. The MoE shader commits
(#26686 row-ID/expert hoisting, #27925 mul_mat_id padding) do not help this
model on RADV. Upstream-rebase path is CLOSED as a decode lever. Keep the
existing build; master is a valid fallback, not an upgrade.

### 6. Where the 24 ms/token actually goes (the governing analysis)

Per token: ~2 GB of weights read (6B active at Q2_K_XL).
- XTX  0.77 GB @ ~960 GB/s = 0.80 ms
- R9700 1.23 GB @ ~640 GB/s = 1.92 ms
Layer split runs the cards SEQUENTIALLY (XTX layers 0-17, then R9700 18-47),
so the floor is 0.80 + 1.92 = **2.7 ms/token = ~367 t/s**.

Measured 42 t/s = 23.9 ms/token. **~21 ms (89%) is dispatch + sync overhead,
not memory traffic.** This is confirmed by tg being pinned near 42 at EVERY
context (82K, 128K, 224K, 256K) and on both quants -- it is structural
per-token cost, independent of KV size.

Consequences:
- Bandwidth-side levers (lower body quant, rebalancing ts for bandwidth) are
  near-worthless. Do not pursue.
- The only lever that can reach tg 50 is REDUCING PER-TOKEN OVERHEAD.
- `-sm row` would let both cards work the same layer concurrently and is the
  one structural fix, but it is unimplemented on RADV.

### 7. Vulkan backend tunables found (full list in ggml-vulkan.cpp)

The relevant ones for dispatch overhead:
- `GGML_VK_MAX_NODES_PER_SUBMIT` -- **default 100**. Graph is thousands of
  nodes, so decode does ~15-30 queue submits per token, each with a sync.
  Prime suspect. Sweeping 500 / 2000 / 100000.
- `GGML_VK_FORCE_MMVQ` / `GGML_VK_DISABLE_MMVQ` -- mat-vec quantized kernel,
  which IS the decode kernel at batch 1.
- `GGML_VK_PERF_LOGGER` -- per-op timing; use to locate the 21 ms directly.
- `GGML_VK_DISABLE_ASYNC`, `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` -- transfer path
  for the CPU-resident PLE.
- `GGML_VK_ENABLE_MEMORY_PRIORITY`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`.
- `GGML_OP_OFFLOAD_MIN_BATCH` (default 32) is CUDA/Metal ONLY -- not read by
  the Vulkan backend. It IS relevant on the ROCm path: at batch 1 < 32, ops on
  CPU-resident tensors stay on CPU. Worth testing when ROCm is revisited.

### 8. Dispatch knob sweep: ALL INSIDE NOISE (no lever found)

224K config, same binary, back-to-back, best-of-2 passes:

| config | tg best |
|---|---|
| baseline (default, submit=100) | 42.10 |
| GGML_VK_MAX_NODES_PER_SUBMIT=500 | 43.13 |
| GGML_VK_MAX_NODES_PER_SUBMIT=2000 | 42.46 |
| GGML_VK_MAX_NODES_PER_SUBMIT=100000 | 42.80 |
| GGML_VK_FORCE_MMVQ=1 | 42.20 |

**Do not read a winner out of this table.** Within a single run, pass1 -> pass2
differs by up to 1.3 t/s (e.g. 41.80 -> 43.13). The between-config spread
(42.1-43.1) is SMALLER than the within-config noise. Correct conclusion:
nodes-per-submit and MMVQ forcing do nothing measurable here. An earlier note
in this session called 43.13 a win; that was over-reading a best-of-2.

METHODOLOGY FIX for everything after this point: n_predict 256+, 3 passes,
discard pass 1 (warmup), report median. Anything under ~1.5 t/s is noise.

### 9. Next: stop guessing knobs, measure the constraint

Running `/tmp/fn_diag.sh`: samples process CPU% and per-card
`gpu_busy_percent` during a sustained 400-token decode.

- If one core pegs near 100% and GPUs sit low -> CPU-submit-bound. Then the
  fix is fewer graph nodes / cheaper submission, and driver choice matters.
- If GPUs are busy and CPU is low -> the overhead is GPU-side kernel launch
  and RADV/kernel efficiency is the ceiling.

This decides which remaining levers are worth any time at all.

## INCIDENT 2026-08-30 ~11:16 -- GPU RESET DURING UNATTENDED RUN

User reported the PC crashed. Evidence gathered after the fact:

- NO reboot (uptime 18:49 at time of check).
- NO kernel OOM kill. RAM was never the problem; the RAM guard was irrelevant.
- amdgpu `0000:14:00.0` re-initialized TWICE (11:16:24, 11:17:28): PSP resume,
  SMU resume, GART re-enable. That PCI address is **card0 = the R9700**.
- No ring-timeout / "GPU reset" / job-timeout line was captured, so the exact
  triggering fault is NOT established. Do not claim one.
- Swap drained 7/7 -> 0/7 GiB, consistent with the desktop session restarting.

### What I got wrong (root cause of the exposure)

`/tmp/vram_guard.sh` guarded ONLY card1 (XTX, 24,300 ceiling) and system RAM
(6 GiB floor). **It deliberately did not guard card0.** The reasoning I wrote
into this file earlier was:

  "the real risk on card0 is OOM inside the allocator, which llama.cpp handles
   by failing to allocate (server exits) -- not a PC crash"

That was wrong. Every single run this session parked card0 at 31,921 MiB of
32,590 MiB -- **669 MiB of headroom, 98% occupancy** -- and held it there for
hours, unattended. The R9700 is the card that reset. The user's instruction
was "mind the VRAM overhead at all times"; I applied a ceiling to one card and
exempted the other.

### Corrective actions

1. Guard now covers BOTH cards. card0 ceiling 31,600 MiB (~1,000 MiB
   headroom), card1 unchanged at 24,300.
2. No config that requires >31,600 MiB on card0 may run unattended. That
   retires the 224K "hi-pp" and 256K builds as overnight configs -- both sit
   at 32,1xx-32,5xx on card0 by design, i.e. 65-444 MiB of margin.
3. Nothing relaunches without the user's explicit go-ahead.

### Consequence for the goal

The 180K+ configs that met the context target all depend on running card0 at
98-99% occupancy. If that is what destabilises the R9700, then "180K+ ctx" and
"safe unattended" are in tension and the user has to make that call, not me.

