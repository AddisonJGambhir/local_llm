# Handoff: Qwen3.8-Flash-Next decode is 2-5x slower here than it should be

Written 2026-08-29 by the prior agent. **This work is unfinished and the prior
agent did not succeed.** The model runs, but at roughly half the throughput
people are getting on materially worse hardware. Treat the numbers below as a
floor to beat, not as a result to preserve.

## The gap you are being asked to close

| setup | VRAM | quant | tg |
| --- | --- | --- | --- |
| 2x RTX 3080 | 20 GB total | Q4_K_XL | **16-17 t/s** |
| RTX 5070 Ti + 5060 Ti | 32 GB total | IQ3_XXS | **23.3 t/s** |
| AI MAX+ 395 (unified) | shared DDR5 | IQ4_XS | 11.15 t/s |
| 2x RTX 4090D | 48 GB total | Q4_K_XL | 44.7 t/s |
| **this box (7900 XTX + R9700)** | **56.6 GB total** | **IQ4_XS** | **9.14 t/s** |

A pair of 3080s with 20 GB of VRAM — a third of this machine's — is nearly
twice as fast. That is the whole problem. Source: the HF discussion
`unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3` ("Share your model speed here").

## START BY DOING RESEARCH. DO NOT START BY BENCHMARKING.

The prior agent went almost straight to benchmarking and burned hours
rediscovering things that were already written down publicly. Before you change
a single flag:

1. Read `unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3` in full and extract
   **every** posted command line, especially from AMD and multi-GPU posters.
   The `--flash-attn on` + `q8_0` KV combination that produced the single
   largest win here (+39%) came from one such post, found late.
2. Read the qwen4exp PRs and their review threads end to end: **#27742**
   (danielhanchen, adds the arch), **#27739** (JJJYmmm, competing
   implementation), **#27880** (ngxson, graph splits). Check whether either has
   been superseded or merged, and whether newer qwen4exp PRs exist. #27739 in
   particular was never evaluated here and may be faster.
3. Search llama.cpp issues for GDN / Gated DeltaNet / SSM Vulkan performance.
   36 of this model's 48 layers are GDN. That is the prime suspect and it was
   never isolated.
4. Read the Qwen reference implementation on how the n-gram (PLE) table is meant
   to be consumed. The paper describes **asynchronous prefetching overlapped
   with model computation**. Determine whether llama.cpp does this at all. If it
   does not, that is a real optimisation and possibly the whole gap.
5. Check `unsloth.ai/docs/models/qwen3.8-next` and the vLLM recipe page for
   flags that have llama.cpp equivalents.

Do not trust the prior agent's reasoning. Do trust the measured numbers in
`records/2026-08-29-flashnext/RESULTS.md`, which were taken from instruments.

## Current best config (the floor)

    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
    llama.cpp-qwen4exp/build-vulkan/bin/llama-server \
      -m llama_server/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
      -c 8192 -ngl 99 --device Vulkan0,Vulkan1 \
      --split-mode layer --tensor-split 40,60 -lm none \
      -ot 'per_layer_token_embd\.weight=CPU' \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0

    pp 23.0 t/s   tg 9.14 t/s   XTX 24,148 MiB   R9700 32,486 MiB

Build is `llama.cpp-qwen4exp` at `aa1b06d0c` (PR #27880, contains #27742).
`build-vulkan` works. `build-rocm` compiles but cannot fit the model.

## Measured facts. These came from instruments, not arithmetic. Reuse them.

- `per_layer_token_embd.weight` = **27,465 MiB (26.8 GiB)**, iq4_nl, **one
  tensor**, 320,001,536 rows x 160 dims = 51.2B params.
- Per-layer experts = **1,136 MiB** (`ffn_down_exps` 450 iq4_nl + `ffn_gate_exps`
  343 iq3_s + `ffn_up_exps` 343 iq3_s).
- 48 blocks, `full_attention_interval 4` -> **12 attention layers, 36 GDN**.
- `head_count_kv 2`, k/v length 256 -> **24 KiB/token at f16** -> 6.0 GiB at
  262144. KV is cheap on this arch; context is not the constrained resource.
- Architecture string is `qwen4exp`. 512 experts, 10 routed + 1 shared.
- Usable VRAM: **XTX ~22,700 MiB** (24,200 ceiling minus ~1,500 desktop idle),
  **R9700 ~32,567 MiB**. Total ~55,267 MiB. Capacity ratio ~41:59.

## SSD OFFLOAD IS A FALSE PREMISE

The goal was framed around SSD offload. It is not happening and is not needed.
Disk throughput measured during sustained generation: **0.0-0.2 MB/s**. The
26.8 GiB table fits in the 59 GiB of RAM with room to spare. Do not design
around mmap paging. `-lm none` (no mmap) is 9% FASTER than mmap here.

## Dead ends. Do not spend time re-deriving these.

- **`-sm tensor` is structurally unavailable.** Aborts with "multi buffers are
  not supported by the meta backend" whenever any tensor is on CPU. The model
  cannot fit without offloading the n-gram table. Therefore tensor split can
  never be used for this model unless that meta-backend limitation is fixed
  upstream. (Fixing it upstream may in fact be the highest-value action
  available — see Leads.)
- **`-sm row`**: "device Vulkan0 does not support split buffers" on RADV, and
  the same on ROCm0. Unimplemented on both.
- **ROCm backend**: always tries to allocate exactly **34,621.23 MiB on device
  1** (a 32,624 MiB card) regardless of `--tensor-split` or how many expert
  layers are offloaded. The figure never moves. Wants ~58,680 MiB total vs
  55,267 available. Either find why the allocation is invariant, or leave ROCm
  alone.
- `--flash-attn on` **by itself** does nothing (6.49 -> 6.47). Only helps paired
  with q8_0 KV (-> 9.03).
- PR #27880 improves **prefill only**: pp 14.6 -> 23.0 (+58%), tg unchanged.

## llama.cpp flag traps discovered the hard way

- **`-ncmoe` is silently ignored when `-ot` is also passed.** Both write the same
  override list; `-ot` wins. No warning is printed. The prior agent wasted
  several runs on this.
- **Repeated `-ot` flags do not accumulate** — only the first applies. Use one
  `-ot` with comma-separated `pattern=buffer` pairs.
- `-v` prints `tensor X (NNN MiB type) buffer type overridden to Y` per tensor.
  **Use this to verify every override actually applied** before trusting a run.
- llama-bench `-ts` and `-dev` use **slashes**; the server uses **commas**.
- **`--device` is mandatory on Vulkan.** This box enumerates THREE Vulkan
  devices; `Vulkan2` is the CPU (RADV RAPHAEL_MENDOCINO) advertising 32 GB free,
  and auto-selection will silently include it and destroy performance. Always
  pass `--device Vulkan0,Vulkan1`.

## The actual diagnosis, as far as it got

Profiled during a 400-token generation:

    XTX 32-53% busy   R9700 49-73% busy   CPU ~38%   disk 0.0-0.2 MB/s

The GPUs **alternate** — when one is at 32% the other is at 73%, then they swap.
Neither ever saturates. That is layer split serialising them: each card computes
its half of the layers while the other waits. Decode is ~28 GB/s effective
against ~800 GB/s of aggregate bandwidth, so it is **not** bandwidth-bound and
**not** disk-bound. CPU sits at a steady ~38% with zero disk I/O, which is more
CPU work than a gather over an embedding table should require.

## Leads, roughly in order of expected value

1. **Is the PLE gather being done efficiently?** CPU at 38% with no disk I/O is
   the strongest unexplained signal. Per token this should be 16 rows x 160 dims
   ~= a few KB. Profile it (`perf top` against the server process during
   generation). If llama.cpp is doing something worse than a gather, fixing that
   is the whole gap.
2. **Try PR #27739** (JJJYmmm's independent qwen4exp implementation). Never
   evaluated here. May have a different, faster graph.
3. **Test UD-Q3_K_XL** — already downloaded, at
   `models/Qwen3.8-Flash-Next/UD-Q3_K_XL/`, 90 GB, ~10 GB smaller main model.
   If tg scales with model size, the bottleneck is bandwidth; if it does not,
   that isolates the kernels. This is the cheapest discriminating experiment
   available and should probably be run first.
4. **GDN kernel performance on Vulkan.** 36 of 48 layers. Compare per-op timings
   against the CUDA path in the PR threads, or build a microbenchmark.
5. **Fix the meta-backend multi-buffer limitation** so `-sm tensor` becomes
   usable. Tensor split is worth ~+37% tg on the 27B model on this exact
   hardware (see `records/2026-08-29-rocm10/`). If it could be made to coexist
   with a single CPU-resident tensor, it would likely close most of the gap,
   because it makes the two GPUs work concurrently instead of alternating.

## Constraints you must respect

- **Never exceed 24,200 MiB on the 7900 XTX (card1).** It drives the desktop and
  overshooting has nearly crashed the machine. Measure the **settled** value
  after a real generation, not the load-time peak — load spikes above the
  ceiling and then drops. The prior agent killed several valid runs by guarding
  during load.
- **Do not start, stop, or reconfigure the user's llama-server.** They launch it
  themselves with their own specs. Use port 18234+ for tests and kill what you
  start.
- Never use `pkill -f` or `pkill -x bash`. The prior agent ran `pkill -x bash`
  and killed a running download's wrapper. Use `pgrep -x` on exact process names
  and kill by PID.
- Do not change system packages, GPU power limits, or kernel parameters without
  asking first.
- Timeouts of 5 minutes or less on every command.

## Files

    llama_server/records/2026-08-29-flashnext/RESULTS.md   full measurements
    llama_server/records/2026-08-29-rocm10/RESULTS.md      27B backend comparison
    llama.cpp-qwen4exp/                                    the build (PR #27880)
    llama_server/models/Qwen3.8-Flash-Next/UD-IQ4_XS/      93.7 GB, verified
    llama_server/models/Qwen3.8-Flash-Next/UD-Q3_K_XL/     90 GB, verified, untested
