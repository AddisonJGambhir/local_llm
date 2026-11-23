# SUPERSEDED by `2026-08-29-FLASHNEXT-HANDOFF-3.md`

# Handoff 2: Qwen3.8-Flash-Next -- shipped at tg 42.5, one unverified change

Supersedes `2026-08-29-FLASHNEXT-HANDOFF.md`, whose numbers (tg 9.14) are stale.
That document's conclusion "SSD offload is a false premise" is WRONG and is
corrected below.

## READ THIS FIRST: this work OOMed the user's machine

On 2026-08-29 the shipped preset was started while the user's own llama-server
was already running. RAM went past 64 GB, the kernel killed their server and
other applications, and swap is still full. Two causes, both avoidable:

1. The preset shipped `--load-mode none`, which disables mmap and loads the
   26.8 GiB PLE table as ANONYMOUS RAM the kernel cannot evict. Fixed in
   commit 3d63f1e -> `--load-mode mmap`.
2. The prior agent checked once, early, whether the user's server was running,
   then launched dozens of test servers over several hours without re-checking.

### MANDATORY before every single server launch

    pgrep -x llama-server                       # must be empty, or STOP
    ss -ltn | grep -E ':(1234|8080) '           # user's port -- must be empty
    free -g                                     # need >= 30 GiB available
    cat /sys/class/drm/card{0,1}/device/mem_info_vram_used

Port isolation is NOT sufficient. Using port 18234 avoids a bind conflict and
does NOTHING about RAM or VRAM. Two 50 GiB models cannot coexist on this box
regardless of ports. If the user's server is up, STOP AND ASK. Do not "just
quickly test".

Other standing constraints:
- NEVER exceed 24,200 MiB on the 7900 XTX (card1) -- it drives the desktop.
  Measure the SETTLED value after a real generation; load spikes above it and
  then drops, and killing on the spike throws away valid runs.
- Never `pkill -f` or `pkill -x bash`. The prior agent ran `pkill -x bash` and
  killed a running download's wrapper. Use `pgrep -x` and kill by PID.
- Do not start/stop/reconfigure the user's server. They launch it themselves.
- No system packages, GPU power limits, or kernel params without asking.
- Command timeouts 5 minutes or less.

## SHIPPED STATE

`llama_server/config/profiles/preset-flashnext-iq3xxs-vulkan-layer-82k.sh`
Vulkan preset menu entry 12. Build: `llama.cpp-qwen4exp/build-vulkan` at
`aa1b06d0c` = PR #27880 (contains the arch PR #27742). Mainline cannot load
this model.

    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 llama-server \
      -m .../UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
      -c 81920 -ngl 99 --device Vulkan0,Vulkan1 \
      --split-mode layer --tensor-split 385,615 \
      -ub 1024 -b 2048 --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 \
      --load-mode mmap -ot 'per_layer_token_embd\.weight=CPU'

    pp 687.9 t/s   tg 42.5 @d0   tg 29.2 @d9060
    XTX 23,140 MiB (1,060 margin)   R9700 32,490 MiB (77 margin)

## TASK 1: the mmap change is UNVERIFIED. Verify it before anything else.

Every number above was measured with `--load-mode none`. The preset now says
`mmap` and that was shipped WITHOUT measurement, on the strength of the user's
independent replication showing mmap within noise of RAM-loaded:

    workload     RAM-loaded   mmap/SSD-backed
    prose            15.28         15.48 t/s
    code             16.93         16.88 t/s
    checklist        18.40         18.60 t/s

Confirm on THIS config: tg should still be ~42.5 at d0 and ~29.2 at d9060, and
RSS should be a few GiB rather than ~27. If mmap costs real throughput, the
answer is still not `none` -- it is to find a third option, because `none` on a
64 GB box is what caused the OOM.

## MEASURED FACTS. From instruments. Reuse, do not re-derive.

- `per_layer_token_embd.weight` is a FIXED tax: **27,465 MiB in UD-IQ4_XS and
  27,466 MiB in UD-IQ3_XXS**, IQ4_NL in both, ONE tensor, 320,001,536 x 160.
  Unsloth keeps it at IQ4_NL regardless of body quant. Only the body varies.
- Per-layer experts: `blk.N.ffn_{down,gate,up}_exps.weight`, 1,136 MiB/layer in
  IQ4_XS, 962 MiB/layer in IQ3_XXS.
- 48 blocks, `full_attention_interval 4` -> 12 attention layers, 36 GDN.
  `head_count_kv 2`, k/v length 256 -> 24 KiB/token f16, 12 KiB at q8_0.
- Usable VRAM: XTX ~22,700 (24,200 ceiling minus desktop), R9700 ~32,567.

### Quant choice is the single biggest lever -- worth 4.7x

    quant        total     - PLE  = main    zero-offload fit?   measured tg
    UD-IQ4_XS    87.3 GiB   26.8    60.5    no, 7 layers to CPU      9.14
    UD-Q3_K_XL   83.8 GiB   26.8    57.0    no, ~3 GiB over            -
    UD-IQ3_XXS   76.4 GiB   26.8    49.6    YES                      42.5

All three are downloaded and verified in `llama_server/models/Qwen3.8-Flash-Next/`.
Q3_K_XL was never benchmarked -- it cannot fit without offload, so it is
probably not worth the time.

### The R9700 is the binding constraint, not the XTX

It sits at 32,490 MiB in EVERY working config. Moving weight OFF the XTX does
not buy safety -- it overruns the R9700 and falls off a cliff:

    ts        XTX    margin   R9700  margin   pp
    38,62    23429      771   32491      76   699.1
    385,615  23140     1060   32490      77   687.9   <- shipped
    36,64    22006     2194   32446     121    67.0   <- CLIFF
    35,65    21978     2222   32513      54    66.8   <- CLIFF

`--tensor-split` takes PROPORTIONS, not percentages, so sub-1% steps are legal.
But layer assignment is discrete over 48 layers, so most fine steps quantise to
the same placement (370,630 and 375,625 gave byte-identical VRAM).

### Cliffs are abrupt and total, not gradual

Context (ub 512):  81920 -> 42.35 | 90112 -> 42.42 | **94208 -> 11.77** | 98304 -> 11.78
Ubatch (ctx 81920): 512 -> pp 577 | 1024 -> pp 695 | **2048 -> pp 69** | 4096 -> pp 71

KV type: f16 is faster where it fits (43.50 vs 42.77 at ctx 65536) but falls off
the cliff at 81920 (11.82). q8_0 is what buys the context.

`--flash-attn on` is worthless alone (6.49 -> 6.47) and +39% paired with
quantised KV. Never ship one without the other.

## MEASURE PREFILL WITH A REAL PROMPT

A 20-token prompt reports pp ~50 t/s on this config. That is startup overhead.
The real figure is ~690. Use the 9,060-token prompt at
`/tmp/.../scratchpad/bigprompt.txt`, or rebuild it by concatenating
`llama_server/src/benchmarks/prompts/{agentic_toolcalls,mix_000,mix_050,ilv_025,ilv_075,mix_100}.txt`.
The prior agent reported a bogus "pp 50.9" in a commit message because of this.

## DEAD ENDS. Do not spend time here.

- **MTP: loads, but is a 2x NET LOSS.** Mainline aborts with "tensor
  'blk.0.hc_attn_norm.weight' not found"; the MTP graph is only built by
  `LaurentZuijdwijk/llama.cpp` branch `qwen4exp/mtp-fix` (337c8bb58), which also
  needs `LLAMA_ATTN_ROT_DISABLE=1`. On that fork, ctx 32768 / ts 42,58 / draft on
  Vulkan1: MTP off 40.98, n1 19.73, n2 19.19, n3 21.84 (99% acceptance),
  adaptive 19.02, n5 5.94. The draft forward costs more than the accepted tokens
  save once the target is at ~43 t/s. Draft PLACEMENT matters a lot (Vulkan1
  21.84 vs Vulkan0 8.14) but never enough to break even.
- **`-sm tensor` is structurally impossible.** Aborts with "multi buffers are not
  supported by the meta backend" whenever any tensor is CPU-resident, and this
  model cannot fit without the PLE on CPU.
- **`-sm row`**: "device Vulkan0 does not support split buffers" on RADV, same on
  ROCm0. Unimplemented on both.
- **ROCm backend**: always demands exactly 34,621.23 MiB on device 1 (a 32,624
  MiB card) regardless of `--tensor-split` or expert offload. The number never
  moves. Wants ~58,680 MiB vs 55,267 available.
- **fork `qwen4exp/ple-sidecar`** (WIP): pp 47.5 / tg 34.62, slower.
- **fork `qwen4exp/mtp-fix` with MTP off**: pp 22.8 / tg 40.98, much worse prefill.

## llama.cpp flag traps

- `-ncmoe` is SILENTLY IGNORED when `-ot` is also passed. Same override list,
  `-ot` wins, no warning.
- Repeated `-ot` flags DO NOT accumulate -- only the first applies. Use one `-ot`
  with comma-separated `pattern=buffer` pairs.
- `-v` logs `tensor X (NNN MiB type) buffer type overridden to Y` per tensor.
  VERIFY every override actually applied before trusting a run.
- `--device` is MANDATORY on Vulkan. This box enumerates THREE Vulkan devices;
  Vulkan2 is the CPU (RADV RAPHAEL_MENDOCINO) advertising 32 GB free, and auto
  selection silently includes it and destroys throughput. An entire round of
  this session's measurements was invalidated by this.
- llama-bench `-ts`/`-dev` use SLASHES; the server uses COMMAS.

## OPEN LEADS

1. **Verify mmap** (Task 1 above). Blocking.
2. **Cherry-pick `vulkan/iq3s-mmv-register-spill`** (d19f92093) onto the
   qwen4exp tree. It targets the IQ3_S mat-vec path this quant leans on. The
   branch itself has no qwen4exp and cannot load the model, so it must be
   cherry-picked. Never evaluated. Most promising untried item.
3. **GDN kernels on Vulkan.** 36 of 48 layers are Gated DeltaNet. Never isolated.
   Compare per-op timings against the CUDA path in the PR threads.
4. **PLE async prefetch.** Qwen's reference overlaps the n-gram lookup with
   compute. Determine whether llama.cpp does this at all; if not it is real work
   but possibly large.
5. PR #27739 (JJJYmmm) is an independent qwen4exp implementation, never
   evaluated here.

## FILES

    llama_server/records/2026-08-29-flashnext/RESULTS.md   all measurements, 3 rounds
    llama_server/records/2026-08-29-rocm10/RESULTS.md      27B backend comparison
    llama.cpp-qwen4exp/build-vulkan                        SHIPPED build (PR #27880)
    llama.cpp-q4e-mtp/{build-vulkan,build-ple}             MTP + ple-sidecar forks
    models/Qwen3.8-Flash-Next/{UD-IQ3_XXS,UD-IQ4_XS,UD-Q3_K_XL,MTP-Q4_K_M}
