# Qwen3.8-Flash-Next independent measurements -- 2026-08-30

This series treats every prior handoff conclusion as unverified. Raw run files
live under `/tmp/fn-sweep/` for the current boot. Every server is protected by
the checked-in two-GPU guard with sustained ceilings of 31,600 MiB on sysfs
card0 (R9700) and 24,200 MiB on sysfs card1 (7900 XTX).

Hardware identity was read directly from sysfs, PCI IDs, and llama.cpp device
enumeration:

- sysfs card0 / PCI 14:00.0 / Vulkan1: AI PRO R9700, 32,624 MiB
- sysfs card1 / PCI 03:00.0 / Vulkan0: RX 7900 XTX, 24,560 MiB
- Vulkan2 is the CPU and is explicitly excluded

## Measurement protocol

`llama_server/bin/flashnext_sweep.sh` runs one warmup followed by three
measured deterministic generations of 256 tokens each. Results report the
median of the three measured samples plus their full min/max spread. The model
server is stopped after each configuration. This replaces the earlier
best-of-two, 96-token protocol, whose noise was too large for optimization
claims.

## Vulkan baseline: Q2_K_XL, Q8 KV, 49,152 context

Run directory: `/tmp/fn-sweep/baseline-q2-49k-4pass`

Configuration:

    build: llama.cpp-qwen4exp/build-vulkan
    model: UD-Q2_K_XL
    ctx: 49,152
    ubatch / batch: 1,024 / 2,048
    devices: Vulkan0,Vulkan1
    split mode / tensor split: layer / 385,615
    KV: q8_0 / q8_0
    PLE: CPU override, mmap, tensor-read-lazy
    fit: off

Raw decode samples:

    warmup: 43.6566 t/s
    measured: 44.2215, 44.8259, 45.2113 t/s
    median: 44.8259 t/s
    spread: 0.9897 t/s

Settled resources after the fourth request:

    R9700 (sysfs card0): 30,302 MiB
    XTX (sysfs card1):   22,129 MiB
    process RSS:          1,048,128 KiB

The guard recorded no breach. Output was coherent. This measurement does not
support treating 42.1 t/s as the machine's fixed shallow-decode ceiling; the
same general configuration at a safely smaller context produced a 44.83 t/s
median. It does not yet identify whether the difference is context, cache
state, thermals, software state, or ordinary run-to-run variance.

## Vulkan timing trace: the decode is not 89% host overhead

Run directory: `/tmp/fn-sweep/perf-concurrent-q2-49k`

This repeated the 49,152-context configuration with Vulkan timestamp logging
enabled and one warmup plus one measured 256-token generation. Timestamp
logging perturbed throughput, so its speed is diagnostic rather than a fair
benchmark:

    warmup:  36.5281 t/s
    measured: 37.4070 t/s

The measured request produced two complete device timing blocks:

    Vulkan0 / XTX / 18-layer portion:   2.32829 s / 256 =  9.095 ms/token
    Vulkan1 / R9700 / 30-layer portion: 3.85375 s / 256 = 15.054 ms/token
    summed GPU timestamp time:          6.18204 s / 256 = 24.149 ms/token

The device attribution is derived from the approximately 18:30 ratio in
per-operation invocation counts, which matches the configured layer split;
it is not copied from a prior report. The logger itself serializes or adds
timestamp-query overhead: the uninstrumented baseline median is 22.309
ms/token, while the instrumented request is 26.733 ms/token. Therefore the
timestamp total should not be used as an unperturbed latency number.

It is nevertheless decisive against the prior claim that only about 2.7
ms/token is real GPU work and approximately 89% of decode is host overhead.
Here, directly measured GPU query time is of the same order as the entire
decode wall time. The profiler slowed decode by about 17%, but no reasonable
interpretation of that perturbation reduces 24.149 ms/token to 2.7 ms/token.

The largest aggregate kernel families on both devices include the Q2 expert
mat-vec path, Q5/Q6 mat-vec operations, and high-count elementwise/fused
operations. On the 30-layer R9700 portion, representative 256-token totals
were:

    MUL:                                      271.2 ms
    IQ2_XS expert MUL_MAT_ID_VEC:             262.1 ms
    Q5_K m=10240 mat-vec bundle:              255.5 ms
    IQ4_NL expert MUL_MAT_ID_MUL:             217.5 ms
    Q6_K m=2560,k=6144 mat-vec bundle:        156.1 ms

These aggregates contain many invocations and are not single-kernel latency.
They show that both quantized expert mat-vec work and graph/kernel multiplicity
are live optimization targets. They do not support dismissing memory/quant
work on the basis of a host-overhead arithmetic estimate.

## IQ3_XXS guarded baseline: Q8 KV, 16,384 context

Run directory: `/tmp/fn-sweep/iq3xxs-independent-16k`

The downloaded `UD-IQ3_XXS` model was loaded from its first shard using the
same build, placement, Q8 KV, mmap PLE override, prompt, and four-pass protocol
as the Q2 baseline. Context was deliberately started at 16,384 because this
larger body leaves little R9700 margin.

Raw decode samples:

    warmup: 42.1638 t/s
    measured: 42.1791, 42.5570, 42.7968 t/s
    median: 42.5570 t/s
    spread: 0.6178 t/s

Settled resources after the fourth request:

    R9700 (sysfs card0): 31,225 MiB
    XTX (sysfs card1):   22,533 MiB
    process RSS:            986,584 KiB

The guard recorded no breach and output was coherent. The R9700 had only 375
MiB of margin under the conservative 31,600 MiB ceiling, so larger-context
steps must remain guarded and incremental. This establishes the model's real
small-context baseline but is not yet an apples-to-apples speed comparison
with the 49,152-context Q2 measurement.

### Rejected 49,152-context step at the same split

Run directory: `/tmp/fn-sweep/iq3xxs-independent-49k`

Raising only context to 49,152 was unsafe at the 385/615 layer split. The run
completed its warmup and two measured requests, then the guard observed the
same over-ceiling R9700 allocation three times at five-second intervals:

    11:36:55 R9700 32,079 MiB, XTX 23,179 MiB, breach 1/3
    11:37:00 R9700 32,079 MiB, XTX 23,215 MiB, breach 2/3
    11:37:05 R9700 32,079 MiB, XTX 23,189 MiB, breach 3/3; killed

The partial generation rates were 44.3229 and 44.5429 t/s, but this is not a
valid completed benchmark and those numbers must not be promoted as a median.
The failure establishes that context growth and/or its compute buffers added
about 854 MiB to the R9700 versus the 16K settled snapshot, exceeding the
31,600 MiB ceiling by 479 MiB. A larger context requires different placement,
not merely assuming the 16K margin scales gently.

### Layer-placement probe: 400/600 at 16,384 context

Run directory: `/tmp/fn-sweep/iq3xxs-split40-16k-probe`

A short one-warmup, one-sample probe changed only the layer split from 385/615
to 400/600. Settled placement changed from 31,225/22,533 MiB to:

    R9700 (sysfs card0): 30,158 MiB  (-1,067 MiB)
    XTX (sysfs card1):   23,619 MiB  (+1,086 MiB)

This confirms that the ratio crossed a discrete approximately one-layer
placement boundary. It gives the R9700 ample room but leaves only 581 MiB under
the guarded XTX ceiling before increasing context. The lone 128-token measured
sample was 44.2001 t/s; it is a placement probe, not a comparable throughput
result.

### Ubatch placement probe: 512 at 16,384 context

Run directory: `/tmp/fn-sweep/iq3xxs-split40-ub512-16k-probe`

At the 400/600 split, reducing ubatch from 1,024 to 512 changed settled usage
from 30,158/23,619 MiB to:

    R9700 (sysfs card0): 29,909 MiB  (-249 MiB)
    XTX (sysfs card1):   23,378 MiB  (-241 MiB)

The lone 128-token measured sample was 44.9364 t/s, again only a probe. The
approximately 240--250 MiB per-card saving is large enough to justify a full
49,152-context guarded run at this split; throughput and safety at that context
still require direct measurement.

## IQ3_XXS safe 49,152-context result

Run directory: `/tmp/fn-sweep/iq3xxs-split40-ub512-49k-full`

Configuration differs from the rejected 49K step only in the independently
measured safer placement levers: layer split 400/600 and ubatch 512.

Raw decode samples:

    warmup: 45.0332 t/s
    measured: 44.8967, 43.5371, 43.9283 t/s
    median: 43.9283 t/s
    spread: 1.3596 t/s

Settled resources after the fourth request:

    R9700 (sysfs card0): 30,477 MiB
    XTX (sysfs card1):   23,857 MiB
    process RSS:          1,093,252 KiB

The guard recorded no breach and output was coherent. This is the first
independently completed IQ3_XXS result at the same context as the Q2 baseline.
Its median is 2.0% below the earlier Q2 median, but that is not yet a quant-only
comparison because the Q2 run used split 385/615 and ubatch 1,024. A Q2 control
with this exact 400/600, ubatch-512 configuration is required.

### Exact Q2 control: body quant costs 2.5% decode

Run directory: `/tmp/fn-sweep/q2-control-split40-ub512-49k`

The Q2 control held context, split, ubatch, Q8 KV, mmap PLE placement, build,
prompt, prediction length, and pass count identical to the safe IQ3 run.

    Q2 warmup: 43.9373 t/s
    Q2 measured: 44.7901, 45.0315, 46.4274 t/s
    Q2 median: 45.0315 t/s
    Q2 spread: 1.6373 t/s
    Q2 settled R9700/XTX: 28,755 / 22,675 MiB
    Q2 RSS: 974,980 KiB

Against the IQ3 median of 43.9283 t/s, Q2 is 2.51% faster under the exact same
runtime configuration. It also uses 2,904 MiB less aggregate settled VRAM.
This isolates a modest but real body-quant cost; it replaces comparisons that
mixed quant, context, layer split, and ubatch.

### IQ3_S register-spill shader patch: no end-to-end win

Run directory: `/tmp/fn-sweep/iq3spill-split40-ub512-49k-full`

The alternate binary at `c98958507` contains the IQ3_S `NUM_COLS > 4` shader
change from `48a9046e2` plus test-only coverage, on the same `aa1b06d0c` base.
It was run with the exact safe IQ3 49K configuration.

    patched warmup: 42.5450 t/s
    patched measured: 43.0711, 43.7388, 43.9635 t/s
    patched median: 43.7388 t/s
    patched spread: 0.8924 t/s
    settled R9700/XTX: 30,477 / 23,857 MiB

The patched median is 0.43% below the unpatched 43.9283 t/s median, with
identical VRAM placement. That difference is smaller than run spread and does
not establish a regression, but it decisively fails to establish an
end-to-end improvement. Do not cherry-pick this patch for the current decode
configuration on the strength of its microbenchmark claim.

## Independent KVarN K6/V5 validation probe

Run directory: `/tmp/fn-sweep/kvarn65-independent-16k-probe`

The dirty BeeLlama/Qwen integration binary identifies itself as commit
`3746fb185-dirty`. It was started at 16,384 context with the IQ3_XXS body,
400/600 layer split, ubatch 512, mmap PLE override, and KVarN6 K / KVarN5 V.
This was one warmup plus one 128-token measured probe, not a benchmark series.

    warmup: 30.2055 t/s
    measured: 29.4223 t/s
    settled R9700/XTX: 29,997 / 23,518 MiB
    RSS: 256,104 KiB

Generated text was coherent. With `GGML_KVARN_DEBUG_ROUTES=1`, both Vulkan
devices independently reported K6/V5 `route=portable-native`, `entry=direct`,
and `fallback=none` for prompt and decode shapes. This establishes that the
probe exercised native KVarN attention rather than silent materialization.

It also establishes a large shallow-decode penalty versus the approximately
44 t/s standard-Q8 path. The debug route reported `tail_stride=0`,
`tail_history=0`, and `tail_current=0`; the fork describes an intrinsic exact
128-token suffix, but that suffix's stage-storage semantics have not yet been
independently audited. Coherent short output is necessary, not sufficient, for
a quality claim.

Source audit after the probe found that `tail_stride` describes only the
optional enlarged exact-tail arena. KVarN's always-present F16 stage buffers
hold the live incomplete 128-token group; the native shader selects those
stage rows before reading older completed groups from quantized records. Thus
`tail_stride=0` is consistent with the intrinsic lossless suffix when no
additional `--kv-tail-tokens` value is requested. This validates routing and
the intended storage mechanism, but it is not a task-level quality evaluation.

### KVarN 65,536-context placement probe

Run directory: `/tmp/fn-sweep/kvarn65-independent-64k-probe`

Changing only context from 16,384 to 65,536 produced:

    warmup: 31.0097 t/s
    one measured sample: 30.7229 t/s
    settled R9700/XTX: 30,701 / 24,123 MiB
    RSS: 261,200 KiB

The XTX had only 77 MiB left under the 24,200 MiB guard ceiling. Aggregate
VRAM grew by 1,309 MiB over 49,152 additional context tokens, or about 26.6
KiB/token. Extrapolation is not a launch authorization, but it does establish
a capacity impossibility: another 114,688 tokens to reach 180K would require
about 2,984 MiB at the measured slope, while only 976 MiB remains under the two
guard ceilings combined. Even the cards' absolute physical capacity provides
only about 2,360 MiB above this 64K allocation.

Therefore IQ3_XXS plus all experts in VRAM plus 180K context cannot fit on
these cards with this K6/V5 implementation. Reaching 180K requires freeing
other model allocation from VRAM, using a smaller body quant, or violating the
physical capacity; layer rebalancing alone cannot change the aggregate.

## Vulkan-focused Qwen branch: full stack regresses decode

Candidate binary: apepojken `qwen4exp-spec-mtp` at `843d5750`

The candidate branch does not implement `--tensor-read-lazy`, so its first
launch rejected that argument before allocating the model. The valid candidate
run and a fresh upstream control therefore both used ordinary mmap with the
lazy-row argument omitted. All other runtime settings were identical: Q2 body,
49,152 context, Q8_0 K/V, split 400/600, ubatch 512, PLE on CPU, one warmup,
three measured 256-token generations, and MTP disabled.

    candidate run: /tmp/fn-sweep/vulkan-mtpbranch-q2-nomtp-49k
    candidate warmup: 35.3449 t/s
    candidate measured: 37.2606, 39.3025, 40.3193 t/s
    candidate median: 39.3025 t/s
    candidate R9700/XTX: 28,476 / 22,475 MiB

    upstream run: /tmp/fn-sweep/upstream-q2-nolazy-49k
    upstream warmup: 46.4823 t/s
    upstream measured: 45.2012, 45.3681, 46.4159 t/s
    upstream median: 45.3681 t/s
    upstream R9700/XTX: 28,755 / 22,715 MiB

The bundled Vulkan/Qwen/MTP branch is 13.37% slower by median in this exact
no-MTP control. Its rising candidate passes make the exact regression size
somewhat warmup-sensitive, but even its best measured pass remains 10.7% below
the upstream median. Do not adopt the full branch. Any promising Vulkan fusion
commits must be isolated on the known-good `aa1b06d0c` base.

### Isolated Vulkan fusions: +7.6% decode

Run directory: `/tmp/fn-sweep/vulkan-fusions-q2-49k`

The two narrow Vulkan changes from the rejected branch were applied without a
commit to an isolated `aa1b06d0c` worktree: mat-vec elementwise epilog fusion
from `3c82e945` and fused REPEAT+MUL+ADD from `a4c5df63`. Only the main-model
hyper-connection graph wiring was ported; the MTP graph was excluded. Runtime
configuration exactly matched the earlier Q2 control, including lazy mmap.

    fusion warmup: 48.6132 t/s
    fusion measured: 48.0208, 48.6790, 48.4513 t/s
    fusion median: 48.4513 t/s
    fusion spread: 0.6582 t/s
    settled R9700/XTX: 28,755 / 22,688 MiB
    RSS: 1,404,068 KiB

This is 7.59% above the exact unpatched 45.0315 t/s median with effectively
identical VRAM placement. Generated text remained coherent. This establishes a
real decode win from the isolated fusion pair, but does not yet distinguish the
two patches from one another.

### One layer shifted to XTX: 49.16 t/s, still below target

Run directory: `/tmp/fn-sweep/vulkan-fusions-q2-split425-49k`

Changing only tensor split from 400/600 to 425/575 shifted one discrete model
layer from the R9700 to the XTX. A 128-token placement probe produced one 49.82
t/s sample; the required full run did not sustain that number.

    full warmup: 49.4520 t/s
    full measured: 48.9480, 49.4649, 49.1566 t/s
    full median: 49.1566 t/s
    full spread: 0.5168 t/s
    settled R9700/XTX: 27,776 / 23,656 MiB
    RSS: 1,384,752 KiB

The placement adds 1.46% over the fused 400/600 median and 9.16% over the
original unpatched 400/600 control. It does not establish 50 t/s. The XTX has
only 544 MiB of guard margin at 49K, so this split cannot be assumed usable at
high context without reallocating KV or model layers.

### Sampler-chain probes: no route to 50 t/s

At the fused 425/575 placement, replacing the request's greedy top-k path with
`top_p;min_p;temperature` forced filtering over the full vocabulary and fell
to a 46.3281 t/s median. `top_k;temperature` measured 49.0762 t/s, effectively
equal to and slightly below the 49.1566 default-chain median. Sampler changes
do not close the remaining shallow-decode gap.

### Real 8,948-token A/B: 1,044 pp and 38.59 deep tg

Patched run: `/tmp/fn-sweep/vulkan-fusions-skinny-radix-q2-depth9k-ignoreeos`

Control run: `/tmp/fn-sweep/upstream-q2-depth9k-ignoreeos-control`

The patched build adds the validated Vulkan fusion pair, the main-model parts
of `b6d995d5` (copy-free stream pooling and skinny inject mat-vec for prefill),
and only the Vulkan backend portion of radix TOP_K `233c0d9c`. The regressive
branch's QSA gather/cache rewrites and MTP graph are absent. Both runs used the
same 35,000-character slice of `/tmp/bigprompt-200k.txt`, producing 8,948
prompt tokens, Q2 body, Q8_0 K/V, context 16,384, ubatch 1,024, split 425/575,
128 forced decode tokens, mmap lazy loading, and identical greedy sampling.

    patched: 1044.0525 pp, 38.5933 tg at depth, R9700/XTX 27,492/23,579 MiB
    upstream: 725.9201 pp, 29.9749 tg at depth, R9700/XTX 27,470/23,479 MiB

The combined stack improves real-prompt prefill by 43.83% and decode at 9K
live depth by 28.75% in this exact one-pass A/B. Both generated all 128 tokens;
the patched continuation was coherent and recognized the synthetic repetition,
while the control fell into repetitive continuation. This establishes a large
stack-level win, not yet attribution among the three changes. The patched
1,044 pp result clears the 600 pp target at this measured depth.
