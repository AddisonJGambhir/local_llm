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
