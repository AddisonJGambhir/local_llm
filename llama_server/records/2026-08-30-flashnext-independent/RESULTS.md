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
