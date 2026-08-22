> **Historical record - not current operating guidance.**  
> Current instructions: [How to optimize](../../docs/HOW_TO_OPTIMIZE.md) | [Setup](../../docs/SETUP.md)

# RCCL on this box: tested properly, does not work (2026-08-21)

## Summary
RCCL cannot perform a two-GPU AllReduce on this machine. This was verified with
AMD's own test suite, with llama.cpp entirely out of the picture. Keep
`GGML_HIP_RCCL=OFF`.

## What was wrong with earlier attempts
Previous "tried RCCL on and off, zero difference" tests were null tests. The
stock build had `GGML_HIP_RCCL:BOOL=OFF` in CMakeCache and was never linked
against librccl. Nothing was being toggled.

`librccl1` / `librccl-dev` 7.1.0-0ubuntu2 WERE installed the whole time, at
/usr/lib/x86_64-linux-gnu/. An earlier check looked in /opt/rocm/lib/, which does
not exist here because ROCm comes from Ubuntu's repos, not AMD's.
Actual RCCL version: 2.72.5.

## Build
    cmake -S . -B build-rccl -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
      -DAMDGPU_TARGETS="gfx1100;gfx1201" -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=ON
Builds clean in ~90s with ccache. Verified linked: librccl.so.1 => /usr/lib/x86_64-linux-gnu/librccl.so.1

## Results (Qwen3.8-27B-Q6_K)
| config | result |
| --- | --- |
| RCCL build, layer split, 2 GPUs | 1208.27 pp -- WORKS |
| RCCL build, tensor split, 1 GPU (ROCm1) | 675.62 pp -- WORKS |
| RCCL build, tensor split, 2 GPUs | SEGFAULT (exit 139) |
| RCCL build, tensor, 2 GPUs, NCCL_P2P_DISABLE=1 | SEGFAULT (exit 139) |

It only fails when a cross-device collective actually fires.

## NCCL_P2P_DISABLE=1 does NOT fix it
With P2P off the transport genuinely changes and it still dies:
    Check P2P Type isAllDirectP2p 0 directMode 1      <- P2P off, confirmed
    Channel 00 : 0[3000] -> 1[14000] via SHM/direct   <- host shared memory
    Connected all rings
    <segfault>
So the crash is NOT specific to the direct-P2P path, contrary to a common
suggestion. It crashes on the host SHM transport too.

Without the flag, RCCL misdetects P2P as available:
    Check P2P Type isAllDirectP2p 1 directMode 1
The two GPUs are on DIFFERENT root ports (01:00.0 vs 12:00.0) with no P2P path.

## AMD's own test suite crashes too (the decisive test)
    UT_MIN_GPUS=2 /usr/libexec/rocm/librccl1-tests/rccl-UnitTests --gtest_filter='AllReduce.*'

    [ RUN      ] AllReduce.OutOfPlace
    [ INFO     ] SP 2 ranks  AllReduce (OP,GM,NL)  ncclFloat32  sum  [393216 elements]
    [ ERROR    ] Child 0 pipe closed unexpectedly
    exit=1

Single-GPU AllReduce (OutOfPlace/InPlace/ManagedMem) PASSES. Two GPUs dies on the
first collective. llama.cpp is not implicated; RCCL itself is broken here.

## Why it would not have helped anyway
There is no P2P route between these cards, so a working RCCL would transport over
host memory -- exactly what llama.cpp's internal allreduce already does. The two
existing implementations already measure identically:
    GGML_CUDA_ALLREDUCE=internal  430.30 pp
    GGML_CUDA_ALLREDUCE=none      429.57 pp   (meta butterfly)
That is the transport setting the ceiling, not the algorithm.

## Untested
`iommu=pt` (IOMMU is in translating mode; not in /proc/cmdline). Expected to fail
for the same reason -- it addresses DMA remapping on the P2P path, and this dies
on SHM too. Requires a GRUB edit + reboot.
Upgrading to AMD's ROCm 7.14 (repo.radeon.com) is the only credible fix path.
The build-rccl/ tree is kept for a 90-second retest if that ever happens.
