# ROCm 10.0.0 vs ROCm 7.1 -- 2026-08-29

Model: Qwen3.8-27B-UD-Q6_K_XL.gguf (25,288,065,024 B, 27.32B params)
Source: identical commit 030ebb558 for both ROCm arms. `-ts 50/50`, f16 KV, no speculation.
llama-bench, r=3 (layer) / r=5 (tensor).

## Install
ROCm 10.0.0 from stable.repo.amd.com/rocm/core/packages/ubuntu2604, installed
ADDITIVELY at /opt/rocm/core-10.0. Distro ROCm 7.1 (/usr/lib) left intact.
Repo has 0 Conflicts/Replaces/Breaks lines and no DKMS packages, so the amdgpu
kernel driver was never touched. Packages: amdrocm10.0-gfx1100,
amdrocm10.0-gfx1201, amdrocm-core-dev10.0-gfx{1100,1201}, amdrocm-blas-dev10.0.

Build: llama.cpp-rocm10, HIPCXX=/opt/rocm/lib/llvm/bin/clang++ (clang 23),
CMAKE_BUILD_RPATH=/opt/rocm/lib so the binaries resolve ROCm 10 with NO env var.
Do NOT set CMAKE_BUILD_WITH_INSTALL_RPATH=ON -- it strips the build-tree RPATH
and the binaries then cannot find their own libllama-*-impl.so siblings.

## sm=tensor (production split mode)     <-- the result that matters

| test          | ROCm 7.1      | ROCm 10       | delta   |       |
|---------------|---------------|---------------|---------|-------|
| pp2048 d0     | 413.46 +-0.54 | 427.62 +-0.20 |  +3.42% | REAL  |
| tg256  d0     |  28.29 +-0.02 |  29.95 +-0.14 |  +5.86% | REAL  |
| pp2048 d65536 | 297.24 +-0.26 | 297.46 +-4.71 |  +0.07% | noise |
| tg256  d65536 |  25.70 +-0.11 |  27.23 +-0.11 |  +5.95% | REAL  |

Decode gains ~6% at both depths with non-overlapping error bars.

## sm=layer (control)

| test          | ROCm 7.1 | ROCm 10 | rocm10 build on 7.1 rt | verdict  |
|---------------|----------|---------|------------------------|----------|
| pp2048 d0     |  1150.52 | 1422.10 |                1387.69 | REAL     |
| tg256  d0     |    21.47 |   21.90 |                  21.52 | marginal |
| pp2048 d65536 |   478.58 |  524.62 |                 521.98 | REAL     |
| tg256  d65536 |    18.32 |   18.80 |                      - | marginal |

The layer-split PREFILL win is the COMPILER, not the runtime: the third arm
(ROCm 10 build, ROCm 7.1 runtime) captures 87% of the d0 gain and 94% of d65k.
Layer-split tg is flat -- clang contributes nothing to decode, as expected for a
bandwidth-bound path.

## Cross-backend, sm=tensor 50/50, ub1024/b2048, devices PINNED  (r=3)

| test          | Vulkan        | Vulkan+gfxq   | ROCm 10       | winner       |
|---------------|---------------|---------------|---------------|--------------|
| pp2048 d0     | 940.68 +-1.47 | 933.63 +-1.06 | 434.91 +-0.13 | Vulkan 2.2x  |
| tg256  d0     |  24.56 +-0.09 |  25.38 +-0.06 |  30.29 +-0.00 | ROCm +19%    |
| pp2048 d65536 | 578.56 +-2.44 | 584.99 +-1.80 | 304.60 +-1.65 | Vulkan 1.9x  |
| tg256  d65536 |  22.57 +-0.07 |  23.14 +-0.08 |  27.18 +-0.27 | ROCm +17%    |

VULKAN WINS PREFILL ~2x. ROCM WINS DECODE ~18%. Not a clean backend win either
way -- it depends whether the workload is prefill- or decode-dominated.

### CRITICAL: you MUST pass --device / -dev on Vulkan

This machine enumerates THREE Vulkan devices. Vulkan2 is the CPU
(RADV RAPHAEL_MENDOCINO) and reports 32,475 MiB "free", so llama.cpp's auto
device selection happily includes it in the tensor split. The result is a
catastrophic slowdown, an idle R9700, and VRAM readings that trip the ceiling
guard for no real reason.

  contaminated (auto):  pp2048 d0 = 391-402   tg256 d0 = 20.97-23.19
  pinned (Vulkan0/1):   pp2048 d0 = 940.68    tg256 d0 = 24.56

The production presets are SAFE -- they set LLAMA_DEVICE="Vulkan0,Vulkan1",
which llamactl passes as --device. Any ad-hoc llama-bench or llama-server
invocation must pass `-dev Vulkan0/Vulkan1` (SLASHES for llama-bench, commas
for the server) or the numbers are worthless. ROCm is unaffected: it enumerates
only 2 devices.

All Vulkan rows in raw/ marked -INVALID-igpu were collected before this was
found. The ROCm-vs-ROCm comparison above was never contaminated.

## GGML_VK_ALLOW_GRAPHICS_QUEUE=1

+3.3% tg at d0, +2.5% tg at d65536, prefill flat. Small but reproducible with
non-overlapping error bars. Upstream (discussion #21043) reports -8% on DENSE
models and +4.7% on MoE; measured POSITIVE here, so the dense warning does not
apply to this model. SHIPPED to the seven 27B Vulkan tensor presets. NOT applied
to the 35B-A3B or Muse Glimmer presets -- different models, unmeasured.

Also tested, both dead:
- `-ub 2048 -b 16384`: claimed +29% prefill upstream. Measured -2.29% with
  +-17.77 variance. Do not use.
- `RADV_PERFTEST=nogttspill`: -0.25% pp, +0.27% tg. Nil. Does not change the
  GTT reading either.

## Negative / dead ends
- `-sm row` CANNOT LOAD this model on either ROCm version ("failed to load
  model"). Row split is not an option for Qwen3.8-27B here. Do not retest.
- llama-bench tensor-split prefill remains the known artifact (413 vs 1150 on
  layer). Absolute pp numbers here are not real-server numbers.

## Shipped
All 8 ROCm presets repointed to llama.cpp-rocm10. The 4 tensor-decode presets
inherit via `source` and needed no edit. Vulkan presets untouched.
Server smoke-tested on port 18234: /health ok, completion returned, 33.8 t/s.

## Ceiling-guard false trip
An earlier ROCm d65536 run tripped the 24,200 MiB guard at 24,347 MiB. The
pinned re-run of the same configuration completed with the XTX at 19,079 MiB.
Treat single-sample guard trips as load spikes, not ceiling breaches -- measure
the SETTLED value.
