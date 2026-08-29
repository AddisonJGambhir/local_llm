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

## Cross-backend, sm=tensor, d0 only

| backend                          | pp2048 | tg256 |
|----------------------------------|--------|-------|
| Vulkan llama.cpp-tp-adaptive     | 391.74 | 20.97 |
| ROCm 7.1                         | 413.46 | 28.29 |
| ROCm 10                          | 427.62 | 29.95 |

ROCm 10 is +42.8% on tg over the shipped Vulkan build at d0. CAVEAT: the Vulkan
arm is commit 7872b3e3e, not 030ebb558, so this comparison carries a commit
delta the ROCm-vs-ROCm one does not. The Vulkan d65536 rows did not complete
and have not been diagnosed. Not yet acted on.

## Negative / dead ends
- `-sm row` CANNOT LOAD this model on either ROCm version ("failed to load
  model"). Row split is not an option for Qwen3.8-27B here. Do not retest.
- llama-bench tensor-split prefill remains the known artifact (413 vs 1150 on
  layer). Absolute pp numbers here are not real-server numbers.

## Shipped
All 8 ROCm presets repointed to llama.cpp-rocm10. The 4 tensor-decode presets
inherit via `source` and needed no edit. Vulkan presets untouched.
Server smoke-tested on port 18234: /health ok, completion returned, 33.8 t/s.
