#!/usr/bin/env bash
# Environment required by EVERY hipfire invocation on this host.
#
# This is not optional setup convenience. hipfire compiles HIP kernels at
# install time AND lazily at runtime, and on this machine three separate things
# are wrong by default. A bare `hipfire` call fails.
#
# Sourced by bin/hipfirectl and by every profile.

# ---------------------------------------------------------------------------
# 1. Binary location. The installer does not put hipfire on PATH.
# ---------------------------------------------------------------------------
HIPFIRE_ROOT="${HIPFIRE_ROOT:-$HOME/.hipfire}"
export PATH="$HIPFIRE_ROOT/bin:$PATH"

# ---------------------------------------------------------------------------
# 2. ROCm device bitcode path.  *** LOAD-BEARING ***
#
# hipfire invokes hipcc with `--rocm-path=/usr`, but this host's ROCm is a
# distro/LLVM-21 install where the amdgcn device bitcode lives under the clang
# resource directory instead. Without this flag clang fails with:
#
#     cannot find ROCm device library; provide its path via '--rocm-path'
#     or '--rocm-device-lib-path', or pass '-nogpulib'
#
# and EVERY kernel fails to compile (observed: 45/45 on install, 44/44 on
# `hipfire profile`). llama.cpp never hits this because CMake's first-class HIP
# language support resolves the bitcode itself.
#
# Verify the path still exists before trusting a fresh failure:
#     ls /usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode/ocml.bc
# An LLVM major-version upgrade WILL move this.
# ---------------------------------------------------------------------------
HIPFIRE_DEVICE_LIB="${HIPFIRE_DEVICE_LIB:-/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode}"
export HIPFIRE_HIPCC_EXTRA_FLAGS="--rocm-device-lib-path=$HIPFIRE_DEVICE_LIB"

# ---------------------------------------------------------------------------
# 3. GPU selection + target arch.  *** LOAD-BEARING ***
#
# This box has three AMD devices:
#     HIP 0 = RX 7900 XTX      gfx1100   24560 MiB   ALSO DRIVES THE DESKTOP
#     HIP 1 = AI PRO R9700     gfx1201   32624 MiB
#     HIP 2 = iGPU (Raphael)   gfx1036
#
# hipfire builds kernels for ONE architecture. The installer refuses to start
# without being told which:
#     hipfire: multiple GPU architectures found: gfx1100, gfx1201, gfx1036
#
# Worse, `--gpu-arch` alone is not enough: it compiled for gfx1201 but then
# validated the module against HIP device 0 (the XTX) and died with
#     hipModuleLoad: device kernel image is invalid  (HIP 200)
# Pinning visibility is what makes the compile target and the validation target
# agree.
#
# We target the R9700 because it is the card with room (32 GiB) and it does not
# drive the desktop. To retarget the XTX you must set BOTH of these to 0/gfx1100
# AND re-run `hipfirectl setup` -- the kernel cache is per-arch.
# ---------------------------------------------------------------------------
export HIP_VISIBLE_DEVICES="${HIPFIRE_GPU:-1}"
export ROCR_VISIBLE_DEVICES="${HIPFIRE_GPU:-1}"
export HIPFIRE_TARGET_ARCH="${HIPFIRE_TARGET_ARCH:-gfx1201}"

# ---------------------------------------------------------------------------
# 4. Service endpoint. hipfire's own default is 0.0.0.0:11435.
# We bind loopback only, and stay off 1234 which llama-server owns.
# The two engines can both be *running* without a port clash; they cannot both
# hold VRAM. See docs/NOTES.md.
# ---------------------------------------------------------------------------
HIPFIRE_HOST="${HIPFIRE_HOST:-127.0.0.1}"
HIPFIRE_PORT="${HIPFIRE_PORT:-11435}"

# hipfire's own log when started with --detach
HIPFIRE_LOG="${HIPFIRE_LOG:-$HIPFIRE_ROOT/serve.log}"
