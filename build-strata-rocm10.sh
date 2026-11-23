#!/bin/bash
# Rebuild Strata 0.2.19 against ROCm 10 (/opt/rocm) instead of Ubuntu's HIP 7.1.1, so hipBLASLt is found
# (solution-table prompt dispatch). Builds into engine/build/hip-rocm10; the working build/hip-compile is untouched.
# Fully non-interactive. Run:  bash ~/Desktop/local_llm/build-strata-rocm10.sh 2>&1 | tee ~/Desktop/local_llm/build-strata-rocm10.log
set -euo pipefail
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
cd "$HOME/Desktop/strata-0219/engine"
cmake -S . -B build/hip-rocm10 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201" -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -Dhip_DIR=/opt/rocm/lib/cmake/hip -Dhipblaslt_DIR=/opt/rocm/lib/cmake/hipblaslt \
  -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_NATIVE_EXPERTS=ON 2>&1 | tee /tmp/strata-rocm10-configure.txt
grep -q "hipBLASLt not found" /tmp/strata-rocm10-configure.txt && { echo "FAIL: hipBLASLt still not found"; exit 1; }
cmake --build build/hip-rocm10 --target strata -j"$(nproc)"
ls -la build/hip-rocm10/strata
ldd build/hip-rocm10/strata | grep -E 'amdhip|hipblaslt'
echo "=== done: tell Claude 'built' ==="
