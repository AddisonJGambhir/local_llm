#!/bin/bash
# Build radiance v2.2.0 for the R9700 (gfx1201) with the ROCm in /opt/rocm, then run its fast tests on the R9700 only.
set -euo pipefail
cd "$HOME/Desktop/local_llm/radiance/src"
export ROCM_PATH=/opt/rocm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1201 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DRAD_WITH_HIP=ON
cmake --build build -j"$(nproc)"
ROCR_VISIBLE_DEVICES=1 ctest --test-dir build -L '^fast$' -j8 --output-on-failure
echo; echo "radiance built: $(ls -la build/bin/radiance)"
