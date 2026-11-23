#!/bin/bash
# Builds the two newer engines side by side with the working ones (nothing existing is overwritten):
#   1. Strata v0.2.19   -> ~/Desktop/strata-0219   (engine/build/hip-compile/strata + terminal/.venv)
#   2. hipfire beta     -> ~/.hipfire-beta         (models shared with ~/.hipfire/models via symlink)
# Run:  bash ~/Desktop/local_llm/build-new-engines.sh 2>&1 | tee ~/Desktop/local_llm/build-new-engines.log
set -euo pipefail
export ROCM_PATH=/opt/rocm

echo "=== [1/2] Strata v0.2.19 engine (same flags as the 0.2.13.1 build) ==="
cd "$HOME/Desktop/strata-0219/engine"
cmake --preset hip-compile \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DSTRATA_PORTABLE=OFF \
  -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201"
cmake --build --preset hip-compile
ls -la build/hip-compile/strata
cd "$HOME/Desktop/strata-0219/terminal"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "Strata 0.2.19: OK"

echo "=== [2/2] hipfire beta into ~/.hipfire-beta ==="
mkdir -p "$HOME/.hipfire-beta"
[ -e "$HOME/.hipfire-beta/models" ] || ln -s "$HOME/.hipfire/models" "$HOME/.hipfire-beta/models"
HIPFIRE_HOME="$HOME/.hipfire-beta" bash "$HOME/.hipfire-beta/src/scripts/install.sh" --yes --gpu-arch gfx1100
ls -la "$HOME/.hipfire-beta/bin/hipfire"
echo "hipfire beta: OK"
echo "=== done: tell Claude 'built' ==="
