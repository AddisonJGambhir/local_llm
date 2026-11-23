#!/bin/bash
# Build radiance with image/video/audio decoding (FFmpeg) for the R9700 into src/build-ffmpeg.
# FFmpeg headers come from ffmpeg-dev/ (apt-get download'ed -dev debs, no sudo); runtime libs are the system's.
set -euo pipefail
R="$HOME/Desktop/local_llm/radiance"; cd "$R/src"
export ROCM_PATH=/opt/rocm
export PKG_CONFIG_PATH="$R/ffmpeg-dev/root/usr/lib/x86_64-linux-gnu/pkgconfig"
cmake -S . -B build-ffmpeg -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1201 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DRAD_WITH_HIP=ON -DRAD_WITH_FFMPEG=ON
cmake --build build-ffmpeg -j"$(nproc)"
echo; echo "radiance (ffmpeg) built: $(ls -la build-ffmpeg/bin/radiance)"
