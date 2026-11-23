#!/usr/bin/env bash
# Serve AMD's Qwen3.8-27B Quark AWQ W4A16 checkpoint on the R9700 only.
# Build the required derived image first:
#   docker --host unix:///run/docker.sock build \
#     -f vllm/Dockerfile.quark-w4a16 \
#     -t local/vllm-rdna4-quark-w4a16:0.28.0-pr48606 vllm
set -euo pipefail

readonly docker_host='unix:///run/docker.sock'
readonly image='local/vllm-rdna4-quark-w4a16:0.28.0-pr48606'
readonly model='amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16'
readonly hf_cache='/home/addison-gambhir/.cache/huggingface'

exec docker --host "$docker_host" run --rm --name vllm-qwen38-int4-r9700 \
  --network host \
  --group-add video \
  --group-add render \
  --device /dev/kfd \
  --device /dev/dri \
  --ipc host \
  -e HIP_VISIBLE_DEVICES=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$hf_cache:/root/.cache/huggingface" \
  --entrypoint /usr/local/bin/vllm \
  "$image" serve "$model" \
    --served-model-name qwen38-int4-r9700 \
    --host 127.0.0.1 \
    --port 8011 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.88 \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --attention-backend TRITON_ATTN \
    --enable-prefix-caching \
    --reasoning-parser qwen3
