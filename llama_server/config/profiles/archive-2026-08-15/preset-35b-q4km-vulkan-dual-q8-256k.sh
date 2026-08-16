#!/usr/bin/env bash
# Dual-GPU Vulkan preset: 35B A3B MoE Q4_K_M, MTP, q8_0 KV, native 256k context.
# Vulkan0 is the RX 7900 XTX and Vulkan1 is the Radeon AI PRO R9700.
# A 70/30 layer split favors the faster XTX while preserving display headroom.

source "$LLAMA_SERVER_ROOT/config/profiles/preset-35b-q4km-vulkan-q8-96k.sh"

PROFILE_NAME="preset-35b-q4km-mtp-vulkan-dual-q8-256k"
LLAMA_DEVICE="Vulkan0,Vulkan1"
LLAMA_CONTEXT=262144
LLAMA_STARTUP_TIMEOUT=420
LLAMA_EXTRA_ARGS=(--split-mode layer --tensor-split 7,3)
