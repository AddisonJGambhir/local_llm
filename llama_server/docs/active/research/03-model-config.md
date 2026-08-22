# Model & config notes (inline research, 2026-08-22)
Companion to ../DECODE_SPEED_REPORT.md.

## External decode datapoints for Qwen3.8-27B class (RDNA)
- JohnTDI (#21043 comment 18, 2026-08-16): Qwen3.8-27B Q8_0 + MTP, SINGLE R9700,
  Mesa 25.2.8, b0d9ceae1e, q4_0 KV, GPU driving desktop:
    baseline (n5, ub512, 2x32k slots) 59.2 t/s
    best (n7, ub288, parallel 3 x 16k slots, GGML_VK_ALLOW_GRAPHICS_QUEUE=1,
          spec-draft-ngl all) 64.95 t/s avg (range 53.5-79.3; structured text/code
          gets higher MTP acceptance)
  vs our best local: ROCm tensor Q6_K 58/42, MTP n3, 192k ctx = 53-54 t/s.
  Differences: quant (Q8 vs Q6), split (single vs tensor), ctx (16k/slot vs 192k),
  n_max (7 vs 3). n7 + short per-slot ctx explain most of it. Note their q4_0 KV
  (we need f16 for quality at 192k).
- PR #27210 (adaptive MTP n12): Qwen3.8-27B Q8_0 on 2x R9700 ROCm: +9.6% coding
  vs fixed n3, prose tie. Convergence: deeper ADAPTIVE MTP helps code workloads.
- #27342 (DFlash2 spec, 2026-08-18): evaluated Qwen3.8-27B Q4_K_M on Apple M5 Pro
  - different spec approach (draft model w/ conv), not applicable to our
  self-MTP setup, watch only.

## Quant / KV for decode (local + external)
- f16 KV everywhere (local rule; raises MTP acceptance). q8_0 KV slightly HARMFUL
  for speed (local null). JohnTDI used q4_0 KV at 16k slots - size tradeoff.
- Q6_K vs Q8_0: Q6 smaller = faster decode per token generally; our Q6 ROCm
  numbers (53-54) already beat the Vulkan Q8 (40.3) - quant + backend compound.
- MTP draft KV q8_0 (shipped) - fine.

## Context depth
- A12 probe queued (llama-bench -d 0/65536/131072/196608, exact commands in the
  report). Two no-P2P boxes report tensor decode collapsing to 9-11 t/s at
  60-160k KV on the PR build; our live 32-41 t/s is at shallow depth.
- #27090: Qwen3.8-27B silent crash at ~520K prefill (extreme, N/A).
- --context-shift OFF in our profile (hybrid model; shift breaks quality).

## Spec (MTP) settings state
- Shipped: n_max 3, p_min 0.0, q8_0 draft KV, n_min 0.
- p_min n8/n16 0.6/0.8: tested locally, rejected.
- n_max 4-6: tested on dense Vulkan (n4 73.5 vs n3 75.93 t/s) - n3 shipped.
- ADAPTIVE n12 (#27210 cherry-pick = A7): +9.6% coding on 2x R9700. Not yet
  tested on this box.
- GPU sampling of the MTP draft is BLOCKED under tensor split (F15, both
  backends, no upstream PR). A GPU-sampler patch under tensor = original work.
- #23476: imatrix collection can activate the MTP head - relevant if the user
  ever re-quantizes with an MTP-aware imatrix (quality, not speed).

## Multi-GPU MTP notes
- #27428: MTP halves PREFILL under layer split (100% pp penalty); tensor cost
  expected small (~3-8%, redundant draft head). A15 probe queued.
- #27460: RADV+MTP self-spec crash after kernel 7.1.3->7.1.7 (KMD class, #24492
  open, no master fix). Watch on kernel/Mesa upgrades (F22).
- #27489 (MTP compute-buffer reuse): N/A to dual-GPU tensor (single-CUDA path),
  saves ~1 GiB VRAM only.
- #25319 (async scheduler memcopies): watch only (two prior reverts).
