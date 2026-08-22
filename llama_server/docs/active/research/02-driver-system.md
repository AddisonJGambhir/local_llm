# Driver & system layer notes (inline research, 2026-08-22)
Companion to ../DECODE_SPEED_REPORT.md. Raw dumps: raw-disc-21043.md (discussion #21043).

## PCIe topology (measured, sysfs current_link_speed/width)
R9700 (14:00.0): 02:02.1(internal) -> 05:00.0(1022:43f4, 16GT x4) -> 08:00.0(internal)
  -> 0c:08.0(1022:43f5, 16GT x4) -> 12:00.0(1002:1478, 16GT x4) -> 13:00.0(1002:1479, 32GT x16)
  -> 14:00.0(R9700, 32GT x16)
Effective end-to-end: Gen4 x4 = ~15.75 GB/s (docs had said 7.9 - wrong by 2x).
The last segment is Gen5 x16: a Gen5 x16 route EXISTS on the board; the chipset
bridge is what constrains. Moving the R9700 to a different slot (if a Gen5 x16
CPU slot is free) could double tensor prefill transfer. USER-HW item, untested.
XTX (03:00.0): Gen4 x16 ~31.5 GB/s directly on the CPU root (docs said "PCIe 5.0" -
wrong; 9950X3D is Gen4-only).
Decode implication: none (10 KB activations per step). Prefill implication: tensor
prefill ceiling ~2x higher than the analysis assumed.

## Power / clocks (measured)
- power_dpm_state: both cards currently "auto" (not forced).
  #21043: digitalscream forcing "high" = "huge difference to TG" (dual R9700);
  JohnTDI (Qwen3.8-27B Q8 MTP, single R9700): "auto remained faster than forcing
  high". Conflicting -> A13 sudo A/B, modest expectations.
- Under load: R9700 sclk 3254 MHz, mclk 1258 (5/5 max). XTX table top 2482. No
  clock headroom.
- ASPM policy = default. Local test: performance = null. #21043 d-shehu: also no
  boost. Excluded. BIOS items (ASPM off, Global C-States off, Re-Size BAR on) are
  a user checklist.
- BAR sizes: R9700 BAR0 = 32 GiB, XTX BAR0 = 24 GiB (full Re-Size BAR).
  The 256MB-BAR ub deadband from #21043 does NOT apply.
- iowait ~34% at 15:37 traced to Docker Desktop qemu VM (virtio-blk on
  Docker.raw), not the LLM server.

## Mesa (26.0.3 installed)
26.0 -> 26.2 RADV scan (766 commits): two relevant:
- 937bd7e0 (2026-05-15) "clear the follower gang semaphore when a cmdbuf is
  reset" - multi-device construct; closest upstream match to the bug behind the
  PR #25051 decode-crawl workaround.
- 7fa1b487 (2026-06-30) "disable AMD_device_coherent_memory on gfx12 due to out
  of order behavior" - R9700 memory-ordering fix.
OP of #21043: NO Mesa/RADV knob moved 27B dense decode (29.3 t/s = 71% of the
640 GB/s ceiling); big Mesa wins are prefill-only. A4 = user-approved upgrade +
A/B, low decode expectation, correctness watch on tensor.

## ROCm (7.1.0-0ubuntu2 installed)
Ubuntu 26.04 ships only 7.1. Upstream: 7.2.0..7.2.4 (2026-01..05), tag 7.14.0
(2026-07). #21043 commenter on 7.2.1: "rocm is actually better" for Qwen3.6-27B
dense. A9 stays high-effort (AMD repo + kernel coupling) - only after build-side
items.

## RADV env / ICD levers from #21043 (cross-checked vs local null table)
- rm_kq=1 source patch: +0.8-1.1% RADV dense, +13% AMDVLK dense. FOLD INTO A1
  REBUILD (one line: uint32_t rm_kq = 1; in ggml-vulkan.cpp).
- GGML_VK_ALLOW_GRAPHICS_QUEUE=1: OP=zero on RADV, JohnTDI=small positive in MTP
  config. A13b at next restart, zero risk.
- AMDVLK: +11.7% dense-27B (with rm_kq=1, no gfx queue), MoE +3.7% (with gfx
  queue); gfx queue HURTS AMDVLK dense (-8%). A14 = install as second ICD,
  A/B via VK_ICD_FILENAMES. Risk: PR #25051 cross-device allreduce uncharted on
  AMDVLK.
- RADV_DEBUG=nocompute, RADV_PERFTEST=sam/bolist/localbos/dmashaders/nircache/
  hic/nogttspill, RADV_PROFILE_PSTATE: all zero on RDNA4 (OP). Closed (A5).
- glslc: our builds use /bin/glslc = shaderc 2026.1-1. The 2023.8 regression
  from JohnTDI does NOT apply.
- power_dpm_force: see above (A13).
- VectorAnvil direct-P2P allreduce (2x R9700: +9-12% tg): N/A, no P2P on this box.
