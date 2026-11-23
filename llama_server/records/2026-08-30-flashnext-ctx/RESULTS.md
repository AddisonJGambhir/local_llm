# 2026-08-30 Flash-Next context campaign: RESULTS

All runs: Qwen3.8-Flash-Next, Vulkan, R9700 (card0, 32,590 MiB) + 7900 XTX
(card1, 24,512 MiB total, desktop-driven, ceiling 24,300 MiB settled per user).
Build `llama.cpp-qwen4exp/build-vulkan` at `aa1b06d0c` unless noted. Q8_0 KV,
FA on, ts 385,615 (Vulkan0 = XTX hosts layers 0-17, Vulkan1 = R9700 hosts
18-47), PLE on CPU mmap in every config. Attention layers: {3,7,...,47}; XTX
hosts 4 of 12, so KV+mask grows ~10.75 KiB/token on card0 and ~6 on card1.

## Ladder at Q8 KV, IQ3_XXS body

| ctx | ub | CPU expert layers | tg shallow | tg @d9060 | pp @d9060 | card0 | card1 |
|-----|----|-------------------|-----------|-----------|-----------|-------|-------|
| 81,920 | 1024 | 0 | 42.5 | 29.2 | 688 | 32,276 (314) | 23,596 (604) |
| 94,208 | 1024 | 0 | 40.35 | 29.27* | 633* | 32,517 (73) | 23,928 (272) |
| 131,072 | 256 | 0 | 40.50/40.83 | 28.52 | 425 | 32,524 (66) | 23,254 (1,046) |
| 262,144 | 256 | blk.2,30,40 | 34.1/35.7 | 22.68 | 292 | 32,479 (111) | 23,573 (727) |

*from the 92K full harness run /tmp/flashnext-q8-92k-full.

Cliff data (shallow probes): ub 1024 dies between 94,208 and 98,304. ub 256
dies between 131,072 and ~140K (96K/98,304 ub 512 = 12.3 t/s cliff; 98,304
at ub 512 = 40.28 OK). 256K requires ub 256 or smaller (the KQ mask is
ctx x ubatch x 4 bytes per attention-hosting card).

## Expert offload costs (measured, 128K, ub 256)

| variant | tg | cost |
|---------|----|------|
| no offload | 40.32 | - |
| blk.30 experts -> CPU (962 MiB) | 37.41 | -7.2% |
| blk.30 experts -> XTX (Vulkan0) | 36.87 | -8.5% |
| all of blk.46 -> XTX | 27.36 | -32% |

Inter-card redistribution costs as much as CPU offload; whole-layer moves are
far worse (two graph crossings per token). Conclusion: keep native layer
placement, offload only the overflow to CPU. The 128K no-offload config is
already the near-zero-margin optimum: card0 has only 66 MiB free, and the next
4K of context (+~43 MiB) does not fit.

- CPU expert layer frees 962 MiB on its host card (blk.2 is 1,138: down 450,
  gate/up 344 each; normal layers 962: down 450, gate 256, up 256).
- token_embd (486) and output (486) are CPU-resident BY DEFAULT in this setup;
  -ot on them changes nothing. No other non-expert tensor is worth offloading.
- ts rebalance does not free the R9700 (handoff-2 table confirmed by probes).

## MTP at long context (mtp-fix fork `llama.cpp-q4e-mtp`, WIP d8d84ea78)

128K, ub 256, 3 CPU layers, draft MTP-Q4_K_M (2.43 GiB, blk.48+heads) on
Vulkan1, LLAMA_ATTN_ROT_DISABLE=1:

| variant | tg |
|---------|----|
| fork, MTP off | 28.28 |
| fork, MTP n1 | 18.77 (24/31 accepted) |

Still a clear net loss. The draft reads its full 2.43 GiB every forward, so the
loss persists even with a 22-28 t/s target, and the draft displaces 2.43 GiB
of context VRAM. Recommend dropping MTP permanently on this hardware.

## KVarN (parked earlier this session)

K6/V5 at 16K on merged Bee build: shallow 28.59/29.07, d9060 tg 15.21,
pp 270.7 vs Q8 same build 40.81/41.68, 27.88, 637.1. Outputs coherent, tail
audit passed (exact suffix served from F16 stage; route log tail_stride=0 is
expected because no enlarged tail was requested).

## VRAM accounting (from GGUF + probes)

- Body IQ3_XXS 49.5 GiB; UD-Q2_K_XL body 46.7 GiB (PLE constant 26.8).
- KV q8_0 = 12 KiB/token (12 layers, 2 heads, 256 dim, K+V).
- idx_ indexer cache ~3 KiB/token (q8, 12 layers, 1 head, 128 dim).
- KQ mask = ctx x ubatch x 4 bytes per attention-hosting card.
- Measured card growth is ~10.75 KiB/token (card0) and ~6 (card1).

## UD-Q2_K_XL results (downloaded this session, byte-verified vs HF)

| ctx | ub | CPU layers | tg shallow | tg @d9060 | pp @d9060 | card0 | card1 |
|-----|----|-----------|-----------|-----------|-----------|-------|-------|
| 131,072 | 1024 | 0 | 42.08/42.47 | 28.80 | 654.0 | 32,146 (444) | 22,880 (1,420) |
| 262,144 | 256 | 0 | 41.39/42.28 | 29.44 | 440.4 | 32,525 (65) | 22,531 (1,769) |

Shipped preset reference (IQ3_XXS 82k ub1024): pp 688, tg 42.5, tg@d9060 29.2.
Q2_K_XL delivers shipped-preset speed at 1.6x context (128k) or 3.2x (256k).
Output quality on both standard probes (mmap/RSS essay, 9,060-token filler
refusal) is coherent and correct; d0 essay prose was cleaner than IQ3_XXS's.

An earlier "nondeterministic loader fit" concern was a harness bug: the full
harness silently ignored FLASHNEXT_MODEL and loaded IQ3_XXS at Q2-only
settings, which spills and cliffs (tg 11.6, pp 60). After fixing the harness,
Q2 fits 3/3 loads. The probe script always had the right model.

Preset written: config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh
(256k variant documented inside it).

## ROCm 10 on Qwen3.8-Flash-Next (2026-08-30, qwen4exp build-rocm)

Build: llama.cpp-qwen4exp/build-rocm at aa1b06d0c, AMDGPU_TARGETS=gfx1100;gfx1201,
ROCm core-10.0 toolchain, --device ROCm0,ROCm1 (ROCm0 = XTX, ROCm1 = R9700).

- `--load-mode mmap --tensor-read-lazy on` HANGS the loader: one core spins
  at 100%, ~20 GiB read in the first minute then zero I/O, no log progress
  for 7+ minutes at any ctx. Reproduced 3x (16K and 224K).
- `--load-mode none` LOADS and runs (the loader warning about CPU tensor
  overrides + mmap is the tell). Costs 26.8 GiB anonymous RAM for PLE.
- 16K quick probe: card0 29,516 / card1 21,032 MiB, tg 27.3.
- 128K, same winning config as Vulkan (ub 1024, blk.30 gate+up offload):
  placement similar to Vulkan (card0 32,124 / card1 23,428), but
  tg 26.6 vs Vulkan 42.1 = 37% DECODE LOSS.
- 224K same config: OOM during compute-buffer allocation on ROCm1
  ("allocating 3,189 MiB on device 1: cudaMalloc failed: out of memory").
  ROCm compute buffers are larger than Vulkan's; 224K does not fit.
- Conclusion: the ROCm 10 decode win measured on Qwen3.8-27B dense
  (+6-18% tg, records/2026-08-29-rocm10) does NOT transfer to qwen4exp.
  The sparse arch (GDN/QSA/PLE) is far slower on HIP in this build.
- Not completed before handoff: -sm tensor and -sm row probes on ROCm 16K
  (user aborted). From the 27B records: -sm tensor worked on ROCm 10 for a
  model with NO CPU-resident tensors; -sm row failed to load entirely.
  Expect -sm tensor on Flash-Next to abort on the meta-backend error because
  PLE is CPU-resident. Verify, do not assume.

## Community research (2026-08-30)

From unsloth/Qwen3.8-Flash-Next-GGUF discussion #3 and #22:

- RTX 6000 PRO 96GB (Q4_K_XL, 262K ctx, q8 KV): 59.7-70 t/s tg, pp 1796-2559.
- 2x RTX 4090D 48GB: 44.7 t/s, pp 1797. Single RTX 6000 Pro: ~70 t/s.
- Strix Halo 128GB: 20-23 t/s. M1 Ultra 128G (IQ1_S): ~20 t/s.
- Dual 16GB cards (5070 Ti + 5060 Ti, 64GB RAM, IQ3_XXS, 32K ctx): 23.3 t/s.
  This is the setup behind the remembered "5070 Ti" claim; it is 23, not 40.
- CPU-heavy offload rigs: 15-19 t/s.
- OUR 40-42 t/s AT 128-256K IS AMONG THE TOP REPORTED FOR COMPARABLE HARDWARE.
- Deep-context decode decay is universal (CUDA user: 60 t/s -> 30 after
  128K ctx at 256K window); the QSA indexer scores every populated cell.
- The 26.8 GiB PLE table is host-RAM for everyone; no rig keeps it in VRAM.
- g4bw (2x CMP 170HX): model leaves ~33 GB in host RAM + 1 core pinned at
  100% during decode on PR 27742 -- "some GPU offload code hasn't been
  written yet". Known upstream issue to watch.

## Deep prefill decay (measured, Q2 224K ub1024 build)

Progress log of a 199,320-token prefill (aborted at 52% by a harness bug,
the curl max-time was 900s vs a ~30 min prefill; fixed to 3600s):

    73,728 tok: avg 193.5 t/s | 82K: 179 | 98K: 131.7 | 102K: 121.6 | 104K: 116.2

Average is cumulative; instantaneous rate at 100K depth is ~110 t/s and
falling. Compare 613 t/s at 9K depth. Full 199K prefill projects to ~30 min.
Model-architecture cost (dense indexer scoring), backend-independent; CUDA
users see the same shape. RERUN NEEDED for the final decode-at-depth number.

## Files

- probe script: /tmp/flashnext_ctx_probe.sh (env-configurable BIN/MODEL/CTX/UB/B/OT/EXTRA/LOADMODE/DEVS)
- full harness: /tmp/flashnext_mmap_verify.sh (env-configurable MODEL/UB/B/OT/PROMPT/NPREDICT)
- 92K full: /tmp/flashnext-q8-92k-full
- 128K full: /tmp/flashnext-q8-128k-full
- 256K full: /tmp/flashnext-q8-256k-full
- Q2 224K full runs: /tmp/flashnext-q2-224k-full, /tmp/flashnext-q2-224k-ub1024-full
- Q2 128K full: /tmp/flashnext-q2-128k-full2
- Q2 256K full: /tmp/flashnext-q2-256k-full3
- probes p0-p14, q2a-q2i: /tmp/flashnext-ctx-probe-*
- ROCm: /tmp/flashnext-rocm-{layer,quick,nommap,layer224,layer128}
- deep 200K partial: /tmp/flashnext-q2-224k-deep200k (server.log has decay curve)
- 200K prompt: /tmp/bigprompt-200k.txt (22x bigprompt.txt, ~199K tokens)
- Q2 download log: /tmp/q2kxl-download.log
