# Handoff 4: Q2_K_XL gives 128K-256K context at shipped-preset speed; KVarN and MTP closed

Supersedes `2026-08-29-FLASHNEXT-HANDOFF-3.md`. The user's objective changed
from "more context than 82K" to "128K+ context at 40-50 tg and 500+ pp, with
good KV quality". That objective is now met and shipped as a preset.

## SHIPPED THIS SESSION

`llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh`

    UD-Q2_K_XL body, ctx 131072, ub 1024, b 2048, q8_0 KV, FA, ts 385,615,
    PLE on CPU mmap, --tensor-read-lazy on, --device Vulkan0,Vulkan1

    pp 654.0 t/s (9,060-token prompt), tg 42.1/42.5 shallow, tg 28.8 @d9060
    R9700 32,146 MiB settled (444 spare), XTX 22,880 (1,420 under 24,300)

256K variant (validated, same model, ub 256, ctx 262144): pp 440, tg 41.4/42.3,
28.8 @d9060, R9700 32,525 (65 spare). 262,144 is the trained maximum.

All numbers this session are from `llama.cpp-qwen4exp/build-vulkan` at
`aa1b06d0c`, measured with /tmp/flashnext_ctx_probe.sh and the generalized
/tmp/flashnext_mmap_verify.sh (both now take FLASHNEXT_MODEL/UB/OT/etc).

## WHY Q2_K_XL IS THE ANSWER

The PLE table is a fixed 26.8 GiB in every quant, so only the body varies.
Q2_K_XL body 46.7 GiB vs IQ3_XXS 49.5: the missing 2.8 GiB is almost exactly
what the KV cache plus the ctx-scaled KQ mask need to go from 82K to 128K at
full ubatch. Body-quant cost is the only price, and on the standard probes
(mmap/RSS essay, filler-refusal at depth 9,060) the outputs are coherent and
correct.

## PLACEMENT LAWS (measured, do not re-derive)

- The R9700 is always the binding constraint. ts rebalancing does NOT move
  weight off it; the discrete 48-layer boundary either fits or spills to CPU.
- Each expert layer on CPU frees 962 MiB on its host card and costs ~7% tg
  (40.3 -> 37.4, one layer). blk.2's set is 1,138 MiB (gate/up are 344 each;
  normal layers: down 450, gate 256, up 256).
- Cross-card expert moves via -ot ...=Vulkan0 cost 8.5% (36.9). Whole-layer
  moves cost 32% (27.4). Both are worse than or equal to CPU offload, so
  never "rebalance" between cards; offload overflow to CPU only.
- token_embd (486 MiB) and output (486 MiB) are CPU-resident BY DEFAULT;
  -ot on them changes nothing. There are no other offloadable tensors.
- The KQ mask is ctx x ubatch x 4 bytes per attention-hosting card. At 256K
  it forces ub 256 (mask 256 MiB/card); at 128K ub 1024 fits (mask 512/card).
- VRAM growth per context token: ~10.75 KiB on card0 (8 of 12 attention
  layers host there), ~6 KiB on card1.
- Loader-fit margins below ~100 MiB are reproducible but thin; 444 (128K
  preset) is comfortable, 65 (256K variant) works 3/3 loads.

## CLOSED PATHS (final)

- KVarN K6/V5: works (16k validated, coherent, tail audit passed: exact
  suffix is served from the F16 stage inside the native shader; route-log
  tail_stride=0 is expected with no enlarged tail), but costs 30% shallow,
  45% deep decode, 57% prefill vs Q8. Parked; user dropped it for speed.
- MTP: re-verified 2026-08-30 at 128k on the mtp-fix fork (draft on R9700,
  LLAMA_ATTN_ROT_DISABLE=1): 18.8 t/s vs 28.3 MTP-off, 77% acceptance. The
  2.43 GiB draft reads its full weights every forward. Closed permanently.
- IQ3_XXS 128k ub 256 (tg 40.5, pp 425, 66 MiB margin): valid fallback if
  Q2 body quality is rejected on real workloads.
- IQ3_XXS 256k (3 CPU expert layers, ub 256): tg 34-36, pp 292. Superseded
  by Q2 at 256k which needs zero offload.

## REMAINING WORK (optional)

1. Quality eval of the Q2 body on real user workloads (only the standard
   probes were run; Q2 vs IQ3_XXS A/B on actual chat tasks is the one open
   quality question).
2. Deep-context decode measurement: all tg@d9060 numbers are at depth 9,060.
   A ~100K-token prefill then n_predict 64 at the 128k preset would give the
   realistic deep-context tg figure (~10-20% below shallow is expected).
3. The beellama tree (llama.cpp-beellama-044, branch codex/qwen4exp-kvarn)
   still holds uncommitted Qwen+KVarN integration; preserve or upstream it
   deliberately. Its scheduler diagnostics are still in ggml-backend.cpp.

## MANDATORY PRE-LAUNCH CHECKS (unchanged)

    pgrep -x llama-server
    ss -ltnp | grep -E ':(1234|8080|18250|18251) '
    free -g   (need >= 30 GiB MemAvailable)
    cat /sys/class/drm/card{0,1}/device/mem_info_vram_used

XTX (card1) settled ceiling 24,300 MiB (user, 2026-08-30). R9700 (card0) total
32,590 MiB. Port 8080 is the Docker searxng listener.

## KEY FILES

- Preset: llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh
- Measurements: llama_server/records/2026-08-30-flashnext-ctx/RESULTS.md
- Q2 model: llama_server/models/Qwen3.8-Flash-Next/UD-Q2_K_XL/ (73.5 GiB)
- Full runs: /tmp/flashnext-q2-128k-full2, /tmp/flashnext-q2-256k-full3,
  /tmp/flashnext-q8-128k-full, /tmp/flashnext-q8-256k-full, /tmp/flashnext-q8-92k-full
- Probe/harness: /tmp/flashnext_ctx_probe.sh, /tmp/flashnext_mmap_verify.sh
- Probe ladder: /tmp/flashnext-ctx-probe-p0..p14, -q2a..q2c