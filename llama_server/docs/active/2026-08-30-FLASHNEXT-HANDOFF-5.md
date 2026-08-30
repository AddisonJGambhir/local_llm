> SUPERSEDED by 2026-08-30-FLASHNEXT-HANDOFF-6.md. Its technical content
> (presets, split map, placement laws) remains valid; its safety rules do NOT --
> handoff 6 adds a mandatory card0 (R9700) ceiling after a machine crash.

# Handoff 5: Q2_K_XL ships 128-256K at full speed; ROCm tested and closed; splits mapped

Supersedes `2026-08-30-FLASHNEXT-HANDOFF-4.md`. Read that document and
`records/2026-08-30-flashnext-ctx/RESULTS.md` for the full measurement log;
this file is the current state. The user's goal evolved to: 128K+ context,
40-50 tg, 500+ pp, Q8 KV quality. That goal is MET and shipped. The open
frontier is deep-prefill behavior and any remaining split/backend tricks.

## CURRENT STATE -- READ THIS FIRST

Shipped preset: `llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh`
(UD-Q2_K_XL body, ctx 131,072, ub 1024, b 2048, q8_0 KV, FA, ts 385,615,
PLE on CPU mmap, --tensor-read-lazy on, Vulkan only, build
`llama.cpp-qwen4exp/build-vulkan` at `aa1b06d0c`).

Validated build menu (all zero offload except noted, all Q8_0 KV):

| build | ctx | ub | pp @d9K | tg shallow | tg @d9K | R9700 spare | XTX spare |
|---|---|---|---|---|---|---|---|
| 128K shipped | 131,072 | 1024 | 654 | 42.1/42.5 | 28.8 | 444 | 1,420 |
| 224K hi-pp | 229,376 | 1024 | 613 | 40.5/41.2 | 29.0 | 677* | 508* |
| 256K max | 262,144 | 256 | 440 | 41.4/42.3 | 29.4 | 65 | 1,769 |

*224K hi-pp offloads 512 MiB (blk.30 gate+up exps) to CPU; tg unaffected.
The whole ladder 128/160/192/224/240/256K runs at ~41.5 tg shallow.
262,144 is the trained maximum (qwen4exp.context_length in the GGUF).

No llama-server is left running by me. The user's own server, if present,
takes priority; check before any launch.

## MANDATORY BEFORE EVERY MODEL LAUNCH

    pgrep -x llama-server
    ss -ltnp | grep -E ':(1234|8080|18250|18251) '
    free -g
    cat /sys/class/drm/card{0,1}/device/mem_info_vram_used

Need >= 30 GiB MemAvailable. Port 8080 is the Docker searxng listener.
Kill only the exact test PID; never `pkill -f` or `pkill -x bash`.
XTX (card1) settled ceiling 24,300 MiB per the user. R9700 (card0) total
32,590 MiB. Vulkan0 = XTX (hosts layers 0-17), Vulkan1 = R9700 (18-47).
ROCm0 = XTX, ROCm1 = R9700.

## ROCm 10 -- TESTED, CLOSED (do not reopen without new evidence)

The user expected ROCm 10 to be a big win. It was tested properly and it is
not, for this model, in this build:

1. `--load-mode mmap --tensor-read-lazy on` HANGS the ROCm loader (1 core
   spins, no I/O, no progress; 3 reproductions). Use `--load-mode none` on
   the ROCm path if you ever rerun it; that costs 26.8 GiB anonymous RAM.
2. Same 128K config as Vulkan: tg 26.6 vs 42.1 = 37% decode LOSS.
3. 224K: OOM on the R9700 compute buffer (ROCm compute buffers are larger
   than Vulkan's).
4. The ROCm 10 dense-27B decode win (+6-18%, records/2026-08-29-rocm10)
   does NOT transfer to the qwen4exp sparse arch.
5. NOT finished (user aborted): `-sm tensor` and `-sm row` on ROCm 16K.
   Expect tensor to abort on the meta-backend error (PLE is CPU-resident)
   and row to fail to load (it did on 27B). Verify quickly if you want the
   matrix complete; do not spend longer.

## SPLIT MODES -- FULL MAP

- Layer split (shipped): works, all measurements above.
- `-sm tensor`: aborts "multi buffers are not supported by the meta backend"
  whenever any tensor is CPU-resident. PLE (26.8 GiB) cannot fit in VRAM,
  so tensor split is dead for this model on any backend. (It worked on
  ROCm 10 for the dense 27B only because nothing was on CPU.)
- `-sm row`: unimplemented ("does not support split buffers" on RADV; failed
  to load even on ROCm 10 for the 27B).
- Cross-card expert moves via -ot: -8.5% tg per 962 MiB set; whole-layer
  moves: -32%. Never rebalance across cards; offload overflow to CPU only.
- Rebalance via ts does not free the R9700 (handoff-2 table, re-confirmed).

## PLACEMENT LAWS (measured this session, reuse)

- CPU expert layer: frees 962 MiB on its host card, costs ~7% tg.
  blk.2's set is 1,138 MiB (down 450, gate/up 344 each; normal layers down
  450, gate 256, up 256).
- token_embd and output (486 MiB each) are CPU-resident BY DEFAULT.
- KQ mask = ctx x ubatch x 4 bytes per attention-hosting card. It is why
  256K forces ub 256. Attention layers {3,7,...,47}; XTX hosts 4 of 12.
- VRAM growth per ctx token: ~10.75 KiB on card0, ~6 KiB on card1.
- The loader's --fit is DEFAULT ON but aborts whenever -ngl 99 is passed
  ("n_gpu_layers already set by user"). Nobody has tested --fit on with
  --fit-target on this model; the HF community reports automatic fit beating
  manual offload on CUDA rigs. If you explore this, remove -ngl 99 and read
  --fit / --fit-target / --fit-ctx help. This is the one untried lever.

## MTP -- CLOSED PERMANENTLY

Re-verified 2026-08-30 at 128K on the mtp-fix fork: tg 18.8 (n1) vs 28.3
off, 77% acceptance. Draft reads its full 2.43 GiB every forward and
displaces 2.43 GiB of context VRAM. Community MTP claims of 40-170 t/s are
on NVFP4/Blackwell CUDA rigs (DataCamp guide) that are not comparable.

## COMMUNITY BENCHMARKS (for calibration, from unsloth discussion #3/#22)

RTX 6000 PRO 96GB: 59.7-70 t/s. 2x 4090D-48GB: 44.7. Strix Halo: 20-23.
Dual 16GB consumer cards: 23.3 (this is the real "5070 Ti" number; 40 t/s
claims for this model on such hardware do not exist). CPU-heavy rigs: 15-19.
Our 40-42 t/s at 128-256K is at the top of comparable hardware. Deep-context
decode decay is universal (CUDA 60 -> 30 after 128K ctx); the QSA indexer
scores every populated cell per token.

## OPEN ITEMS, IN PRIORITY ORDER

1. Deep-context measurement rerun (was killed twice by harness bugs, both
   fixed): 224K hi-pp build, /tmp/bigprompt-200k.txt (~199K tokens),
   n_predict 64. Use FLASHNEXT_* envs on /tmp/flashnext_mmap_verify.sh.
   Known so far: cumulative prefill avg 116 t/s at 104K depth and falling;
   a full 199K prefill is ~30 min. The missing number is tg at ~199K depth.
2. --fit on / --fit-target exploration (the one untried placement lever;
   see PLACEMENT LAWS above).
3. ROCm -sm tensor / -sm row 16K probes if you want the matrix complete.
4. The beellama tree (llama.cpp-beellama-044, branch codex/qwen4exp-kvarn)
   still holds the uncommitted Qwen+KVarN integration; preserve or upstream
   deliberately. KVarN itself is parked: works, coherent, tail audit
   passed, but 15.2 vs 28 tg @d9K.

## HARNESS NOTES (my bugs, already fixed, do not repeat)

- /tmp/flashnext_mmap_verify.sh originally hard-coded the model path, ub 1024
  and a 900s curl timeout. Three "collapsed" Q2 runs were actually IQ3_XXS
  loaded at Q2-only settings; the loader was never nondeterministic.
- The deep-200K run died twice: once to the user interrupt (expected),
  once to the 900s curl timeout. Now 3600s.
- Both scripts are now fully env-configurable; read their headers before
  inventing new flags.

## KEY FILES

- Preset: llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-layer-128k.sh
- Measurements: llama_server/records/2026-08-30-flashnext-ctx/RESULTS.md
- Models: llama_server/models/Qwen3.8-Flash-Next/{UD-Q2_K_XL,UD-IQ3_XXS,MTP-Q4_K_M}
- Probe/harness: /tmp/flashnext_ctx_probe.sh, /tmp/flashnext_mmap_verify.sh
- Full runs: /tmp/flashnext-q2-128k-full2, -q2-224k-full, -q2-224k-ub1024-full,
  -q2-256k-full3, -q8-128k-full, -q8-256k-full
- ROCm: /tmp/flashnext-rocm-{layer,quick,nommap,layer224,layer128}
- 200K prompt: /tmp/bigprompt-200k.txt
- Prior handoffs: docs/active/2026-08-29-FLASHNEXT-HANDOFF-{,2,3}.md, 2026-08-30-FLASHNEXT-HANDOFF-4.md