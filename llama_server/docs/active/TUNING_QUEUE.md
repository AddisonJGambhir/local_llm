# Tuning queue, 2026-08-16

Live worklist. Status: ☐ todo · ◐ running · ☑ done · ✗ rejected

## Constraints that apply to everything

- **7900 XTX (card1) must stay under 24,200 MiB.** It drives the desktop; from
  ~24,180 MiB throughput silently collapses (GTT spillover), and at 24,388 MiB
  the compositor is evicted to GTT and the screen drops to ~3 fps.
  This is NOT an OOM boundary, llama.cpp loads past it and takes the session
  with it. Desktop baseline has grown to ~2.0–3.0 GiB, so budgets from earlier
  tonight are ~1.3 GiB too optimistic.
- Guard: `bin/xtx-guard` (`xtx-guard 23400 &`). Method: `docs/HOW_TO_OPTIMIZE.md`.
- llama.cpp pinned at b10358; b10450 measured −4.8% pp / −4.0% tg.
- Every model here is hybrid (SWA, SSM, or recurrent), draft depth and KV
  precision do not transfer between them.

## Work items

☑ **Muse Glimmer ROCm**, best: DFlash n-max 2 · ubatch 512 · batch 6144 ·
  **f16 KV** · ts 1,1 · 128k → ~986 pp / ~33 tg.

- DFlash worth +47% decode, costs ~11% prefill.
- n-max swept 1/2/4/8/15: acceptance 60.5 → 43.6 → 21.7 → 11.9 → 6.5%.
    Upstream recommends 15; it measures WORST here. Archived note saying 15 was
    for Q5_K_M on a SINGLE gpu.
- **f16 KV: +19% decode, acceptance 43.6 → 58.5%.** KV precision drives
    speculative acceptance, the drafter and target disagree less.
- ts 30,22 (more layers on XTX) is worse on both metrics; keep 1,1.

☑ **Nemotron 3.5 Lightning**, SHIPPED (256k + 1M, speculation OFF), 1M native, Mamba/SSM hybrid MoE, built-in MTP.

- 1M at ts 1,1 = 24426 MiB on XTX → over ceiling. R9700-weighted ts 22,30
    fits at 23330 MiB.
- Unsloth's published 1M config is `-b 1024 -ub 256` on a single 32 GiB card;
    that constraint does not apply here with 56 GiB across two cards.
- Recurrent state is only 47.6 MiB (vs Qwen3.8's 598), should be largely
    immune to the prompt-cache-eviction bug.

☑ **Muse Glimmer Vulkan**, n-max 1/2/4/8/15 + batch + f16. Draft depth has
  diverged by backend on this box before (archived: 15 ROCm / 4 Vulkan on the
  Q5_K_M single-GPU).

☑ **Muse Glimmer YaRN long context**, 3/3 retrieval at 149k, 256k first, 1M only if 256k passes.

- Flags verified present in b10358: `--rope-scaling yarn`, `--rope-scale`,
    `--yarn-orig-ctx`, `--override-kv KEY=TYPE:VALUE`, `-np`.
- Runtime override only; do NOT patch the GGUF.
- **Acceptance criterion is retrieval, not startup**: needle-in-a-haystack at
    10% / 50% / 90% depth. Report prompt size, accuracy, VRAM, pp, tg.
- Evaluate whether DFlash destabilises long context; drop it if so.
- Promising because most attention is sliding-window (2048, pattern 4) and the
    global layers are NoPE, so YaRN should extend more robustly than full-RoPE.

☑ **Qwen3.8 27B with f16 KV**, shipped ts 28,37. REQUIRED, not optional.** It is the only model
  still on q8_0, and only because f16 did not fit at ts 1,1 (KV 8.7 → ~16 GiB at
  256k). Shift layers to the R9700 to make room; drop context below 256k only if
  that is not enough.

## STANDING RULE: f16 KV everywhere

q8_0 KV is not a neutral space/speed tradeoff on these models:

- Muse Glimmer: f16 gave +19% decode and raised DFlash acceptance 43.6 → 58.5%.
    KV precision drives how often drafter and target agree.
- Nemotron 3.5: q8_0 is BROKEN. A 512-token generation freezes at ~488 tokens
    and the response is never sent (4 reproductions, clean server, independent of
    ignore_eos). f16 completes the same request in 6.4 s at 80.6 t/s.
- Qwen3.6-35B: already f16.
Use q8_0 ONLY where f16 genuinely cannot fit, and record why in the profile.
The benchmark harness now defaults to f16 so it cannot be tested by accident.

☐ **`--main-gpu` on the new models**, every profile uses `--main-gpu 1`
  (R9700), inherited from a Qwen3.8 measurement where it was speed-neutral
  (1086.0 vs 1086.7 pp, 38.02 tg both) but kept ~2.5 GiB off the XTX. With
  split-mode layer it governs only non-layer buffers (output, intermediates,
  projector), not which card computes, layers follow --tensor-split. UNTESTED on
  Muse (6656 hidden dim → larger intermediates). Test 0 vs 1.

☐ **Two tiers per model**, a small-context/high-throughput profile and a
  large-context profile, since batch size and context trade against each other
  through the compute buffers.

☐ **Same-harness comparison of all models**, current numbers were produced by
  three different harnesses across the session and are NOT comparable. Qwen3.8's
  headline 38 tg came from a 50-token-context synthetic; its real Pi figure was
  27.84. Re-measure everything with one harness before publishing any table.

☑ **Ship**: profiles + launcher menus for all four models × both backends.

## Known bugs found along the way

- `common.sh` defaulted `LLAMA_DFLASH_SPEC_TYPE` to `dflash`; the current build
  requires `draft-dflash`. Any DFlash profile relying on the default fails to
  start. **Fixed.**
- `common.sh` only emitted `--spec-draft-type-k/v` on the mtp path, so the
  DFlash drafter's KV precision was unreachable from a profile. **Fixed.**
- `--cache-ram` / `--ctx-checkpoints`: a cached context state is KV +
  N checkpoints; `server_prompt_cache::alloc()` silently refuses entries larger
  than the whole budget (warns only at trace level). Qwen3.8 at 32 checkpoints ≈
  30 GiB per entry against a 16 GiB budget → never cached → full reprocess.
  Mitigated with `--ctx-checkpoints 8` + `--cache-ram 24576`.
  **CONFIRMED FIXED under real use, 2026-08-17.** The user ran long sessions
  after the change and the 119k-token reprocessing has not recurred. Shipped in
  all 12 profiles. This line previously said "NOT yet confirmed"; it is now.

## Session close, 2026-08-16

All models except the two open upstream bugs are shipped and boot-verified.
See RESULTS_2026-08-16.md for the measured table and the findings that generalise.

Remaining, in value order:

1. Same-harness Pi validation of the shipped set (numbers above are harness, not Pi).
2. `--main-gpu 0 vs 1` on Muse (inherited from one Qwen3.8 test).
3. b10450 + a one-line `-funsafe-math-optimizations` patch, to get newer server
   fixes at no perf cost.
4. Muse YaRN at 1M (rope-scale 8), only after 256k has seen real use.

## Note to self, filed 2026-08-17 — prompt-cache follow-ups

Both of these are **quality-of-life, not bugs**. The context-loss failure is
fixed and confirmed; nothing here is urgent and nothing is currently broken.

☐ **Re-tune `--ctx-checkpoints` now that the crisis is over.** 8 was picked to
  get the entry under budget, not because 8 was measured as optimal. The
  tradeoff it buys is coarser rewind granularity: when a harness edits history
  mid-conversation, you replay from a more distant checkpoint. On Qwen3.8 27B a
  checkpoint is 690 MiB, so the budget math per cached context is:

      KV at 256k f16  ~17.4 GiB
      + N x 690 MiB   (N = --ctx-checkpoints)
      must stay under --cache-ram 24576 MiB

  That leaves roughly 7 GiB of headroom, i.e. **N up to ~10** on Qwen3.8 before
  entries stop being cached again. On Qwen3.6 35B a checkpoint is only 251 MiB
  and KV is 5.44 GiB at 256k, so that model could carry far more.
  Worth doing: raise per-model to just under its own ceiling rather than using
  one flat 8 everywhere. Measure by timing a mid-history edit, not by startup.
  RISK IF YOU OVERSHOOT: silent, no error — you are straight back to full
  reprocesses. Leave margin and re-verify with a real long session.

☐ **Watch upstream for the silent `alloc()` failure.** The real defect is that
  `server_prompt_cache::alloc()` returns nullptr and only warns at trace level
  when an entry exceeds the budget. A one-line warn-level log would have turned
  a multi-session mystery into an obvious message. Worth filing upstream, or
  carrying locally if we ever stop pinning b10358.

NOT open, recorded so it is not re-investigated:

- `--cache-reuse` is already removed from every profile. It is inert on hybrid
  models regardless — `llama_memory_can_shift()` returns false, so
  `server-context.cpp` forces `n_cache_reuse = 0` at load and `ctx_shift` off.
  The "+3% prefill" once logged for it was something else. Nothing to change.

STALE IN THIS FILE: the Nemotron 3.5 Lightning work item above refers to a model
that was deleted 2026-08-17.
