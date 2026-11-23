# int8 KV cache for the Inkling-2-Mini Loom R9700 package (DONE 2026-11-23: see kv8_patch.py and README.md; original plan below)

Goal: halve KV memory (f16 -> int8) so the R9700 fits ~450K+ context (today: 288K window, 300K kills the model)
and images above ~253K.

## Why fixed-scale int8 is viable (no per-token scale buffers)
From `g_ink_kv_prep_store` in program/llm.loom:
- K is L2-clamped per head to norm <= 22.627 (`%qk_thr = 22.6274`, sqrt(512)), 128 dims.
- V is RMS-normalized per head (`%vir = rsqrt(mean(v^2) + eps)`), so rms = 1, |v| <= sqrt(128) = 11.31.
So static scales work: K_q = round(k * 127/22.63), V_q = round(v * 127/11.31); dequant = q * (22.63/127) / (11.31/127).
Expected per-element step ~0.18 (K, typical |k| ~2) and ~0.09 (V, typical |v| ~1): ~30 dB SNR, the same class as
static-scale int8 KV in vLLM. Must be verified with needles + output comparison against f16 before any deploy.

## Edit surface (program/llm.loom, 6 MB generated, no kit to regenerate it)
- Position stride `%pos_bytes = index.constant 2048 : offset` appears 21x; storage.loom has
  `%kv_position_bytes = 2048` (arena sizing) and `%block_bytes = 32768` (+ draft block 32768).
- 25 kernels touch `buffer.view %arena`:
  - writers: g_ink_kv_prep_store, _lv, _swa_cur(_lv), _swa_ring, g_ink_sconv1_dyn_kv_prep_store,
    g_ink_sconv1_swa_kv_prep_store, g_dsp_kv_store_paged (drafter)
  - readers: g_ink_attn_split, _split_span, _direct, _split_swa, _split_swa_span, g16_ink_attn_split_listed(_swa),
    g16_ink_attn_combine_listed, g_dsp_attn_split_full
  - copies/state: g_ink_swa_commit, g_ink_swa_fork_copy_all, g_ink_sconv_* (commit/replay/stat/dyn) — these share
    the arena with the short-conv recurrent state at offsets derived from the same stride: halving the stride moves them.
- Conversion ops available in this IR: vector.fptosi / vector.sitofp / vector.fptrunc / vector.extf / bitcast.

## Steps
1. Copy Inkling-2-Mini-r9700-c262144-d3-vision -> ...-kv8 (hard links for weights; new program files).
2. Map the arena layout precisely (g_kv_offset, storage.loom geometry[], sconv/swa regions) before touching kernels.
3. Writers: f32 -> scale -> fptosi -> vector<4xi8> store into view<8x32x4xi8>; stride 2048 -> 1024 for K/V planes only.
4. Readers: load vector<4xi8> -> sitofp -> mul scale (replaces the f16 load + extf).
5. storage.loom: kv_position_bytes 2048 -> 1024, block_bytes 32768 -> 16384 (KV planes only; keep sconv regions).
6. Test on loom-vtest (R9700, production stopped): smoke, needle at 50/90% of 450K, 12-turn soak, BetterBench, and
   token-level comparison of 20 prompts vs the f16 package (same seed) — deploy only if equal-or-near-tie.
Estimate: 3-5 days; real risk of silent quality loss, hence the gate in step 6.
