## ⏩ HANDOFF — CURRENT STATUS (2026-06-10, ~23:30)

Goal in flight: **MoE (Qwen3.6-35B-A3B) + DFlash + KVarN** on the 7900 XTX. Most of it WORKS.

**Done & verified:**
- ✅ MoE DFlash drafter downloaded: `~/Desktop/local_llm/llama_server/models/qwen36-35b-a3b-dflash-IQ4_XS.gguf` (266 MB).
  The 27B drafter does NOT work with the MoE — this is the right one.
- ✅ BeeLlama **v0.3.2 Preview** built at `~/Desktop/local_llm/llama.cpp-beellama-032-hip/`
  (tag `preview-v0.3.2`, commit `98caf25`). Binary: `build/bin/llama-server`.
- ✅ gfx1100 `simm16` turbo-FA overflow recurred (same as v0.3.1) and was FIXED with the stub
  workaround: `gen_hip_turbo_stubs.py` (copied from the v0.3.1 dir) generated 137 stubs into
  `ggml/src/ggml-cuda/template-instances/hip-stubs/`, and `ggml/src/ggml-hip/CMakeLists.txt` was
  patched in 3 places (turbo-mma glob exclude+stub, `ggml_add_fattn_vec_pair_rocm` macro redirects
  turbo pairs to hip-stubs, ALL_QUANTS glob filter). Rebuild = 0 simm16 errors. KVarN is unaffected
  (own `kvarn.cu` path).
- ✅ **KVarN alone** on MoE @ 64k, `--cache-type-k kvarn5 --cache-type-v kvarn4`: healthy,
  coherent ("Paris"), **100.6 t/s decode**, VRAM 20.1 GB.
- ✅ **KVarN + DFlash together** on MoE @ 64k: loads healthy, `dflash: contract ok`, `vocab_match=1`,
  GPU cross ring enabled, KVarN non-unified KV (`n_parallel=4`) coexists with DFlash. VRAM 21.5 GB
  (~3 GB headroom on 24).

**NEXT STEP (was interrupted here):** run a longer completion against the KVarN+DFlash server to
(a) confirm generation stays coherent and (b) measure decode t/s vs the KVarN-alone 100.6 baseline
to see DFlash's actual speedup. The server **may still be running on port 1235** — check with
`curl -sf http://127.0.0.1:1235/health`; kill with `fuser -k 1235/tcp` when done. Launch cmd is in
§3 below (smoke test 2b). Test prompt that was about to run:
```bash
curl -sf http://127.0.0.1:1235/completion -H "Content-Type: application/json" \
 -d '{"prompt":"<|im_start|>user\nExplain binary search trees.<|im_end|>\n<|im_start|>assistant\n","n_predict":200,"temperature":0.0,"cache_prompt":false}'
```

**⚠️ IMPORTANT CAVEAT found at runtime:** the server warns
`KVarN preset kvarn_k5v4_g128 is experimental; only kvarn_k4v2_g128 is reference-aligned`.
So `kvarn5/4` runs but is flagged experimental by the author; only **`kvarn4-kvarn2`** is
"reference-aligned" (fully validated). Decision needed: ship the experimental sweet-spot `kvarn5/4`
(best quality-per-MiB per the chart) or the conservative reference `kvarn4-kvarn2` (smallest, but
chart shows ~turbo3-level KLD). Worth benchmarking both.

**Remaining TODO after verification:** add a MoE+DFlash+KVarN option to
`llama_cpp_server_launch.sh` (new menu entry → v0.3.2 binary, plain MoE target
`Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`, `--cache-type-k kvarn5 --cache-type-v kvarn4`, the dflash flags,
`--spec-dflash-cross-ctx 1024`). NOTE: DFlash and MTP are mutually exclusive — use the **plain MoE**
target, not the MTP variant.

---

# KVarN KV-Cache Experiment Plan (BeeLlama v0.3.2 Preview)

Goal: try Anbeeld's new **KVarN** KV-cache quantization on the 7900 XTX (gfx1100, 24 GB)
and find a K/V pair that beats our current `turbo3` on quality at similar VRAM, then wire the
winners into `benchmark.py`.

Source chart: Anbeeld, "Qwen 3.6 27B KV cache quant benchmarks: 75 pairs" (r/Qwen_AI, June 2026).
Articles: <https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context>
and "KVarN KV Cache: Implementation and Benchmarks".

---

## 0. Why bother — read the chart first

Axes: **X = KV cache size (MiB)** (smaller = less VRAM), **Y = mean KLD** (lower = closer to
fp16, i.e. better quality). Lower-left is the ideal corner.

The thing that should make us switch:

| What we run now | Size | KLD | Verdict |
|---|---|---|---|
| **`turbo3` (current default)** | ~800 MiB | **~0.012** | **worst quality point on the whole chart** |
| `turbo4` | ~1050 MiB | ~0.0047 | |

KVarN sits on the Pareto frontier — it gets q5/q6-class quality at q4-class size:

| KVarN pair (K-V) | Size | KLD | Beats… |
|---|---|---|---|
| `kvarn3-kvarn3` | ~890 MiB | ~0.0053 | same size as turbo3, **~2.3× better KLD** |
| `kvarn4-kvarn3` | ~1010 MiB | ~0.0038 | small bump, **~3× better than turbo3** |
| `kvarn4-kvarn4` | ~1120 MiB | ~0.0029 | smaller **and** better than `q5_0` (1410/0.0032) |
| `kvarn5-kvarn4` | ~1250 MiB | ~0.0028 | sweet spot — q6-class quality, q4-class size |
| `kvarn6-kvarn5` | ~1410 MiB | ~0.0026 | matches `q6_0` quality at less size |
| `kvarn8-kvarn8` | ~2170 MiB | ~0.0023 | ties `q8_0` (best quality, same size — no win) |
| `kvarn4-kvarn2` | ~880 MiB | ~0.0104 | smallest KVarN but ~turbo3-bad — **avoid** |

**Takeaway:** for ~the same VRAM we spend on `turbo3` today, `kvarn3-kvarn3` roughly halves the
quality loss; for a ~200–450 MiB bump, `kvarn4-kvarn4` / `kvarn5-kvarn4` get us to q5/q6 quality.
That's free context-quality on the exact hardware we already have.

> Note: numbers above are read off the scatter plot, so treat as ±. The point of this doc is to
> reproduce them locally and confirm on *our* card.

---

## 1. What KVarN is / flag syntax

- New KV cache quant type in BeeLlama, **v0.3.2 Preview** (tag `preview-v0.3.2`, 2026-06-05).
- Exposed through the normal args, K and V independently:
  ```
  --cache-type-k kvarn8  --cache-type-v kvarn5
  ```
- Levels: `kvarn2 | kvarn3 | kvarn4 | kvarn5 | kvarn6 | kvarn8` (number ≈ effective bits/precision;
  higher = bigger + better quality). 9 valid K/V combinations.
- Works on **128-slice-compatible heads** — Qwen3.6-27B's 128-dim heads qualify. Larger heads are
  treated as multiple pseudo-heads. Runtime validates unsupported placements and aborts cleanly.
- **AMD path exists**: the v0.3.2 notes explicitly add "KVarN support for ROCm/HIP and Vulkan",
  with a "low-shared-memory store path optimized for AMD hardware constraints." So this is meant
  to run on our gfx1100, unlike the turbo FA mess we had to stub.
- Same FA rule as all quantized KV in BeeLlama: **flash attention required** (`--flash-attn on`).

---

## 2. Upgrade BeeLlama v0.3.1 → v0.3.2 Preview

Keep the working v0.3.1 build; put the preview in a **separate dir** so a failed build doesn't
brick our DFlash setup.

```bash
cd ~/Desktop/local_llm
git clone --branch preview-v0.3.2 --depth 1 \
  https://github.com/Anbeeld/beellama.cpp.git llama.cpp-beellama-032-hip
cd llama.cpp-beellama-032-hip
```

### gfx1100 build — reuse our v0.3.1 workaround if needed
The v0.3.2 notes claim AMD KVarN support, but the **turbo FA simm16 branch overflow** that bit us
on v0.3.1 may still be present (it's a separate kernel family). Build the normal way first:

```bash
HIP_VISIBLE_DEVICES=0 cmake -B build \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
  -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

- **If it builds:** great, the preview fixed it or doesn't trigger it.
- **If it dies with "branch size exceeds simm16"** on `fattn-*turbo*` kernels: re-apply our v0.3.1
  fix — copy `gen_hip_turbo_stubs.py` from `../llama.cpp-beellama-hip/`, run it to regenerate the
  55 stub `.cu` files, and port the CMakeLists stub-swap block in
  `ggml/src/ggml-hip/CMakeLists.txt`. KVarN kernels are a *different* family and should NOT be
  stubbed — only stub the `*turbo*` FA files exactly as before.

Binary lands at `build/bin/llama-server`.

---

## 3. Smoke test (before benchmarking)

```bash
HIP_VISIBLE_DEVICES=0 ~/Desktop/local_llm/llama.cpp-beellama-032-hip/build/bin/llama-server \
  -m ~/Desktop/local_llm/llama_server/models/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  -ngl 99 -c 65536 \
  --cache-type-k kvarn5 --cache-type-v kvarn4 \
  --flash-attn on --port 1235 --host 127.0.0.1
```
Confirm: (a) it loads without "unsupported placement" abort, (b) `/health` returns ok,
(c) a quick completion is coherent. Then Ctrl-C.

---

## 4. Configs to add to benchmark.py

`benchmark.py` already keys off a per-config `quant` string passed straight to
`--cache-type-k/-v`. KVarN needs **independent K and V** values, which the current single-`quant`
field can't express. Two options:

- **Quick:** for symmetric pairs (`kvarn4-kvarn4`, etc.) the single field works as-is — pass
  `"kvarn4"` and it sets both K and V the same.
- **Proper:** widen the tuple to allow `quant` to be either a string (symmetric) or a
  `(k, v)` tuple, and have `build_server_cmd` split it. Asymmetric winners (`kvarn5-kvarn4`,
  `kvarn6-kvarn5`, `kvarn8-kvarn5`) need this.

Add `BIN_BEELLAMA_032 = f"{LLM_DIR}/llama.cpp-beellama-032-hip/build/bin/llama-server"` and these
rows on the 27B Q4_K_M model at 64k (matches the article's test context) and 128k:

```
# label                       model          ctx  K-cache   V-cache   spec    binary
27B kvarn3/3   64k             MODEL_27B_MTP  64   kvarn3    kvarn3    none    BIN_BEELLAMA_032   # turbo3-size, better
27B kvarn4/3   64k             MODEL_27B_MTP  64   kvarn4    kvarn3    none    BIN_BEELLAMA_032
27B kvarn4/4   64k             MODEL_27B_MTP  64   kvarn4    kvarn4    none    BIN_BEELLAMA_032
27B kvarn5/4   64k             MODEL_27B_MTP  64   kvarn5    kvarn4    none    BIN_BEELLAMA_032   # sweet spot
27B kvarn6/5   64k             MODEL_27B_MTP  64   kvarn6    kvarn5    none    BIN_BEELLAMA_032
# baselines for apples-to-apples on OUR card:
27B turbo3     64k             MODEL_27B_MTP  64   turbo3    turbo3    none    BIN_TURBOQUANT     # current default
27B turbo4     64k             MODEL_27B_MTP  64   turbo4    turbo4    none    BIN_TURBOQUANT
27B q8_0       64k             MODEL_27B_MTP  64   q8_0      q8_0      none    BIN_MAINLINE       # quality ceiling
# repeat the kvarn5/4 + kvarn4/4 winners at 128k to confirm VRAM headroom
```

The benchmark only measures **size (VRAM) + speed (t/s)** — it does NOT measure KLD/quality.
The chart already gives us quality; our job here is to confirm the VRAM numbers hold on a 24 GB
7900 XTX and that decode t/s isn't worse than turbo. (KVarN being heavier math than turbo could
cost a few t/s — that's the tradeoff to watch.)

---

## 5. Open questions to settle during the run

1. Does the preview build on gfx1100 clean, or does it still need the turbo-FA stubs?
2. KVarN decode t/s vs turbo3/turbo4 — is the quality win worth any speed cost?
3. Can `kvarn5-kvarn4` hold **128k** context on 24 GB alongside the Q4_K_M weights (16 GB)?
4. KVarN + speculative decoding (confirmed from v0.3.2 changelog + KVarN article):
   - **DFlash + KVarN: explicitly supported.** "Target-only KVarN parameters are cleared from
     DFlash draft contexts so draft setup stays independent" — KVarN compresses the *target* KV,
     draft runs on its own path. They tuned "DFlash + KVarN visible context sizing".
   - **MTP + KVarN: no documented blocker, but not explicitly tested.** The rollback machinery any
     spec decoder needs (`seq_rm`, prompt-cache rollback) is implemented for the full-KVarN path.
     Treat as "should work — verify with a launch smoke test" for the dense-27B + kvarn + MTP combo
     we'd ship. Watch the unified-vs-non-unified stream edge (KVarN forces non-unified streams).
   - **NOT supported: hybrid cache** — mixing a standard type and KVarN across K/V
     (e.g. `--cache-type-k q8_0 --cache-type-v kvarn4`). Keep both sides KVarN. This is a cache
     limitation, unrelated to spec decoding.

---

## 6. If it wins

Add a KVarN option to `llama_cpp_server_launch.sh` (e.g. `5) kvarn5/4 (best quality-per-MiB)`)
pointing at the v0.3.2 binary, and make it the new default over `turbo3`.
```
