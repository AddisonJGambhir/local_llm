# Handoff 10: Flash-Next tuning — what shipped, what is closed, and the rules

Supersedes handoff 9 and the GPT interim handoff. Model: Qwen3.8-Flash-Next
125B-A6B UD-Q2_K_XL on R9700 (32,624 MiB, headless) + 7900 XTX (24,560 MiB,
**drives the desktop**).

---

## 0. READ THIS BEFORE TOUCHING A GPU

**Compute memory fit before running anything.** Model weights are about **52 GB**.

- XTX capacity 24.5 GB  ->  it can never hold more than **~47%** of the weights.
- Any `--tensor-split` putting >=47% on the XTX is **arithmetically guaranteed
  to spill**. Do not run it "to see".
- The VRAM guard only arms *after server readiness*. It cannot prevent a
  load-time blowup, and it cannot prevent a spill from freezing the desktop.
- The XTX drives the display. Spilling it dropped the user's desktop to **3 FPS**.
  This already happened once (ts 500,500). Do not repeat it.
- ROCm OOM at load **logs the user out**. This also already happened. Never start
  ROCm at a large unproven context; begin at ctx8192/16384.
- `pgrep -a -x llama-server` first. **Do not stop a server you did not start.**

---

## 1. Production state

Launcher `llama_server/bin/llama-launch`, Vulkan menu now 1-14.

| preset | profile | binary | @199,340 tok |
|---|---|---|---|
| 12 | `preset-flashnext-q2kxl-vulkan-fusions-262k.sh` | `llama.cpp-qwen4exp-vulkan-fusions` | 346.75 pp / 12.854 tg |
| 13 | `preset-flashnext-q2kxl-vulkan-scanfix-200k.sh` | `llama.cpp-flashnext-fusionscan` | 348.09 pp / 17.910 tg |
| **14** | `preset-flashnext-q2kxl-vulkan-qsagather-200k.sh` | `llama.cpp-qsagather` | **347.83 pp / 18.591 tg** |

Common config: Vulkan layer split, `--tensor-split 400,600`, ctx 200704,
ub192, batch 2048, q8_0 K/V, flash-attn on, no speculation, all experts in VRAM,
`-ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'`, `--cache-ram 8192`.

Records: `llama_server/records/2026-08-31-flashnext-scanfix/RESULTS.md` (Parts 1-11).

---

## 2. The one large win, and why it existed

Preset 13 is **+39.3% tg over preset 12** (12.854 -> 17.910 at 199,340 tokens),
prompt speed unchanged, fully resident.

Cause: the fusions worktree sits on base `aa1b06d0c` and **predated merged
upstream commit `62acc89c2` (PR 28011)**. `llama_kv_cells::for_each_token_in`
walked all 256 `LLAMA_MAX_SEQ` slots for every used cell. That path is reached
`for_each_token_in <- get_prev_tokens <- build_ple` — the per-layer-embedding
n-gram lookup, which is **model architecture and runs on every decode step even
with `--spec-type none`**. At 199K with one sequence that is roughly **51 million
wasted CPU iterations per generated token**.

The gain scales with used cells: +7.4% at 11.5K, +39.3% at 199K.

**Lesson: check what merged upstream fixes the local worktree is missing before
optimising anything.** The base was 56 commits behind.

Preset 14 adds PR 27977 commit `d7ca8ea` (QSA gather), decode-only by
construction: it fires when `4*n_tps*width < n_kv`, i.e. width 2051 gives 8204 at
decode (fires) and 1.57M at ub192 prefill (never fires). Handoff 9 wrongly blamed
this commit for a prompt regression.

---

## 3. Measured and rejected — do not redo these

All at 199,340 tokens, real sampling (temp 1.0 / top_k 20 / top_p 0.95) unless noted.

| lever | result |
|---|---|
| **MTP / speculative decode** | controlled 4-arm A/B at 11.5K: no-spec **613 pp / 40.92 tg** vs MTP 378-380 pp / 31.7-33.9 tg. Control wins **+62% pp, +21% tg**. `--spec-draft-backend-sampling` does NOT help (arm C worse than B). Draft costs ~38% prefill before speculation earns anything. |
| PR 28040 (O(log n) get_prev_tokens) | 350.05 pp / 18.012 tg — flat. 28011 already took the win. |
| PR 27973 (Vulkan GDN fusion) | flat (40.46 vs 40.59 tg @11.5K). PR is draft and has a real defect — see §6. |
| PR 28023 (indexer head slices) | noise on Vulkan (598.62 vs 597.69 pp). Author agrees it is mostly readability off fast CUDA. |
| ubatch 256 | 237 MiB migrated off card0 from 161K on |
| ubatch 384 | guard-killed, xtx 24,552 > ceiling, GTT 1,346 -> 2,069 |
| batch 4096 / 8192 | 348.84 pp / 18.62 tg — flat |
| tensor-split 390 / 410 / 420 | flat; 400,600 is optimal within the feasible band |
| ROCm 10 layer split | 310.06 pp / 10.519 tg — loses badly |
| `--split-mode tensor` | requires non-quantized KV; f16 does not fit at ctx200704. At ctx16384/f16: Vulkan 302/24.5, ROCm-noRCCL 369/27.4 vs ~810/39 for layer |
| `--split-mode row` | deprecated upstream |
| Hyperloom / ROCm.AI | `model_gate.py::_detect_gguf_only_checkpoint` **rejects GGUF-only checkpoints**; drives vLLM/SGLang; validated on MI300X/MI355X only |

---

## 4. Methodology rules (learned the hard way)

1. **GTT is not a spill signal on Vulkan.** Healthy runs sit at 1,100-1,900 MiB
   card1 GTT. ts 410 pushed GTT to 2,237 MiB with speed unaffected.
   **The tell is pp/tg collapsing.** Real spill (ts 500,500) cost **-32% pp,
   -41% tg**. That is what a spill looks like.
2. The guard's migration heuristic (VRAM drop >=512 MiB AND GTT rise >=512 MiB)
   **false-positives** and killed a healthy control. Relax with
   `R9700_MIGRATION_MAX=4096 XTX_MIGRATION_MAX=4096` and judge on speed.
3. **Never record MTP metrics at temp 0.** Greedy flatters speculation.
   Use `FN_TEMPERATURE=1 FN_TOP_K=20 FN_TOP_P=0.95`.
4. **temp-1 tg variance is large** (MTP arm B spread 8.12 tg across 3 passes).
   Single-shot numbers are not evidence. Medians of 3+.
5. **Control everything**: same build, prompt, allocation, sampler, seed. Two
   separate conclusions this project got wrong were confounded comparisons
   (fusions-vs-stock, and a cross-session MTP comparison).
6. **The baseline is non-deterministic at temp 0** — GPU parallel reductions mean
   two identical greedy runs differ. Output divergence is NOT proof of a bug.
7. Shallow results do not predict depth. ub384 gave 744 pp at 11.5K and was
   guard-killed at 199K. Anything claimed for a 200K preset must be measured deep.

---

## 5. The structural ceiling

`sched_reserve: graph splits = 2`. The decode graph is **not** fragmented by the
CPU-resident PLE table (commit `aa1b06d0c` already fixed that). Two splits under
layer split means plain pipeline parallelism: **at batch 1 one GPU computes while
the other waits.** Decode runs at roughly a single card's bandwidth regardless of
total VRAM — which is why this machine lands near single-GPU CUDA systems with
far less memory.

The split ratio is **capacity-bound, not speed-bound**: the XTX can hold at most
~47% of the weights, so the feasible band is ts ~390-420, and that band is flat.

Escaping this needs one of:
- quantized-KV support for `--split-mode tensor` (upstream issue **#23567**), or
- a faster GPU-to-GPU interconnect than PCIe.

---

## 6. Worktrees — **preserve all dirty state, do not reset/clean/checkout**

| worktree | base | contents |
|---|---|---|
| `llama.cpp-qwen4exp-vulkan-fusions` | aa1b06d0c | **ORIGINAL, 7 dirty files. Never modified. Source of truth for the Vulkan fusions.** |
| `llama.cpp-flashnext-fusionscan` | aa1b06d0c | preset 13 = fusions + 39efb639/28e37926/b88668b |
| `llama.cpp-qsagather` | aa1b06d0c | **preset 14** = fusionscan + d7ca8ea |
| `llama.cpp-flashnext-full` | 175b66c51 | fusions + scan + full MTP stack + the inject-swap fix (§7) |
| `llama.cpp-gdnfusion` | aa1b06d0c | fusionscan + PR 27973, with the cache-binding defect fixed |
| `llama.cpp-p28040` | aa1b06d0c | fusionscan + PR 28040 |
| `llama.cpp-rocm10-fn` | master 458681e1d | ROCm build, `GGML_HIP_RCCL=OFF` (RCCL was unstable) |

The Vulkan fusions are worth roughly **2x prompt speed at depth** and must be
preserved in every build (stock upstream gives 168 pp where fusions give 348).

---

## 7. Two upstream defects fixed locally (worth upstreaming)

**a) Fusion skinny-inject vs MTP** (`llama.cpp-flashnext-full/src/models/qwen4exp.cpp`).
The fusion's `nt > 8` path does `ggml_mul_mat(ctx0, xn, w_inject)` — operands
swapped, so the *weight* lands on `src1` and Vulkan sees `b_type ==
w_inject->type`. `ggml_vk_get_dequantize_mul_mat_vec` only accepts
F32/F16/Q8_1, so the K-quantised MTP draft aborts. Fixed by guarding the swap on
`w_inject->type` being F32/F16.

**b) PR 27973 cache binding** (`llama.cpp-gdnfusion`). The draft PR initialises
`cache_buf = dst_buf` as a dummy and **never binds the real recurrent-cache
buffer**, while `state_out_off` is an offset into that cache. Fixed by carrying
the CPY destination tensor in `vk_gdn_fused_cache` and binding it via
`ggml_vk_tensor_subbuffer`. (Even fixed, it gives no measurable gain here.)

---

## 8. Open leads, in priority order

1. **`token_embd.weight` on GPU.** The preset offloads *two* tensors but they are
   not one decision. `per_layer_token_embd` is ~20 GB and genuinely cannot fit.
   `token_embd.weight` is only **0.407 GiB**, and the model has a **separate
   `output.weight`** (both `[2560, 248320]`), so the tied-embedding worry does not
   apply — but the input gather may still be running on CPU for no reason.
   Cost to test: 0.407 GiB against ~1.5 GB card0 headroom. **Fits by arithmetic.**
   Try `-ot 'per_layer_token_embd\.weight=CPU'` alone.
2. **MTP at deep context under representative sampling.** Never tested. At 11.5K
   the no-spec control wins decisively, but MTP amortises per-step cost and the
   baseline falls from 40.9 tg (11.5K) to 18.6 (199K). Greedy testing showed MTP
   +46.7% at 151K, but greedy is not admissible. Needs ctx<=163840 (the draft does
   not fit at 200704) and the 4-arm control design from §3.
3. Watch upstream issue **#23567** (quantized KV for `--split-mode tensor`). That
   is the only thing that would break the §5 ceiling.

---

## 9. Harness

`llama_server/bin/flashnext_sweep.sh`. Key vars:
`FN_BIN FN_CTX FN_UB FN_B FN_TS FN_SPLIT_MODE FN_CTK FN_CTV FN_OT FN_DEV
FN_LOADMODE FN_LAZY FN_PROMPT_FILE FN_NPRED FN_IGNORE_EOS FN_WARMUP_PASSES
FN_PASSES FN_SPEC_TYPE FN_EXTRA FN_TAG FN_THREADS`
Sampling: `FN_TEMPERATURE FN_TOP_K FN_TOP_P FN_SEED` (defaults greedy; warns when
speculation is tested greedily).
Guard: `R9700_MAX XTX_MAX MEM_MIN R9700_MIGRATION_MAX XTX_MIGRATION_MAX`.

Note `--tensor-read-lazy` exists on base aa1b06d0c but was **removed upstream** —
use `FN_LAZY=skip` for `llama.cpp-flashnext-full` / `-rocm10-fn`.
On ROCm, `--load-mode mmap` with `-ot ...=CPU` **hangs at load**; use
`--load-mode none` (25s vs >13min).

Prompts: `/tmp/fn-prompt/deep199k.txt` (199,340 tok), `d150k.txt` (150,913),
`d10k.txt` (11,568). Depth-curve tool: `/tmp/fn-prompt/depthcurve.py <rundir>`
prints prefill rate vs depth alongside VRAM/GTT.

No server was running at the end of this handoff.
