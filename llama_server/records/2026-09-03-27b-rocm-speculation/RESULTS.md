# Qwen3.8-27B UD-Q6_K_XL on ROCm — speculation and split-mode, 2026-09-03

Goal: make the user's favourite build (UD-Q6_K_XL, ROCm, `--split-mode tensor`)
much faster, and check for ROCm updates / tensor-split wins.

All runs: ctx 32768, 10,776-token prompt, `n_predict 256`, temp 0.7 / top_k 20 /
top_p 0.95, seed 42, `cache_prompt:false`, 3 passes. `-ub 1024 -b 2048 -fa on`,
f16 KV, `--no-mmap`, `-ngl 99 -fit off`. Binary `llama.cpp-rocm11-dflash2`
(master `67a17c17c`) unless noted.

---

## 1. CORRECTION: the "n-max 5 win" was a shallow-context artifact

An earlier version of this document reported n-max 5 as a +19.4% decode win and
shipped it. **That was wrong and has been reverted.** It was measured at
ctx 32768; the preset runs **ctx 262144**, where n-max 5 is 4.2% SLOWER.

Prefill flat (402-412 pp) at every point in both sweeps -- decode-only throughout.

| n-max | tg @ctx 32768 | tg @ctx 262144 (production) |
|---|---|---|
| 2 | - | 51.87 |
| **3** | 52.44 | **57.31** <- peak, and what was already shipped |
| 4 | 58.92 | 56.21 |
| 5 | **62.63** | 54.90 |
| 6 | 59.52 | - |
| 7 | 54.70 | - |
| 9 | 52.86 | - |
| 12 | 60.20 | - |

n-max 3 reproduced twice at 262144 (57.28, 57.31); n-max 5 reproduced twice at
32768 (62.56, 62.63). Both sweeps are internally sound -- **the optimum genuinely
moves with the KV allocation.** `--spec-draft-n-max` is only a ceiling; the
adaptive PROFIT controller picks real depth and behaves differently at 262144.

**The user's preset was already optimally tuned. No n-max speedup exists for it.**

Handoff 10 4.7 already said "shallow results do not predict depth". This is that
rule again, and it cost a shipped regression. **Never tune this preset at a
smaller context than it runs at.**

---

## 2. Split mode vs speculative method — at the SHIPPED ctx 262144

Everything below is measured at **ctx 262144**, the context the presets actually
run. The ctx-32768 column is kept only to show how badly a shallow proxy misleads.

| split | spec | pp @262k | tg @262k | XTX @262k | tg @32k |
|---|---|---|---|---|---|
| tensor | none | - | - | - | 27.7 |
| **tensor** | **MTP n3** | 412.4 | **57.31** | 24,325 | 52.44 |
| tensor | MTP n5 | 411.3 | 54.90 | 24,354 | **62.63** |
| tensor | DFlash2 / DSpark | **abort** | — | — | abort |
| **layer** | **MTP n3** | **1012.6** | 46.76 | **20,918** | 43.15 |
| layer | DFlash2 n7 | 969.3 | 47.44 | 22,318 | 47.01 |
| layer | DFlash2 n12 | 968.2 | 46.97 | 22,661 | 47.93 |
| layer | DSpark n7 | - | - | - | 41.85 |
| layer | DSpark n12 | - | - | - | 41.44 |

**Two presets ship, both tuned at 262144:**

- **entry 10, tensor + MTP n3** — best decode, 57.31 tg. *Unchanged from before
  this session; it was already optimal.*
- **entry 13, layer + MTP n3** — best prefill, 1012.6 pp (2.5x) for 18% less
  decode, and it leaves the XTX at 20,918 instead of 24,325 (the tensor preset
  sits ~235 MiB from the card's 24,560 physical).

**DFlash2 was measured and rejected.** At 262144 it buys +1.5% decode over plain
MTP (47.44 vs 46.76) while losing 4.3% prefill, and costs a 1.1 GB draft plus a
newer binary. Its ctx-32768 edge (47.0 vs 43.2) does not survive to full context.
DFlash2 n7 also beats n12 at 262144 while losing at 32768 — the same
context-dependence as the MTP n-max curve.

---

## 3. DFlash2 and DSpark cannot use `--split-mode tensor`

Both abort at load in the Meta Backend (tensor-parallel) path:

```
DFlash2: ggml-backend-meta.cpp:543 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)
DSpark : ggml-backend-meta.cpp:537 GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN)
```

Pinning the draft to one GPU with `--spec-draft-device ROCm0`/`ROCm1` does **not**
work around it — it fails differently: *"pre-allocated tensor (output.weight) in a
buffer (Meta()) that cannot run the operation (NONE)"*. So it is the target's
tensor-split graph, not draft placement.

This is the known ROCm/HIP multi-GPU tensor-split limitation flagged in PR #27342.
**Any `dflash`-arch draft is layer-split-only on this machine.**

## 4. DSpark loses to DFlash2 here

Contrary to the general claim that DSpark beats DFlash on all categories, on this
model/hardware DFlash2 wins by ~15% (47.93 vs 41.44). Its Q8_0 draft is 1.46 GB
against DFlash2's 1.1 GB Q4_K_M, so it costs more per step, and it accepts less
(45% vs 54%). Raising DSpark n-max 7 -> 12 changed nothing (41.85 -> 41.44):
identical drafted/accepted counts, so the controller was already saturated.

Drafts used: `z-lab/Qwen3.8-27B-DFlash2-GGUF` Q4_K_M (1.1 GB) and
`erlidev/Qwen3.8-27B-DSpark-GGUF` Q8_0 (1.46 GB). Both `general.architecture =
dflash`; the DSpark one additionally carries `markov_w1/w2.weight`, which is what
makes llama.cpp auto-detect DSpark over DFlash.

---

## 5. ROCm and tensor-split interconnect — nothing to gain

- **ROCm 10.0.0 (2026-08-26) is the latest release.** Already installed. No
  version upgrade exists to leverage.
- **`GGML_CUDA_P2P=1`: no effect.** 52.36 tg vs 52.44 without — identical.
  Confirms the existing note in `docs/active/SUDO_BACKLOG.md`: *"there is no P2P
  route between these cards."*
- **`GGML_HIP_RCCL=ON`: already known-blocked.** `docs/HOW_TO_OPTIMIZE.md` records
  that it works with `-sm layer`/`none` but **segfaults under `-sm tensor`**, and
  warns `Missing "iommu=pt"`. IOMMU here is active (`ivhd0`) in Translated mode.
  Both need a GRUB change + reboot — tracked in SUDO_BACKLOG.

The user had already tested these; they were re-confirmed, not rediscovered.

---

## 6. Build

New worktree `llama.cpp-rocm11-dflash2` at master `67a17c17c` (2026-09-03).
The existing `llama.cpp-rocm10` build is from **2026-08-11** and predates DFlash2
(merged `b10f9ca58`, 2026-08-27), so it cannot load these drafts at all.

Configured identically to the old one: `GGML_HIP=ON`,
`AMDGPU_TARGETS=gfx1100;gfx1201`, Release, ccache, `GGML_NATIVE=ON`,
`GGML_RPC=OFF`, `LLAMA_CURL=OFF`, `GGML_HIP_RCCL=OFF`. Built in ~6 minutes.
`llama.cpp-rocm10` was left untouched.

Gotcha: `common.sh` defaults `--spec-type` to bare `dflash`, which llama.cpp
rejects. The valid value is `draft-dflash`, set via `LLAMA_DFLASH_SPEC_TYPE`.
Caught with `llamactl command --profile <path>` before shipping.
