# Draft comment for llama.cpp PR #25051

Rewritten 2026-08-22. The previous draft is kept as
`PR25051_COMMENT_DRAFT-SUPERSEDED.md`; it contained two factual errors (PCIe
generation, and an RCCL claim made against a build where RCCL was never
compiled in) and led with the weaker ROCm comparison instead of the direct
stock-vs-PR one.

**REVIEW BEFORE POSTING.** Nothing here is posted automatically.

---

AMD dual-GPU data, since the thread asked for non-NVIDIA testers.

**Hardware.** RX 7900 XTX 24 GB (gfx1100, RDNA3, drives the desktop) + AI PRO
R9700 32 GB (gfx1201, RDNA4). Ryzen 9 9950X3D, 64 GB, Ubuntu 26.04, Mesa RADV.

Topology matters here and is unusual:

    XTX    03:00.0 <- 02:00.0 <- 01:00.0 <- root 00:01.1     Gen4 x16, CPU-direct, ~31.5 GB/s
    R9700  14:00.0 <- 13:00.0 <- 12:00.0 <- 4x AMD 600-series chipset bridges
                                         <- root 00:02.1     Gen4 x4,  ~7.9 GB/s

No P2P (different root ports), no xGMI. Asymmetric ~4:1 in link bandwidth and
~1.5:1 in memory bandwidth. Every allreduce round-trips through system memory.

**Model.** Qwen3.8-27B dense, arch `qwen35`, 65 blocks, hybrid (carries
`qwen35.ssm.*`). Q6_K, f16 KV, flash-attn on.

---

## 1. This PR is ~2.9x stock on Vulkan tensor mode, and the effect is isolated to the tensor path

`llama-bench -r 3`, matched settings (`-ub 8192 -p 8192`, `-sm tensor -ts 50/50`):

| build | `-sm tensor` pp8192 | `-sm layer` pp8192 |
| --- | --- | --- |
| stock Vulkan `936918514` | 312.68 | 838.31 |
| **this PR `13ac90156`** | **910.13** | 836.62 |

**2.91x in tensor mode; 0.2% apart in layer mode.** The layer row is the control:
the two builds are indistinguishable when the allreduce path is not exercised,
so the tensor-mode difference is not a general build-version effect.

Caveat, stated plainly: the PR branch bases on b9834 and the stock build here is
b10334, so they are ~500 builds apart. The layer-mode control is what makes the
comparison meaningful despite that; without it I would not quote the number.

`-ub 8192` is used because stock aborts at `-ub 1024` on this model (see §4).

## 2. Output is byte-identical to a known-good reference

Greedy decode (`temp 0`, `top_k 1`, fixed seed), same prompt, 96 tokens:

    PR build,  -sm tensor 50/50 : identical
    PR build,  -sm layer        : identical
    ROCm b10358, -sm layer      : identical   (reference)

Character for character across all three, including the same truncation point.
Worth noting because tensor mode is documented as not implemented for
hybrid/recurrent architectures, and this model is one.

## 3. The advantage grows with context depth

Marginal prefill rate (`-p 2048 -n 0`, `-r 2`) at increasing depth:

| config | d0 | d32k | d65k | d98k | decay |
| --- | --- | --- | --- | --- | --- |
| VK Q8_0 **tensor** 50/50 (this PR) | 959.0 | 744.2 | 587.3 | **495.0** | -48% |
| VK Q6_K **tensor** 50/50 (this PR) | 900.5 | 708.7 | 562.7 | **477.3** | -47% |
| VK Q8_0 row | **1312.7** | 756.9 | 516.6 | 360.5 | -73% |
| VK Q6_K layer 28/37 | 1134.6 | 699.5 | 488.6 | 345.3 | -70% |
| ROCm Q6_K layer 50/50 | 995.3 | 593.3 | 423.6 | 314.9 | -68% |
| ROCm Q6_K tensor 50/50 | 433.8 | 370.1 | 307.4 | **crash** | - |

At empty KV, tensor mode looks worst. At 98k it is the fastest prefill measured,
+48% over the best alternative, because it decays ~47% where layer and row decay
65-73%. Most published benchmarks are at d0, which is the depth least favourable
to this PR.

ROCm tensor cannot complete the d98k cell at all: `HSA exception:
MemoryRegion::BlockAllocator::alloc failed` then `ggml-backend-meta.cpp:1576`.
The Vulkan path completes it.

**Total encode time is a separate question and layer still wins there.** One real
71,449-token prompt through a live server: VK layer 28/37 = 73.1 s, VK row =
72.2 s, VK tensor 50/50 = 103.6 s. Integrating from zero, layer's early advantage
outweighs tensor's late one at this length. I mention it because the marginal-rate
table alone would overstate the case.

## 4. Bug: `--spec-draft-n-max > 8` aborts under `-sm tensor`, works under `-sm layer`

Reproducible, Qwen3.8-27B Q6_K, ROCm b10358, `-c 8192`:

    -sm tensor --spec-draft-n-max 10/12/16  -> ggml-backend-meta.cpp:1576:
                                               GGML_ASSERT(meta_buf_ctx->bufs[i]) failed
    -sm tensor --spec-draft-n-max 8         -> works
    -sm layer  --spec-draft-n-max 16        -> works

Not memory pressure (XTX at ~12 GB of 24 when it aborts) and not a model limit,
since layer split runs the identical configuration. Looks like the meta backend
sizing its buffer array from something that does not account for larger draft
graphs. Possibly related to #24549 / #24566, which fix a different MTP-plus-tensor
crash in the same component.

## 5. RCCL is not a workaround on this platform

Correcting something I nearly reported wrongly: my earlier "RCCL makes no
difference" observation was invalid, because `GGML_HIP_RCCL` was OFF and the
binary was never linked against librccl. Rebuilt properly
(`-DGGML_HIP_RCCL=ON`, `librccl.so.1` linked, RCCL 2.72.5 from Ubuntu's ROCm
7.1 packaging):

- `-sm layer`, 2 GPUs: works (1208 pp)
- `-sm tensor`, 1 GPU: works (676 pp)
- `-sm tensor`, 2 GPUs: **segfault**
- `-sm tensor`, 2 GPUs, `NCCL_P2P_DISABLE=1`: transport switches to
  `via SHM/direct`, **still segfaults**

AMD's own test suite fails the same way with llama.cpp out of the picture:

    UT_MIN_GPUS=2 rccl-UnitTests --gtest_filter='AllReduce.*'
    [ RUN      ] AllReduce.OutOfPlace
    [ ERROR    ] Child 0 pipe closed unexpectedly

Single-GPU AllReduce passes. So on mainline-kernel + distro-packaged ROCm, RCCL
is not an available fallback, which raises the value of a CPU-proxy path.

---

Full detail (exact commands, build SHAs, 72-run matrix, VRAM peaks, sweeps) is
in the repo; happy to paste any specific cell inline, re-run at different depths,
or test a revision of the branch.
