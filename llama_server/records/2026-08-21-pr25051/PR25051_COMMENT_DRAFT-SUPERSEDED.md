> **Historical record - not current operating guidance.**  
> Current instructions: [How to optimize](../../docs/HOW_TO_OPTIMIZE.md) | [Setup](../../docs/SETUP.md)

# TLDR comment for PR #25051. Paste into the PR after committing + pushing the results doc.

> REVIEW BEFORE POSTING. The link below only works after `git add + commit + push` (the doc is
> currently untracked in the repo). One-liner to verify post-push:
> curl -s -o /dev/null -w "%{http_code}" https://github.com/AddisonJGambhir/local_llm/blob/main/llama_server/docs/PR25051_AMD_DUAL_GPU_RESULTS.md

---

AMD data for this PR (you asked for non-NVidia testers):

**Hardware:** RX 7900 XTX 24 GB (drives the desktop, PCIe 5.0 x16) + AI PRO R9700 32 GB (PCIe 4.0 x4 chipset link), Ryzen 9 9950X3D, 64 GB RAM, Ubuntu 26.04, RADV. No P2P, no xGMI between the cards: every allreduce round-trips through system memory.

**Model:** Qwen3.8-27B (dense + SWA hybrid + built-in MTP), f16 KV. This PR's branch (HEAD 13ac90156) for the Vulkan tensor rows; b10358 for the ROCm baselines.

**The matrix, 7 of 8 cells measured.** pp = llama-bench prefill, depth stated per row (ctx 0 = empty context; d65k/d131k = marginal rate at that depth; "pp8192 @ c N" = N-token context allocated, KV empty). tg raw = unspeculated; tg MTP = live server, n3.

| Quant | Config (split) | Ship ctx | pp (depth) | tg raw | tg MTP |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | Vulkan layer 28/37 | 262k | 825 (6.2k shallow) | n/m | 35.0 |
| Q8_0 | **Vulkan tensor 50/50 (this PR)** | 200k | 946.7 (ctx 0) / 561.7 (d65k) / 324.1 (d200k) | 21.4 (ctx 0) | 40.3 |
| Q8_0 | ROCm layer 1/1 | 245k | 1099 (d0) / 509.4 (d65k) | 18.4 (d65k) | 34.4 |
| Q8_0 | ROCm tensor 45/55 | 245k | 275.2 (d65k) | 22.6 (d65k) | 44.8 |
| Q6_K | Vulkan layer | - | not measured | - | - |
| Q6_K | **Vulkan tensor 50/50 (this PR)** | 262k | 889.6 (ctx 0) / 882.5 (pp8192 @ c 262k) | 23.2 (ctx 0) | 43.4 |
| Q6_K | **Vulkan tensor 70/30 (this PR)** | 128k | 729.8 (ctx 0) / 709.1 (pp8192 @ c 131k) | 29.2 (ctx 0) | 54.5 |
| Q6_K | ROCm layer 38/27 | 245k | 565.9 (d65k) / 353.3 (d131k) | 20.0 / 18.1 | 32.2 |
| Q6_K | ROCm tensor 1/1 | 245k | 294.5 (d65k) | 27.3 (d65k) | 43.6 |

**What the matrix says:**
- At matched 65k depth (Q8_0), this PR's Vulkan tensor is 561.7 pp: 2.04x ROCm tensor (275.2) and +10% vs ROCm layer (509.4). Only config that beats both ROCm regimes on prefill at depth.
- Decode splits by quant. Q8_0: ROCm tensor keeps the MTP lead (44.8 vs 40.3). Q6_K: this PR leads both axes (raw 29.2 vs 27.3; MTP 54.5 @128k vs 43.6 @245k).
- Q6_K is what makes 262144 ctx fit at all (Q8_0 Vulkan tensor caps at 200704; Q6_K ROCm caps at 245760).
- Depths are mixed in the table because that is what was measured; the doc states the protocol for every row and lists the unmeasured cells explicitly (Q6_K Vulkan layer, and Q6_K Vulkan tensor at real depth).

**Two observations relevant to the review:**
1. RCCL on the ROCm side makes zero measurable difference here (RCCL ON vs OFF: same pp/tg within noise; RCCL=ON segfaults intermittently). Either it silently falls back to the generic path, or the bottleneck is the interconnect, not the collective.
2. The CPU-proxy F16 allreduce is far less penalized by this PCIe-only topology than the ROCm tensor path, even though both are CPU-routed.

Full reproducible detail (exact server command, builds with SHAs, full split/ubatch/ctx/MTP-depth sweeps, VRAM peaks, the GTT cliff at ts 55, topology notes): https://github.com/AddisonJGambhir/local_llm/blob/main/llama_server/docs/PR25051_AMD_DUAL_GPU_RESULTS.md

Happy to run the missing cells (Q6_K Vulkan tensor at d65k, Q8 Vulkan tensor tg at depth) or different depths if useful.
