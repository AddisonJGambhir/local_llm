# Discussion #21043: RDNA4 Llama Experiments - Squeezing Every Token/s from the R9700

## OP BODY


50+ experiments over several days to find every optimization that matters for llama.cpp Vulkan on RDNA4. All benchmarks were run and verified manually on real hardware. Claude (Anthropic) assisted throughout — helping analyze results, suggest hypotheses for unexpected findings (like the PCIe ASPM discovery), and structure this document. Full results below.
 
## System Configuration
 
| Component | Details |
|-----------|---------|
| **GPU** | AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB GDDR6, 64 CUs) |
| **Memory bandwidth** | 640 GB/s (MCLK 1258 MHz, level 5/5 — verified during every test) |
| **PCIe** | PCIe 5.0 x16, 32 GT/s |
| **CPU** | AMD Ryzen 9 9900X 12-Core |
| **RAM** | 64 GB DDR5 |
| **OS** | Ubuntu 24.04.4 LTS |
| **Kernel** | 6.19.8-061908-generic |
| **Mesa (RADV)** | 25.2.8-0ubuntu0.24.04.1 |
| **AMDVLK** | Installed alongside RADV |
| **llama.cpp** | commit `dc8d14c58` (build 8554) |
| **Build** | `cmake -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` |
 
### Driver identification
 
RADV reports: `AMD Radeon AI PRO R9700 (RADV GFX1201) (radv)`
AMDVLK reports: `AMD Radeon AI PRO R9700 (AMD open-source driver)`
 
**All benchmarks use explicit `VK_ICD_FILENAMES` to guarantee driver selection.**
 
```bash
# RADV:
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json GGML_VK_VISIBLE_DEVICES=1
 
# AMDVLK:
VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json  # use -dev Vulkan1 for dGPU
```
 
---
 
## Models Tested
 
| Model | Type | Total Params | Active/Token | File Size | Quantization |
|-------|------|-------------:|-------------:|----------:|--------------|
| **Qwen3.5-35B-A3B** | MoE | 34.66B | ~3.5B | 18.32 GiB | UD-Q4_K_XL |
| **Qwen3.5-27B** | Dense | 26.90B | 26.90B | 15.58 GiB | Q4_K_M |
 
---
 
## Results: Qwen3.5-35B-A3B (MoE, 35B total, ~3.5B active)
 
### Decode
 
FA ON, 3 reps, values in tokens/s.
 
| Context | Stock RADV | Stock AMDVLK | RADV+ub2048 | RADV+rm_kq1+ub2048 | AMDVLK+rm_kq1 |
|--------:|-----------:|-------------:|------------:|--------------------:|---------------:|
| tg128 | 147.8 | 153.2 | 147.9 | **149.2** | **156.3** |
| tg512 | 146.4 | 151.4 | 146.0 | **148.7** | **155.3** |
| tg2048 | 143.9 | 148.4 | 144.4 | **146.8** | — |
 
### Prefill
 
FA ON, 3 reps, values in tokens/s.
 
| Prompt | Stock RADV | Stock AMDVLK | RADV+ub2048 | RADV+rm_kq1+ub2048 | FA OFF+ub2048 |
|-------:|-----------:|-------------:|------------:|--------------------:|--------------:|
| pp128 | 1,210 | 1,134 | 1,209 | 1,207 | — |
| pp512 | 2,404 | 1,831 | 2,400 | 2,400 | 2,398 |
| pp2048 | 2,381 | 1,823 | **3,074** | **3,075** | 2,983 |
| pp8192 | 2,262 | 1,742 | **2,983** | **2,984** | — |
 
---
 
## Results: Qwen3.5-27B (Dense, 27B)
 
### Decode
 
| Context | Stock RADV | Stock AMDVLK | RADV+ub2048 | RADV+rm_kq1+ub2048 | AMDVLK+rm_kq1 |
|--------:|-----------:|-------------:|------------:|--------------------:|---------------:|
| tg128 | 29.07 | 29.01 | 29.01 | **29.31** | **32.75** |
| tg512 | 29.08 | 29.15 | 29.05 | **29.32** | **32.82** |
| tg2048 | 28.97 | 28.96 | 28.98 | **29.25** | — |
 
### Prefill
 
| Prompt | Stock RADV | Stock AMDVLK | RADV+ub2048 | RADV+rm_kq1+ub2048 | FA OFF+ub2048 |
|-------:|-----------:|-------------:|------------:|--------------------:|--------------:|
| pp128 | 631 | 182 | 631 | 631 | — |
| pp512 | 798 | 203 | 799 | 800 | 795 |
| pp2048 | 799 | 202 | **823** | **823** | 806 |
| pp8192 | 772 | 199 | **797** | **797** | — |
 
---
 
## RADV vs AMDVLK
 
| Metric | RADV | AMDVLK | Winner |
|--------|-----:|-------:|--------|
| **35B MoE decode tg128** | 147.8 | **153.2** | AMDVLK +3.7% |
| **35B MoE prefill pp512** | **2,404** | 1,831 | RADV +31% |
| **35B MoE prefill pp2048** | **2,381** | 1,823 | RADV +31% |
| **27B Dense decode tg128** | 29.07 | 29.01 | Same |
| **27B Dense prefill pp512** | **798** | 203 | **RADV +293%** |
 
**RADV wins overall.** AMDVLK has a moderate decode advantage on MoE (+3.7%), but RADV's prefill is dramatically faster, especially on dense models where AMDVLK is nearly 4× slower.
 
---
 
## Optimization Impact (RADV)
 
| Optimization | 35B decode | 35B prefill pp2048 | 27B decode | 27B prefill pp2048 |
|---|---:|---:|---:|---:|
| Stock | 147.8 | 2,381 | 29.07 | 799 |
| **+ `-ub 2048`** | 147.9 (+0%) | **3,074 (+29%)** | 29.01 (+0%) | **823 (+3%)** |
| **+ `rm_kq=1`** | **149.2 (+1%)** | 3,075 (+0%) | **29.31 (+1%)** | 823 (+0%) |
| **+ FA ON** | — | +3% vs FA OFF | — | +2% vs FA OFF |
 
### `rm_kq=1` code change
 
One line in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
```cpp
uint32_t rm_kq = 1; // was 2; reduces VGPR pressure, improves RDNA4 occupancy
```
 
### AMDVLK + rm_kq=1 (surprise finding)
 
| Model | AMDVLK stock | AMDVLK+rm_kq1 | Improvement |
|---|---:|---:|---:|
| 35B MoE tg128 | 153.2 | **156.3** | **+2.0%** |
| 27B Dense tg128 | 29.01 | **32.75** | **+12.9%** |
 
`rm_kq=1` has a large effect on AMDVLK dense decode (+13%), much more than RADV (+1%). This suggests AMDVLK's LLPC compiler benefits more from reduced register pressure on RDNA4.
 
---
 
## Quality & VRAM Verification
 
### Qwen3.5-35B-A3B — WikiText-2 Perplexity
 
| Config | PPL | Model VRAM | KV VRAM | Compute VRAM | Total |
|--------|----:|----------:|---------:|-------------:|------:|
| Stock (rm_kq=2) | **6.9472 ± 0.046** | 18,492 MiB | 40 MiB | 498 MiB | **19,030 MiB** |
| rm_kq=1 | **6.9472 ± 0.046** | 18,492 MiB | 40 MiB | 498 MiB | **19,030 MiB** |
| rm_kq=1 + ub=2048 | **6.9472 ± 0.046** | 18,492 MiB | 40 MiB | 498 MiB | **19,030 MiB** |
 
**PPL and VRAM identical across all configurations.** No quality or memory impact from any optimization.
 
---
 
## Reproduction
 
```bash
# Build
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) -- llama-bench
 
# Verify setup
cat /sys/class/drm/card1/device/pp_dpm_mclk | grep "*"  # must show 1258Mhz
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
GGML_VK_VISIBLE_DEVICES=1 ./build/bin/llama-bench --list-devices  # must show "(RADV GFX1201)"
 
# Stock benchmark
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json GGML_VK_VISIBLE_DEVICES=1 \
./build/bin/llama-bench -m MODEL.gguf -t 1 -ngl 99 -fa 1 \
  -p 128,512,2048,8192 -n 128,512,2048 -r 3
 
# Optimized (add -ub 2048 for prefill boost)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json GGML_VK_VISIBLE_DEVICES=1 \
./build/bin/llama-bench -m MODEL.gguf -t 1 -ngl 99 -fa 1 \
  -p 128,512,2048,8192 -n 128,512,2048 -ub 2048 -b 16384 -r 3
```
 
---
 
## Exhaustive Flag Testing
 
### Qwen3.5-35B-A3B (MoE) — Decode tg128, rm_kq=1 active
 
#### RADV experiments
 
| Flag | tg128 | vs stock 149.5 |
|------|------:|--------:|
| Stock (rm_kq=1, no env) | **149.5** | — |
| + gfx queue | 149.2 | -0.2% |
| + nocompute | 149.3 | -0.1% |
| + gfx + nocompute | 149.4 | -0.1% |
| + disable coopmat | 149.3 | -0.1% |
| + force MMVQ | 149.5 | 0% |
| + disable int dot | 149.3 | -0.1% |
| + FA OFF | 149.3 | -0.1% |
| + disable F16 | 149.5 | 0% |
| + q8_0 KV | 148.8 | -0.5% |
| + disable graph opt | 148.6 | -0.6% |
| + bolist | 149.7 | +0.1% |
| + SAM | 148.8 | -0.5% |
| + disable fusion | **121.8** | **-18.5%** |
 
**gfx queue has zero effect on RADV 35B MoE decode.** Disable fusion catastrophically hurts.
 
#### AMDVLK experiments
 
| Flag | tg128 | vs stock 156.3 |
|------|------:|--------:|
| Stock (rm_kq=1, no env) | 156.3 | — |
| **+ gfx queue** | **163.7** | **+4.7%** |
| + gfx + memory_priority | 163.9 | +4.9% |
| + gfx + no-host | 163.9 | +4.9% |
| + gfx + nopo | **164.0** | **+4.9%** |
| + gfx + MMVQ disabled | **164.0** | **+4.9%** |
| + gfx + coopmat2 disabled | 163.7 | +4.7% |
| + gfx + F16 disabled | 163.9 | +4.9% |
| + gfx + disable int dot | 162.5 | +4.0% |
| + gfx + disable graph opt | 157.3 | +0.6% |
| + gfx + disable coopmat | 161.5 | +3.3% |
| + gfx + FA OFF | 162.6 | +4.0% |
| + gfx + q4_0 KV | 163.2 | +4.4% |
| + gfx + q8_0 KV | 163.2 | +4.4% |
| + gfx + host memory | 9.3 | -94.0% |
| + gfx + disable fusion | — | — |
 
**gfx queue gives +4.7% on AMDVLK 35B MoE.** No other flag breaks through 164 t/s.
 
---
 
### Qwen3.5-27B (Dense) — Decode tg128, rm_kq=1 active
 
#### RADV experiments
 
| Flag | tg128 | vs stock 29.30 |
|------|------:|--------:|
| Stock (rm_kq=1) | **29.30** | — |
| + gfx queue | 29.31 | 0% |
| + nocompute | 29.27 | -0.1% |
| + disable coopmat | 29.26 | -0.1% |
| + force MMVQ | 29.30 | 0% |
| + disable int dot | 29.32 | +0.1% |
| + FA OFF | 29.35 | +0.2% |
| + disable F16 | 29.35 | +0.2% |
| + q8_0 KV | 29.21 | -0.3% |
| + q4_0 KV | 29.19 | -0.4% |
| + disable graph opt | 28.99 | -1.1% |
| + disable fusion | **27.81** | **-5.1%** |
| + SAM | 29.27 | -0.1% |
 
**Nothing moves RADV 27B decode.** 29.3 t/s = hard BW ceiling (15.58 GiB × 29.3 = 456 GB/s = 71% of 640 GB/s).
 
#### AMDVLK experiments
 
| Flag | tg128 | vs stock 32.73 |
|------|------:|--------:|
| **Stock (rm_kq=1, NO gfx)** | **32.73** | **— (BEST!)** |
| + gfx queue | 30.07 | **-8.1%** |
| + gfx + disable coopmat | 29.93 | -8.6% |
| + gfx + force MMVQ | 30.03 | -8.2% |
| + gfx + disable int dot | 30.08 | -8.1% |
| + gfx + disable graph opt | 29.72 | -9.2% |
| + gfx + FA OFF | 30.06 | -8.2% |
| + gfx + q8_0 KV | 29.99 | -8.4% |
| + gfx + disable fusion | 28.61 | -12.6% |
| + gfx + memory_priority | 30.04 | -8.2% |
| rm_kq=2 (default, no gfx) | 28.97 | -11.5% |
 
**AMDVLK + rm_kq=1 without gfx = best dense decode (32.73 t/s, +13% over stock rm_kq=2!)**
**gfx queue HURTS dense AMDVLK by -8%** — opposite of MoE where it helps +4.7%.
 
---
 
### rm_kq impact across all configs
 
| Model | Driver | rm_kq=2 | rm_kq=1 | Improvement |
|-------|--------|--------:|--------:|------------:|
| 35B MoE | RADV | 147.8 | **149.5** | +1.1% |
| 35B MoE | AMDVLK (no gfx) | 153.2 | **156.3** | +2.0% |
| 35B MoE | AMDVLK+gfx | — | **163.7** | — |
| 27B Dense | RADV | 29.07 | **29.30** | +0.8% |
| **27B Dense** | **AMDVLK (no gfx)** | **28.97** | **32.73** | **+13.0%** |
 
`rm_kq=1` has the largest impact on **AMDVLK dense decode** (+13%). This suggests AMDVLK's LLPC compiler benefits significantly from reduced VGPR pressure on RDNA4 wave32 architecture. RADV's ACO compiler handles register allocation differently, gaining less from the same change.
 
---
 
## Best Achievable Performance
 
### 35B MoE
 
| | RADV | AMDVLK |
|---|---:|---:|
| **Best decode** | 149.5 (rm_kq=1) | **163.7 (gfx+rm_kq=1)** |
| **Best prefill pp2048** | **3,075 (ub=2048)** | 2,170 (gfx+ub=2048) |
 
### 27B Dense
 
| | RADV | AMDVLK |
|---|---:|---:|
| **Best decode** | **32.5 (rm_kq=1, ASPM perf)** | **33.2 (rm_kq=1, ASPM perf, NO gfx!)** |
| **Best prefill pp2048** | **993 (Mesa 25.3.6, ASPM, ub=2048)** | 207 (ub=2048) |
 
Dense decode improved by **+10.8% on RADV** and **+14.5% on AMDVLK** (vs stock rm_kq=2 + ASPM default) from combined `rm_kq=1` + PCIe ASPM performance mode.
 
### Key findings
 
- Decode is 100% memory bandwidth bound. No flag or parameter breaks through the ceiling.
- **`rm_kq=1` is the single most impactful code change**: +1% RADV, +2% AMDVLK MoE, **+13% AMDVLK dense**.
- **gfx queue helps AMDVLK MoE (+4.7%)** but **hurts AMDVLK dense (-8%)**. Zero effect on RADV.
- **Use gfx queue for MoE, disable for dense** when running AMDVLK.
- RADV is the best single driver (best prefill on all models, competitive decode).
 
---
 
## PCIe ASPM Discovery
 
Setting PCIe ASPM to performance mode eliminates L1 exit latency:
```bash
echo "performance" | sudo tee /sys/module/pcie_aspm/parameters/policy
```
 
| Model | Driver | ASPM default | ASPM perf | Change |
|-------|--------|------------:|-----------:|-------:|
| **27B Dense** | **RADV** | **29.30** | **32.46** | **+10.8%** |
| 27B Dense | AMDVLK | 32.73 | 33.17 | +1.3% |
| 35B MoE | RADV | 149.5 | 149.4 | 0% |
| 35B MoE | AMDVLK+gfx | 163.0 | 163.9 | +0.5% |
 
**ASPM L1 power saving adds latency to every PCIe transaction.** Dense models suffer most because they read the entire model (~15.6 GB) every token with many small transactions. MoE models batch work more efficiently, hiding PCIe latency.
 
This is a **system-level optimization** — no code change, no driver change. Persists until reboot. To make permanent: add `pcie_aspm.policy=performance` to kernel boot parameters.
 
---
 
## Known Issues
 
- **Kernel 6.19 RADV decode regression (theory):** ~10% slower than kernel 6.17.0-14. AMDVLK unaffected. Suspected root cause in amdgpu DRM scheduler, but not yet confirmed via bisect.
- **AMDVLK dense prefill significantly slower:** 4× slower than RADV on 27B dense (207 vs 823 t/s). Disabling coopmat (`GGML_VK_DISABLE_COOPMAT=1`) improves AMDVLK dense prefill by +17% (207→243) — suggests AMDVLK's cooperative matrix codegen is suboptimal for dense models. RADV's coopmat works correctly.
- **MCLK stuck on kernel 6.17:** MCLK won't boost to 1258 MHz on kernel 6.17.0-14/19. Works fine on 6.19.8.
- **PCIe ASPM default = powersave:** Resets on every reboot. Dense models lose 10% until set to performance.
 
---
 
## Exhaustive Experiment Log (50+ combinations tested)
 
### Parameters with REAL impact
 
| Discovery | Model | Driver | Gain | How |
|---|---|---|---:|---|
| **PCIe ASPM=performance** | Dense (all) | RADV | **+10.8%** decode | `echo performance > /sys/module/pcie_aspm/parameters/policy` |
| **PCIe ASPM=performance** | 30B MoE | RADV | **+10%** decode | same |
| **rm_kq=1** | Dense 27B | AMDVLK | **+13%** decode | 1 line in ggml-vulkan.cpp |
| **rm_kq=1** | MoE 35B | AMDVLK | +2% decode | same |
| **-ub 2048** | MoE 35B | RADV | **+29%** prefill pp2048 | `-ub 2048 -b 16384` |
| **Mesa 25.3.6** | MoE 35B | RADV | **+48%** prefill pp2048 | custom Mesa build |
| **Mesa 25.3.6** | Dense 27B | RADV | **+21%** prefill pp2048 | custom Mesa build |
| **gfx queue** | MoE 35B | AMDVLK | **+4.7%** decode | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` |
| **disable coopmat** | Dense 27B | AMDVLK | **+17%** prefill | `GGML_VK_DISABLE_COOPMAT=1` |
 
### Parameters with ZERO impact (all tested, all confirmed ±0.3%)
 
**RADV flags:** gfx queue (on RADV), RADV_DEBUG=nocompute, RADV_PERFTEST=sam/bolist/localbos/dmashaders/nircache/hic/nogttspill, RADV_PROFILE_PSTATE
 
**llama.cpp env vars:** GGML_VK_DISABLE(F16/BF16/COOPMAT2/INTEGER_DOT_PRODUCT/ASYNC/GRAPH_OPTIMIZE), GGML_VK_FORCE_MMVQ, GGML_VK_DISABLE_MMVQ, GGML_VK_DMMV_LARGE, GGML_VK_ENABLE_MEMORY_PRIORITY, GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM, GGML_VK_FORCE_MAX_ALLOCATION_SIZE, GGML_VK_FORCE_MAX_BUFFER_SIZE, GGML_VK_SUBALLOCATION_BLOCK_SIZE (16MB and 1GB)
 
**llama.cpp params:** -t 1/4/12 (thread count), --no-host, -nopo (no-op-offload), -dio (direct-io), -mmp 0 (no mmap), -sm row (split mode), -b 1/2/512 (batch size), --prio 2 (scheduling priority), -ctk/-ctv q8_0/q4_0 (KV cache quant)
 
**Code changes:** rm_stdq=2, rm_kq_int=2, rm_stdq_int=2, rm_kq=3/4
 
**System tuning:** hugepages (16GB), transparent hugepages=always, CPU pinning (taskset), nice -n -20, GPU power profile (COMPUTE/3D_FULL_SCREEN)
 
**DISABLE_FUSION is catastrophic:** -18.5% on MoE, -5.1% on dense. Never disable.
 
### Bandwidth utilization analysis
 
| Model | Best decode | Model size | BW used | BW peak | Utilization |
|-------|----------:|-----------:|--------:|--------:|------------:|
| 35B MoE (AMDVLK+gfx) | 163.8 t/s | ~2.4 GB/token | 393 GB/s | 640 GB/s | **61%** |
| 35B MoE (RADV) | 149.5 t/s | ~2.4 GB/token | 359 GB/s | 640 GB/s | **56%** |
| 27B Dense (AMDVLK+ASPM) | 33.2 t/s | 15.58 GB/token | 517 GB/s | 640 GB/s | **81%** |
| 27B Dense (RADV+ASPM) | 32.5 t/s | 15.58 GB/token | 506 GB/s | 640 GB/s | **79%** |
 
Dense models reach **79-83% BW utilization** with ASPM fix. MoE models are lower (56-61%) due to dispatch overhead from expert routing. The remaining 17-20% gap on dense is primarily from:
1. ACO compiler inefficiency (31 redundant `s_wait_kmcnt` per Q4K GEMV shader)
2. Cache line waste (Q4K 144B blocks in 192B cache lines = 75% utilization)
3. Memory controller overhead (DRAM refresh, row precharge)


 Please share your discoveries too — I'm curious what's the max we can get out of RDNA4.

## comment 1: zedbytes @ 2026-03-27

@JohnTDI-cpu thanks again, do you mind sharing huggingface links to the two models you used 

## comment 2: zedbytes @ 2026-03-27

@JohnTDI-cpu 

All my stats seem to perform better overall.  Also i'm running this on a  custom LACT R9700 Profile
power : 210W
GPU Clock Offset : -500 MHz
Maximum VRAM Clock : 2518 MHz
Minimum VRAM Clock : 194 MHz
GPU voltage Offset : -88 mV

Differences I can tell which may have mattered are 
GGML_VK_ALLOW_GRAPHICS_QUEUE=1
Mesa RADV 26.0.3 
llama.cpp 8555

## Models (GGUF)

| Role | File | Quant |
|------|------|-------|
| MoE 35B | [`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/JohnTdi/Qwen3.5-Unsloth-GGUF-R9700-Benchmark) | UD-Q4_K_XL |
| Dense 27B | [`Qwen3.5-27B-Q4_K_M.gguf`](https://huggingface.co/JohnTdi/Qwen3.5-Unsloth-GGUF-R9700-Benchmark) | Q4_K_M |
---
## System Configuration

| Component | Details |
|-----------|---------|
| GPU | 2× AMD Radeon AI PRO R9700 at PCIE gen 3 x8 ; AMD Radeon PRO W6600 at PCIE gen 3 x4 |
| Memory bandwidth | 640 GB/s (MCLK 1258 MHz, level 5/5) |
| PCIe | PCIe 3.0 x8/x8/x4 |
| CPU | Intel Core i9-9900K (8-core / 16-thread) |
| RAM | 64 GB DDR4 |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.19.8-061908-generic |
| Mesa (RADV) | Mesa 26.0.3 - kisak-mesa PPA (`mesa-vulkan-drivers`) |
| AMDVLK | Not installed |
| llama.cpp | commit `48cda24c1175363dd17925102dbe1da49279940e` (build 8555) |
| Build | `cmake -B build -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` |

---

## Test Configuration

### Environment

| Variable | Value |
|----------|--------|
| `GGML_VK_VISIBLE_DEVICES` | `1` — single R9700 (first PCIe slot) |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` | `1` |


### `llama-bench` flags (every run)

| Flag | Value |
|------|--------|
| `-t` / threads | `1` |
| `-ngl` | `99` |
| `-fa` | `1` (flash attention on) |
| `-p` | `128,512,2048,8192` (prefill sizes) |
| `-n` | `128,512,2048` (decode / generation lengths) |
| `-r` | `3` (repetitions) |

Not set: `VK_ICD_FILENAMES`

### Configs (columns in results)

| Column | Binary / batching |
|--------|-------------------|
| **Stock RADV** | Default `llama-bench` batching: `-b` **2048**, `-ub` **512** (no extra flags). `ggml-vulkan.cpp`: `rm_kq = 2` (upstream default). |
| **RADV+ub2048** | Same binary as stock; add **`-ub 2048`** and **`-b 16384`**. |
| **RADV+rm_kq1+ub2048** | Rebuild with **`uint32_t rm_kq = 1`** in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (line that defaults to `2`); same flags as **RADV+ub2048**. |

Backend is **Vulkan / RADV** via the cmake build (`GGML_VULKAN=ON`).

## Detailed results: Qwen3.5-35B-A3B (MoE)

**Decode** 

| Context | Stock RADV | RADV+ub2048 | RADV+rm_kq1+ub2048 |
|--------:|-----------:|------------:|-------------------:|
| tg128 | 153.4 | 153.1 | 155.6 |
| tg512 | 152.3 | 151.7 | 151.8 |
| tg2048 | 149.7 | 150.4 | 152.3 |

**Prefill**

| Prompt | Stock RADV | RADV+ub2048 | RADV+rm_kq1+ub2048 |
|-------:|-----------:|------------:|-------------------:|
| pp128 | 1839 | 1814 | 1742 |
| pp512 | 3314 | 3265 | 3257 |
| pp2048 | 3272 | 3964 | 3946 |
| pp8192 | 3131 | 3846 | 3832 |

## Detailed results: Qwen3.5-27B (Dense)

**Decode** 

| Context | Stock RADV | RADV+ub2048 | RADV+rm_kq1+ub2048 |
|--------:|-----------:|------------:|-------------------:|
| tg128 | 32.25 | 32.17 | 32.06 |
| tg512 | 32.27 | 32.21 | 32.06 |
| tg2048 | 32.09 | 32.09 | 31.89 |

**Prefill** 

| Prompt | Stock RADV | RADV+ub2048 | RADV+rm_kq1+ub2048 |
|-------:|-----------:|------------:|-------------------:|
| pp128 | 841 | 821 | 830 |
| pp512 | 942 | 914 | 921 |
| pp2048 | 933 | 923 | 930 |
| pp8192 | 883 | 880 | 890 |

build: `48cda24c1` (8555)

---

## Condensed comparison (three RADV configs)

All values are **t/s** from the detailed tables above. **RADV+ub2048** and **RADV+rm_kq1+ub2048** use **absolute t/s** with **Δ vs Stock RADV** in parentheses (percent, rounded).

| Model | Test | Stock RADV | RADV+ub2048 | RADV+rm_kq1+ub2048 |
|-------|------|------------|-------------|-------------------|
| MoE 35B | tg128 | 153.4 | 153.1 (−0.2%) | 155.6 (+1.4%) |
| MoE 35B | pp512 | 3314 | 3265 (−1.5%) | 3257 (−1.7%) |
| MoE 35B | pp2048 | 3272 | 3964 (+21.2%) | 3946 (+20.6%) |
| MoE 35B | pp8192 | 3131 | 3846 (+22.8%) | 3832 (+22.4%) |
| Dense 27B | tg128 | 32.25 | 32.17 (−0.2%) | 32.06 (−0.6%) |
| Dense 27B | pp512 | 942 | 914 (−3.0%) | 921 (−2.2%) |
| Dense 27B | pp2048 | 933 | 923 (−1.1%) | 930 (−0.3%) |
| Dense 27B | pp8192 | 883 | 880 (−0.3%) | 890 (+0.8%) |



## comment 3: digitalscream @ 2026-04-17

Here's something interesting, for those folk with dual R9700s - messing with the power profiles can make a huge difference to TG.

```
sudo su
echo "high" > /sys/class/drm/card1/device/power_dpm_force_performance_level
echo "high" > /sys/class/drm/card2/device/power_dpm_force_performance_level
```

The results are...interesting, to say the least.

Before:

```
| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |           pp512 |     1758.08 ± 117.77 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |          pp1024 |      2735.68 ± 46.49 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |          pp2048 |       3110.86 ± 1.91 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |           tg128 |         88.16 ± 1.19 |
```

After:

```
| model                          |       size |     params | backend    | ngl | type_k | type_v |     sm | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |           pp512 |       1817.02 ± 8.52 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |          pp1024 |       2291.08 ± 5.69 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |          pp2048 |       2589.24 ± 1.22 |
| qwen3next 80B.A3B Q4_K - Medium |  34.21 GiB |    60.33 B | Vulkan     |  99 |   q8_0 |   q8_0 |    row |  1 |           tg128 |        100.16 ± 0.24 |

```

So a 14% increase in TG, but a significant _decrease_ in PP. I don't have the tables to hand right now, but the change for Qwen3-30B-A3B-Q4_K_M was also interesting; 159t/s TG on a single card, 152t/s TG on two.

I'm not entirely sure how to explain that.

## comment 4: zedbytes @ 2026-05-14

anyone tried and tested the new qwen 3.6 27B and 35B optimisations like MTP , Turboquant and Dflash ? 

## comment 5: aguaishuo @ 2026-05-15

## AMD Radeon AI PRO R9700 (RDNA4, gfx1201) Benchmark & Optimization Results
## AMD Radeon AI PRO R9700 (RDNA4, gfx1201) 测试结果与优化经验分享

**Hardware / 硬件配置:**
- GPU: AMD Radeon AI PRO R9700 — 32GB GDDR6, 256-bit bus (~576 GB/s)
- CPU: AMD Ryzen 7 5700X
- OS: Ubuntu, llama.cpp Vulkan (RADV Mesa 26.0.3 in Docker)
- Model / 模型: Qwen3.6-27B-Q4_K_M

---

### Optimizations confirmed on RDNA4 / RDNA4上验证有效的优化项

All recommendations from this thread apply equally to gfx1201.  
本帖所有优化建议在 gfx1201 上同样适用：

| Optimization / 优化项 | Effect / 效果 |
|---|---|
| `echo performance > /sys/module/pcie_aspm/parameters/policy` | ✅ effective / 有效 |
| `echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level` | ✅ stable clocks / 稳定时钟 |
| `-b 16384 -ub 2048` | ✅ significant pp boost / 显著提升 pp |
| `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` | ✅ recommended — 7 ICDs present otherwise / 建议显式指定，否则系统有7个ICD同时加载 |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` — **do NOT set for dense models** | ⚠️ confirmed -8% on dense / dense模型下确认-8%，勿设 |

---

### Results — Vulkan + MTP (spec-draft-n-max=3, parallel=1)
### 测试结果 — Vulkan + MTP（spec-draft-n-max=3，parallel=1）

| Metric / 指标 | R9700 AI | 7900XTX (this thread) | Ratio / 比值 |
|---|---|---|---|
| Memory bandwidth / 内存带宽 | ~576 GB/s | ~960 GB/s | 60% |
| tg non-MTP base / 不开MTP基础速 | ~20 tok/s | ~29 tok/s | 69% |
| **tg with MTP / 开MTP生成速** | **44–48 tok/s** | **~81 tok/s** | ~57% |
| pp (945 tokens) | ~470–560 tok/s | ~823 tok/s (pp2048) | — |
| MTP acceptance rate / MTP接受率 | 95–97% | — | — |

---

### Key finding / 关键发现

**MTP provides a consistent ~2× speedup on RDNA4**, compensating significantly for the narrower memory bus.  
**MTP在RDNA4上稳定提供约2倍加速**，很大程度上弥补了内存总线较窄的劣势。

R9700's tg/bandwidth ratio is slightly *better* than 7900XTX (69% vs 60%), suggesting RDNA4 is marginally more compute-efficient per GB/s.  
R9700的 tg/带宽 比值略优于 7900XTX（69% vs 60%），说明RDNA4每GB/s的计算效率略高于RDNA3。

**Why R9700 is slower than 7900XTX for decode / 为什么R9700解码比7900XTX慢：**  
Despite being newer and marketed for AI (1531 TOPS INT4), the R9700 uses a **256-bit bus vs 384-bit on 7900XTX**. LLM token generation is memory-bandwidth bound — the AI accelerators help compute-bound tasks but not bandwidth-bound inference.  
尽管R9700是更新的AI加速卡（1531 TOPS INT4），但它使用的是**256-bit内存总线，而7900XTX是384-bit**。LLM的token生成是内存带宽瓶颈，AI加速单元对算力密集型任务有帮助，但对带宽受限的推理没有直接收益。

## comment 6: zedbytes @ 2026-05-18

My preliminary results testing MTP out , the best performance boost are actually for single GPU R9700 27B Q4 , and the one I actually use daily is dual GPU 35B Q8, gets a modest 6-7% boost at the expense of PP and TTFT

do you guys have similar results , any tips on making 35B get as much boost as 27B does ? 


## MTP Benchmarks

```
llama.cpp:   release b9203 
AMD GPU:     Navi 48 [Radeon AI PRO R9700] (kernel 6.19.8-061908-generic)
RADV:        mesa vulkan driver v26.1.0
```
### Single R9700 + Qwen3.6-27B-UD-Q4_K_XL  + KV Q4_0

#### Prompt Processing (PP)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| pp128 | 736.8 | 692.1 | 698.1 | 697.3 | 716.9 |
| pp2048 | 924.2 | 836.2 | 839.0 | 837.7 | 840.6 |
| pp8192 | 888.9 | 803.5 | 801.4 | 803.4 | 803.1 |
| pp16384 | 825.6 | 748.6 | 746.6 | 748.4 | 749.2 |

#### TTFT (seconds)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| pp128 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| pp2048 | 2.0 | 2.2 | 2.2 | 2.2 | 2.2 |
| pp8192 | 9.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| pp16384 | 19.6 | 21.6 | 21.7 | 21.7 | 21.6 |

#### Text Generation (TG)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| tg128 | 28.2 | 41.6 | 44.9 | 47.4 | 46.4 |
| tg512 | 29.6 | 44.3 | 51.7 | 51.2 | 55.5 |
| tg4096 | 29.9 | 44.1 | 50.6 | 48.3 | 47.1 |
| tg16384 | 29.8 | 44.6 | 50.4 | 49.5 | 46.0 |

#### TG Acceptance Rate

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| tg128 | - | 95.4% | 84.0% | 81.1% | 74.8% |
| tg512 | - | 88.9% | 82.8% | 72.1% | 76.6% |
| tg4096 | - | 86.2% | 77.8% | 64.6% | 59.4% |
| tg16384 | - | 87.3% | 77.6% | 67.9% | 58.3% |


### Dual R9700 + Qwen3.6-35B-A3B-UD-Q8_K_XL + KV F16

#### Prompt Processing (PP)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| pp128 | 1027.2 | 842.0 | 855.8 | 860.3 | 836.9 |
| pp2048 | 3195.7 | 2850.9 | 2864.4 | 2846.4 | 2849.7 |
| pp8192 | 4121.6 | 3009.2 | 3027.8 | 3024.6 | 3051.7 |
| pp16384 | 4700.5 | 2856.1 | 2860.3 | 2861.7 | 2874.3 |
| pp65536 | 3248.8 | 1916.9 | 1913.7 | 1919.7 | 1921.4 |

#### TTFT (seconds)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| pp128 | 0.2 | 0.3 | 0.3 | 0.3 | 0.3 |
| pp2048 | 0.6 | 0.7 | 0.7 | 0.7 | 0.7 |
| pp8192 | 1.9 | 2.7 | 2.6 | 2.6 | 2.6 |
| pp16384 | 3.5 | 5.7 | 5.7 | 5.7 | 5.6 |
| pp65536 | 20.1 | 34.1 | 34.2 | 34.1 | 34.0 |

#### Text Generation (TG)

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| tg128 | 72.8 | 76.7 | 83.9 | 82.8 | 71.0 |
| tg2048 | 83.2 | 89.9 | 87.5 | 87.8 | 77.9 |
| tg8192 | 83.7 | 89.2 | 91.1 | 82.2 | 77.1 |
| tg16384 | 83.6 | 89.5 | 88.2 | 84.7 | 80.2 |
| tg65536 | 83.5 | 88.9 | 89.3 | 87.8 | 75.6 |

#### TG Acceptance Rate

| Test | baseline | mtp-1 | mtp-2 | mtp-3 | mtp-4 |
|------|------|------|------|------|------|
| tg128 | - | 100.0% | 94.3% | 93.0% | 64.1% |
| tg2048 | - | 86.7% | 73.2% | 65.6% | 54.1% |
| tg8192 | - | 85.3% | 77.0% | 57.4% | 52.7% |
| tg16384 | - | 84.4% | 75.2% | 63.0% | 55.4% |
| tg65536 | - | 85.0% | 75.6% | 64.9% | 51.3% |




## comment 7: yiwiz-sai @ 2026-05-19

After a few days of testing, I would like to share some of my thoughts.

My system ubuntu 26.04，and I upgraded kernel to 7.0.0-15。

I tested vulkan and rocm, I think rocm is actully better when run Qwen3.6-27b (dense)

docker pull ghcr.io/ggml-org/llama.cpp:full-rocm --->it is b9209 when I tested

I run "amd-smi version" in container and check ROCm verion, and get：
AMDSMI Tool: 26.2.2+e1a6bc5663 | AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1 | amdgpu version: Linuxversion7.0.0-15-generic(buildd@lcy02-amd64-048)(x86_64-linux-gnu-gcc(Ubuntu15.2.0-16ubuntu1)15.2.0,GNUld(GNUBinutilsforUbuntu)2.46)
...

At first, the r9700 is optimized specifically for data types such as INT4, INT8, and FP8. Therefore—whether for model precision or KV cache precision—you'd better use Q4 or Q8. Do not use Q6, nor the mixed-precision UD-Q4_K_XL; otherwise, performance will suffer significantly (I have already tested a lot), particularly when using the ROCm driver.
https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html

I used unsloth/Qwen3.6-27B-MTP-GGUF  Qwen3.6-27B-Q8_0.gguf，I believe if you use Qwen3.6-27B-Q4_K_M.gguf will get better speed and context window.

Fortunately, 32GB of VRAM is just enough to accommodate a Q8 model combined with a q4_0 KV cache. However, you cannot set the parameters to `-ub 2048 -b 16384`, or you will run out of VRAM. Additionally, you cannot exceed two checkpoints (`-ctxcp 2`)，I awalys use 64k context window, which is sufficient for my needs.  **If you can accept 40k context window, you can adjust kv cache to q8_0.**

Now, see my test.（Qwen3.6-27B-Q8_0.gguf + q4_0 kv cache）

Request Concurrency=1
llama  | prompt eval time =    3817.17 ms /  **3666** tokens (    1.04 ms per token,   **960.40** tokens per second)
llama  |        eval time =   23372.95 ms /  1000 tokens (   23.37 ms per token,    **42.78** tokens per second)
llama  |       total time =   27190.12 ms / 4666 tokens

Request Concurrency=1
llama  | prompt eval time =   24630.81 ms / **22108** tokens (    1.11 ms per token,   **897.57** tokens per second)
llama  |        eval time =   28044.28 ms /  1000 tokens (   28.04 ms per token,    **35.66** tokens per second)
llama  |       total time =   52675.09 ms / 23108 tokens

Request Concurrency=1
llama  | prompt eval time =   63075.54 ms / **47679** tokens (    1.32 ms per token,   **755.90** tokens per second)
llama  |        eval time =   32893.92 ms /  1000 tokens (   32.89 ms per token,    **30.40** tokens per second)
llama  |       total time =   95969.46 ms / 48679 tokens

Request Concurrency=1
llama  | prompt eval time =   82972.57 ms / **58723** tokens (    1.41 ms per token,   **707.74** tokens per second)
llama  |        eval time =   35708.77 ms /  1000 tokens (   35.71 ms per token,    **28.00** tokens per second)
llama  |       total time =  118681.34 ms / 59723 tokens

The above is a single-request test for the Q8 model, which I consider sufficiently good. I'm too lazy to test Q4.

---------------------------
Below are the results of two concurrent tests:

Request Concurrency=2

thread1:
llama  | prompt eval time =    7155.37 ms /  3680 tokens (    1.94 ms per token,   514.30 tokens per second)
llama  |        eval time =   32801.60 ms /  1000 tokens (   32.80 ms per token,    30.49 tokens per second)
llama  |       total time =   39956.97 ms /  4680 tokens
**llama  | draft acceptance rate = 0.77667 (  699 accepted /   900 generated) ---> it is good !!!!!**

thread2:
llama  | prompt eval time =    6110.39 ms /  3680 tokens (    1.66 ms per token,   602.25 tokens per second)
llama  |        eval time =   50425.68 ms /  1000 tokens (   50.43 ms per token,    19.83 tokens per second)
llama  |       total time =   56536.08 ms /  4680 tokens
**llama  | draft acceptance rate = 0.31012 (  481 accepted /  1551 generated) --->very bad !!!!!**

As shown above, under concurrent conditions, the mtp for the second thread is extremely poor. I am not sure whether this is a bug. As pp continues to increase, tg deteriorates significantly, so I did not proceed with further testing.


---------------------------
Below is my Docker Compose configuration:
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:full-rocm
    #image: ghcr.io/ggml-org/llama.cpp:full-vulkan
    container_name: llama
    restart: unless-stopped
    shm_size: '16gb'
    ulimits:
      memlock:
        soft: -1
        hard: -1
    devices:
      - /dev/kfd
      - /dev/dri
    ports:
      - "8080:8080"
    volumes:
      - /home/yourname/ai/hf:/root/.cache/huggingface
      # - /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro
      # - /etc/vulkan/icd.d/amd_icd64.json:/etc/vulkan/icd.d/amd_icd64.json
    environment:
      - HF_HUB_OFFLINE=1
      # - VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
      # - VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json
      # - GGML_VK_VISIBLE_DEVICES=0
      # - GGML_VK_ALLOW_GRAPHICS_QUEUE=1 # do not set this
    command:
      # -s -cl, -s means "run llama-server", -cl means "list models"
      -s -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0 -fa on -ngl 99 -ctk q4_0 -ctv q4_0 -t 8 -kvu -rea off --spec-type draft-mtp --spec-draft-n-max 3 --draft-p-min 0.5 --temp 0.8 --top-k 20 --top-p 0.95 --min-p 0.01 --no-mmap -ctxcp 2 -c 65536 --host 0.0.0.0 --port 8080 --metrics -np 2 -dev ROCm0 --fit-target 32
      # -dev Vulkan1
      # --no-warmup
      # -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL
      # -tools all
	  

param refer to:
https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html

for kv cache type: as mentioned before, the speed of q4_0 is far better than q4_1, iq4_nl, q5_0, q5_1.  
**If you can accept 40k context window, you can adjust kv cache to q8_0.**

"-sm row" option will not impact speed obviously, "-sm layer" is default.

----------------------------
before reproducing, you should config your machine：

Step 1: Enable Resizable BAR in BIOS
Navigate to Advanced → PCI in your BIOS settings and set Resizable BAR to Enabled.

----------------------------
Step 2: Lock GPU to Highest Performance Level  **(maybe you don't need this step, see below "two key findings")**
There are two common approaches:

Method 1: Using rocm-smi (Recommend)
rocm-smi --setperflevel high
This is simple and automatically applies to all AMD GPUs.

Method 2: Directly via sysfs
echo 'high' > /sys/class/drm/cardXXX/device/power_dpm_force_performance_level
You need to identify the correct cardXXX.
List symlinks to find the device directory: ls -la /sys/class/drm/cardXXX/device
Find your GPU’s PCI address: lspci | grep -i amd
Match the PCI address with the symlink directory to determine the correct cardXXX.

Persistence:
Select either of the two commands above and add it to /etc/rc.local
Ensure the file starts with #!/bin/bash and is executable:  chmod +x /etc/rc.local
Verification after Reboot:
Run rocm-smi and confirm that the performance level is set to high.

----------------------------
Step 3: Enable ASPM
First, set ASPM to Enabled in your BIOS under Advanced → Platform Misc Configuration. Then, enable performance mode in the OS using one of the two methods below:

Method 1: Via Kernel Boot Parameters  (Recommend)
Edit /etc/default/grub and modify the GRUB_CMDLINE_LINUX_DEFAULT line.
Add the following parameter:
pcie_aspm.policy=performance
For this method, don't forget to run ./update-grub

Method 2: Using a Script
Log in and check the current policy:
cat /sys/module/pcie_aspm/parameters/policy
The default output is typically:
[default] performance powersave powersupersave
Switch to root and set the policy to performance:
sudo su root  
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
Verify the change:
The output should now show:
default [performance] powersave powersupersave
For this method, to make this persistent, you need to add command to /etc/rc.local.

Verification after Reboot:
cat /sys/module/pcie_aspm/parameters/policy

----------------------------
Step 4: Disable ECC for RX 9700 Series GPUs  (I'm not sure if this has any impact, but I did it anyway.)
Edit /etc/default/grub and modify GRUB_CMDLINE_LINUX_DEFAULT.

Add the following parameter:
amdgpu.ras_enable=0

Verification after Reboot:
cat /sys/module/amdgpu/parameters/ras_enable

-------------------------
Finally, my /etc/default/grub contains:
GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.runpm=0 pcie_aspm.policy=performance amdgpu.ras_enable=0"

then I run update-grub and verify that the kernel parameters appear correctly in /boot/grub/grub.cfg under the menu entries.
Remark:
The parameter amdgpu.runpm=0 is added to resolve GPU wake-up issues. On my system, without this setting, rocminfo cannot even detect the GPU.

-----------------------------


**Two key findings! (I only tested rocm)**

First:
I had previously set the `-c` parameter to 65536. When generating 2,000 TGs (with 60,000 PP), this occasionally resulted in VRAM overflow errors. I have since changed the setting to `-c 60000`; after several test runs generating 5,000 TGs, everything is now running smoothly.


**Second: 
Running `rocm-smi --setperflevel high` did not yield better results than `rocm-smi --setperflevel auto`—in fact, it performed slightly worse.**

The latter is more power-efficient (idle power consumption typically hovers around 20W in `auto` mode, compared to 50W in `high` mode), and I found that it consistently delivers superior performance during high-intensity computational tasks.

Shown below is the `auto` mode configuration: 60,000 PP + 5,000 TGs. As you can see, power consumption reached 298W, and the SCLK hit 3017 MHz.

> rocm-smi
=========================================== ROCm System Management Interface ===========================================
===================================================== Concise Info =====================================================
Device  Node  IDs              Temp    Power   Partitions          SCLK     MCLK     Fan     Perf  PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Edge)  (Avg)   (Mem, Compute, ID)
========================================================================================================================
0       1     0x7551,   23334  70.0°C  298.0W  N/A, N/A, 0         3017Mhz  1258Mhz  70.98%  auto  300.0W  99%    100%





## comment 8: d-shehu @ 2026-05-24

The OP deserves a special gold star. These 2 optimizations boosted performance for Qwen3 122B from 15 t/s to 48 t/s with 40 t/s achieved with just #1. And it worked even with RPC using 2 nodes.

(1) -ub 2048 | MoE 35B | RADV | +29% prefill pp2048 | -ub 2048 -b 16384
(2) GGML_VK_ALLOW_GRAPHICS_QUEUE=1

I saw significant performance improvements with other MOE models like Qwen 3 Next 80. I have not yet seen any boost with dense models but I just started testing.

Node 1:
Ubuntu 24.04 HWE
2x R9700 AMD GPU
RADV Vulkan (AMD installer) 
Llama.cpp b9222

Node 2:
Ubuntu 24.04 HWE
RTX 3090

Tyvm for this post! 

## comment 9: zedbytes @ 2026-07-22

anyone got any udpates on qwen 3.6 27b perhaps

## comment 10: d-shehu @ 2026-07-22

@zedbytes  Of the optimizations listed below, I can report that the latest mesa build did increase prefill by roughly the numbers shown below for both dense (Qwen 3.6 27b) and moe (Qwen 3.5). Likewise for me, I ran the llama-bench to optimize b and ub for your setup. All tested with llama.cpp release b10068.

The only boost I did not see is from setting pcie_aspm to performance as shown.  Possibly this is due to my motherboard/bios settings or the fact I have dual R9700 one each on a PCIe Gen4 x8 slot.  

---

Discovery | Model | Driver | Gain | How
-- | -- | -- | -- | --
PCIe ASPM=performance | Dense (all) | RADV | +10.8% decode | echo performance > /sys/module/pcie_aspm/parameters/policy
PCIe ASPM=performance | 30B MoE | RADV | +10% decode | same
-ub 2048 | MoE 35B | RADV | +29% prefill pp2048 | -ub 2048 -b 16384
Mesa 25.3.6 | MoE 35B | RADV | +48% prefill pp2048 | custom Mesa build
Mesa 25.3.6 | Dense 27B | RADV | +21% prefill pp2048 | custom Mesa build



## comment 11: joshuaca7971 @ 2026-07-23

Interesting discussion! It's always helpful to see people sharing ideas and solutions. For Android streaming app updates and guides, you can also visit [https://cinemaappbrasil.com/](https://cinemaappbrasil.com/).


## comment 12: amittell @ 2026-07-25

## A `-ub` dead band that silently wrecks Qwen3.5/3.6 numbers on RDNA4

Adding R9700 data, with one finding that I think invalidates a chunk of the
benchmarks in this thread including my own.

**Caveat on my setup, stated up front:** my R9700 is in a DEG2 dock behind a
**USB4 tunnel**, not a PCIe slot. Prefill lands where everyone else's does, but
decode is lower (26 t/s where slotted cards report 29-48), so treat my absolute
decode numbers as topology-limited. The finding below is a *relative* effect and
reproduces regardless.

### The finding

On Qwen3.6-27B (`qwen3_5`, hybrid GatedDeltaNet + full attention), setting
`n_ubatch` between roughly **65 and 256** collapses throughput. Outside that
band it is fine. llama.cpp b10107, Vulkan/RADV Mesa 26.0.3, `-ngl 99 -fa on`:

| `-ub` | prefill pp512 | decode tg128 |
|---|---|---|
| 64 | 457.7 | 25.4 |
| 65 | **16.2** | - |
| 68 | - | **2.09** |
| 128 | **17.1** | **2.09** |
| 256 | **19.3** | **2.08** |
| 288 | 653.3 | 25.3 |
| 512 | 774.4 | 25.3 |
| 2048 | 773.9 | 25.4 |

28x on prefill between 64 and 65. 40x between 65 and 288. 12x on decode.

Control on the same card, same sweep, gpt-oss-20b (non-hybrid MoE): 1323 /
1957 / 2803 / 3674 t/s at `-ub` 64/128/256/512. Clean monotonic scaling, no
dead band. So this is specific to the hybrid architecture, not the driver, not
the card, not the tunnel.

### Why it matters even if you never touch `-ub`

`llama-bench` sets `n_batch = n_ubatch = n_gen` for a `tg` test when `n_gen` is
smaller than the default. So a plain `llama-bench -n 128` puts you in the dead
band without asking:

| test | implied `n_ubatch` | reported |
|---|---|---|
| tg64 | 64 | 25.5 t/s |
| tg96 | 96 | 2.08 t/s |
| tg128 | 128 | 2.09 t/s |
| tg256 | 256 | 2.09 t/s |
| tg512 | 512 | 25.3 t/s |

Generating 128 tokens takes 60.7 s; generating 512 takes 20.2 s. Fewer tokens,
three times the wall clock.

**So any tg96/tg128/tg256 number posted for a Qwen3.5/3.6 hybrid is roughly 12x
too low**, and comparisons against a tg512 or against a non-hybrid model are
meaningless. My first pass at these benchmarks was wrong the same way, which is how I found it.

It also reframes the `-ub 2048` advice in the opening post: it is not only worth
+29% prefill, it keeps you out of a 40x hole. And `-ub 128` or `-ub 256`, which
are otherwise perfectly natural choices, are the worst thing you can pick.

### What it is not

- Not amortization or warmup. A fixed per-test cost cannot make 512 tokens
  finish in less absolute time than 128, and llama-bench hoists every cache
  clear and context setup outside its timed region.
- Not a stall: GPU busy is 95-99% in both regimes, so the slow path is doing
  real GPU work.
- Not paging or VRAM spill: zero disk reads in both, both resident (15.9 vs
  16.4 GiB), recurrent-state buffer on device (149.62 MiB, `Vulkan0`) in both.
- Not a graph fallback: identical graph node count (3655) and the fused
  GatedDeltaNet path reports enabled in both.

### Pointer for whoever wants to fix it

`GGML_VK_PERF_LOGGER=1` attributes it to `RMS_NORM`, whose per-row cost is flat
at 0.17 us up to 64 rows and flat at ~26 us above, a 157x step exactly at the
RDNA wave size of 64:

| RMS_NORM(5120, N) | us/call | us/row |
|---|---|---|
| 32 | 10.0 | 0.31 |
| 64 | 10.7 | 0.17 |
| 65 | 1710.5 | 26.3 |
| 128 | 3352.7 | 26.2 |
| 256 | 6839.4 | 26.7 |

I am flagging that as a lead, not a root cause: those absolute per-op numbers
do not reconcile with `-ub 2048` being fast end to end, so something about the
attribution is incomplete. The end-to-end table is the solid part.

Repro, no special build:

```
llama-bench -m Qwen3.6-27B-Q4_K_M.gguf -ngl 99 -fa on -p 512 -n 0 -b 16384 -ub 64
llama-bench -m Qwen3.6-27B-Q4_K_M.gguf -ngl 99 -fa on -p 512 -n 0 -b 16384 -ub 65
```

Happy to open a separate issue with the full data if that is more useful than a
discussion comment.

---

## Smaller items

**DPM `high` is 8-9% slower for prefill, and here is why.** Following up on the
report that forcing `power_dpm_force_performance_level=high` raised decode but
lowered prefill: the sclk DPM table on this card is only three entries deep.

```
$ cat /sys/class/drm/card1/device/pp_dpm_sclk
S: 0Mhz *
1: 500Mhz
2: 2350Mhz
```

`high` pins level 2 = **2350 MHz**, but under `auto` the card boosts to
**3348 MHz**, above the whole table. So "high" does not mean maximum, it means
*fixed*, and it disables opportunistic boost. Prefill is clock-bound and loses
8-9%; decode is bandwidth-bound and does not care. It does cut power from
139 W to 94 W, so it is a real efficiency lever, just not a performance one.

**ASPM did nothing on my topology.** `pcie_aspm.policy=performance` moved
prefill -0.8% (noise) on the tunnelled card, against +10.8% reported on
PCIe 5.0 x16 and no reproduction on Gen4 x8. Three topologies, one positive
result, so I would not treat it as general.

**Check your build before publishing hybrid numbers.** Same card, same model,
same flags: a 2026-07-06 build prefills at 17 t/s where b10107 does 799 t/s.
The hybrid path was slow before and is fixed now.

**eGPU users: `current_link_speed` lies.** My GPU reports `32.0 GT/s x16`,
which is the link to the bridge *inside* the dock. Every hop upstream is
2.5 GT/s x1-x4, and a warm 16 GiB model load measures under 1 GB/s end to end.
If you are on Thunderbolt or USB4, measure, do not read sysfs.

**One hazard worth knowing.** Loading a 22.4 GB MoE across the tunnel
reproducibly timed out the SDMA ring (`ring sdma0 timeout` ->
`Ring sdma0 reset failed` -> `GPU reset begin` -> MES stops answering). The
driver's own recovery wedged, VRAM stayed claimed, every hwmon sensor went
unreadable, and the host's `irq/N-pciehp` thread got stuck so the slot could
not re-enumerate. A dock re-plug did not clear it; only a reboot did. The
17 GB dense model never triggered it.

## comment 13: sswtodo @ 2026-07-27

BIOS -> PCIe ASPM -> Disable and it will sit in perf mode all the time.... otherwise change it through sysfs. Don't forget to disable Global C-State too.

So to get better performance you need:
PCIe Re-Size BAR : enabled
PCIe Above 4G decoding: enabled
PCIe ASPM: disabled
GLOBAL C-State: disabled

I have checked that <kbd>rm_kq = 1</kbd> and it didn't bring any advantages with single and dual RDNA3 gpus.


## comment 14: VectorAnvil @ 2026-08-05

# 2× R9700 / gfx1201 result: +12.47% tg128 across 3 launches, matched PPL

Tested the HIP direct-P2P AllReduce experiment on 2× Radeon AI PRO R9700 (`gfx1201`).

## Environment

- OS: Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
- ROCm: 7.2.0; HIP 7.2.26015-fc0010cf6a
- AMDSMI: 26.2.1
- llama.cpp commit/build: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f` / 9940
- patch repo commit: `f6f66106f2a960cec2da5a2f1a0f6d476989815f`
- build: Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1201`
- model: Qwen2.5-7B-Instruct Q6_K

The unpatched and patched binaries were built independently from the same upstream commit with identical CMake settings.

## llama-bench

Equivalent command shape:

```bash
HIP_VISIBLE_DEVICES=0,1 llama-bench \
  -m Qwen2.5-7B-Instruct-Q6_K.gguf \
  -p 512 -n 128 -b 2048 -ub 512 \
  -sm tensor -fa 1 -r 5
```

| Variant | pp512, mean ± sample SD | pp512 CV | tg128, mean ± sample SD | tg128 CV |
| --- | ---: | ---: | ---: | ---: |
| Matched unpatched | 2869.17 ± 2.04 t/s | 0.071% | 104.608 ± 0.245 t/s | 0.235% |
| Direct-P2P patched | 2870.34 ± 1.55 t/s | 0.054% | 117.657 ± 0.511 t/s | 0.434% |

- Mean decode change: **+12.47%**
- Mean prompt change: **+0.04%**
- All six launches completed successfully with no parse errors.

tg128 launch aggregates:

- Unpatched: `104.562309, 104.872658, 104.388528` t/s
- Patched: `118.019591, 117.072290, 117.878628` t/s

Each launch contains five internal llama-bench samples, giving 15 internal samples per build. The launch-aggregate tg128 ranges do not overlap: the patched minimum (`117.072290` t/s) is above the unpatched maximum (`104.872658` t/s).

The unpatched runs logged the HIP internal-AllReduce fallback to the meta-backend butterfly. The patched runs did not log that fallback.

## WikiText-2 perplexity

Command shape:

```bash
HIP_VISIBLE_DEVICES=0,1 llama-perplexity \
  -m Qwen2.5-7B-Instruct-Q6_K.gguf \
  -f wiki.test.raw \
  -sm tensor -fa on -c 512 -b 2048 -ub 1 \
  -ngl 99 -t 12 --fit off --chunks 16
```

| Variant | Split | PPL |
| --- | --- | ---: |
| Matched unpatched | tensor | `7.5520 ± 0.31781` |
| Direct-P2P patched | tensor | `7.5520 ± 0.31781` |
| Direct-P2P patched | none | `7.5560 ± 0.31789` |

The unpatched and patched tensor runs produced identical cumulative PPL values at every one of the 16 chunks. Patched tensor versus patched non-split differed by 0.0040 PPL (about 0.053%).

Dataset: Salesforce WikiText-2 raw test split pinned at revision `d2aad4e8c2e761345360926487e081c41c50d1e1`; reconstructed `wiki.test.raw` SHA-256 `7952b062817620ac99b47296dc945942b55068d0e85baf1f95f300c7ad1480ef`.

The performance figures above are descriptive results from three independent launches per build, not a formal statistical claim.

## comment 15: VectorAnvil @ 2026-08-05

# Follow-up: 2× R9700 / Huihui Qwen3.6-27B — +9.37% tg128 with HIP direct-P2P

Follow-up to my earlier 2× R9700 result: the first benchmark used Qwen2.5-7B Q6_K. That result remains a valid matched comparison, but it was not the model used in my intended deployment. I repeated the comparison with the directly relevant Huihui Qwen3.6-27B Q6_K model.

## Environment

- GPUs: 2× Radeon AI PRO R9700 (`gfx1201`)
- OS: Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
- ROCm: 7.2.0; HIP 7.2.26015-fc0010cf6a
- AMDSMI: 26.2.1
- llama.cpp commit/build: `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f` / 9940
- patch repo commit: `f6f66106f2a960cec2da5a2f1a0f6d476989815f`
- build: Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1201`
- model: `Huihui-Qwen3.6-27B-abliterated-ggml-model-Q6_K.gguf`
- reported architecture/size: qwen35 27B Q6_K, 27,320,697,856 parameters, 22,420,004,864 bytes loaded

The unpatched and patched binaries were built independently from the same upstream commit with identical CMake settings.

## llama-bench

Equivalent command shape:

```bash
HIP_VISIBLE_DEVICES=0,1 llama-bench \
  -m Huihui-Qwen3.6-27B-abliterated-ggml-model-Q6_K.gguf \
  -p 512 -n 128 -b 2048 -ub 512 \
  -sm tensor -fa 1 -r 5
```

Three independent launches were run per build. Each launch is llama-bench's aggregate over five internal samples, for 15 internal samples per build.

| Variant | pp512, mean ± sample SD | pp512 CV | tg128, mean ± sample SD | tg128 CV |
| --- | ---: | ---: | ---: | ---: |
| Matched unpatched | 812.215 ± 3.272 t/s | 0.403% | 33.552 ± 0.057 t/s | 0.171% |
| Direct-P2P patched | 820.486 ± 1.455 t/s | 0.177% | 36.695 ± 0.057 t/s | 0.155% |

- Mean decode change: **+9.37%**
- Mean prompt change: **+1.02%**
- All six launches completed successfully with no parse errors.

Launch aggregates:

- Unpatched pp512: `808.438130, 814.012222, 814.194721` t/s
- Patched pp512: `822.165317, 819.618791, 819.673175` t/s
- Unpatched tg128: `33.613977, 33.501248, 33.540290` t/s
- Patched tg128: `36.739477, 36.631137, 36.715313` t/s

The tg128 launch ranges do not overlap: the patched minimum (`36.631137` t/s) is above the unpatched maximum (`33.613977` t/s).

This GGUF contains an embedded MTP layer, but llama-bench does not exercise server-side MTP speculative decoding. These numbers isolate the direct-P2P change on the correct 27B model. I verified separately that the patched llama-server can load this model in tensor mode and perform MTP-assisted inference, but that server smoke test is not included in the throughput comparison above.

Queue IDs for the recorded Llama Lab evidence:

- matched smoke pair: `68a1357c-f589-40de-8d8c-fd752d3f836e`
- three-launch repeatability: `9974c344-558d-4f2e-9a99-20ca0812d508`

These are descriptive results from three independent launches per build, not a formal statistical claim.

## comment 16: amittell @ 2026-08-12

## Correction to my `-ub` dead band report: it's a 256 MB BAR, not RDNA4

I posted the `-ub` 65–256 collapse back in July and it's been folded into the summary as an RDNA4 characteristic. I've now root-caused it, and **I was wrong about the scope — it is not an RDNA4 or llama.cpp issue.** It's what a 32 GB card looks like through a 256 MB BAR. If you're running a slotted R9700, your `-ub 128`/`256` numbers are fine and I'm sorry for the noise.

My card is in a DEG2 dock behind a USB4 tunnel. Same card, same model, same llama.cpp build (`ebb546b7e`), same Mesa 26.0.3 — only the BAR differs:

| `-ub` | 256 MB BAR | 32 GB BAR |
|---|---|---|
| 192 | 326.71 | 617.08 |
| 224 | **18.94** | 637.45 |
| 256 | **19.54** | 761.55 |
| 288 | 400.22 | 693.10 |
| 512 | 891.84 | 893.65 |

`-ub 256` goes from 19.5 to 761 t/s — **39×**. The "dead band" simply doesn't exist once the aperture is whole.

### How the BAR gets lost

The kernel assigns the card a correct 32 GB BAR early in boot. Then the `thunderbolt` driver loads, drops the link, and pciehp re-adds the device — and the re-add sizes the bridge window to *current demand*:

```
[0.418219] pci 0000:02:00.0: bridge window [mem 0x7800000000-0x800fffffff 64bit pref]: assigned   ← 32 GB
[0.418245] pci 0000:05:00.0: BAR 0     [mem 0x7800000000-0x7fffffffff 64bit pref]: assigned       ← 32 GB
[2.150456] ACPI: bus type thunderbolt registered
[2.165482] pcieport 0000:00:01.1: pciehp: Slot(0): Link Down
[2.165489] pcieport 0000:00:01.1: pciehp: Slot(0): Card not present
...
[5.380555] amdgpu 0000:05:00.0: Not enough PCI address space for a large BAR.
[5.380576] amdgpu 0000:05:00.0: [drm] Detected VRAM RAM=32624M, BAR=256M
```

Window ends up 258 M — exactly `256M + 2M`, i.e. fitted to the BAR that had already fallen back.

### Diagnosing it from op timings

If you suspect this, the per-op profile is unambiguous. `GGML_VK_PERF_LOGGER=1`, in-band vs out-of-band (total µs):

| op | `-ub 256` | `-ub 288` | ratio |
|---|---|---|---|
| ADD | 4,605,550 | 16,141 | **285×** |
| MUL | 2,672,607 | 10,737 | 249× |
| GLU | 6,464,740 | 26,730 | 242× |
| RMS_NORM_MUL | 5,167,378 | 30,405 | 170× |
| MUL_MAT | 23,095,087 | 1,369,132 | **17×** |

*Everything* slows down, and the small elementwise ops worst — matmul is the *least* affected. That's fixed per-dispatch cost dominating, i.e. memory starvation. A broken shader would show one op exploding. This is also why my original "RMS_NORM at wave boundaries" guess was wrong: RMS_NORM runs in every model, and dense models were always fine.

### The fix, and a second bug on the way to it

```
modprobe.blacklist=thunderbolt amdgpu.rebar=0
```

`modprobe.blacklist=thunderbolt` stops the takeover, so the original 32 GB allocation survives (the tunnel is already enumerated by firmware; the card still works). **Only do this if you have no other USB4 devices** — it's a diagnostic, not a general fix.

`amdgpu.rebar=0` is needed because of a separate bug: with the BAR already large, amdgpu's ReBAR *renegotiation* wedges init —

```
amdgpu 0000:05:00.0: hw_init of IP block <gfx_v12_0> failed -110
amdgpu 0000:05:00.0: amdgpu_device_ip_init failed
amdgpu 0000:05:00.0: Fatal error during GPU init
```

It does **not** disable the large BAR — the BAR stays 32 GB (`Detected VRAM RAM=32624M, BAR=32768M`); it just stops amdgpu resizing what it already has. I isolated this by holding thunderbolt blacklisted and toggling only `rebar`.

Above 4G Decoding and Re-Size BAR Support were already enabled in my BIOS throughout, and `pci=realloc,hpmmioprefsize=32G` makes no difference here (`02:00.0` reports `HotPlug-`, so it never applies).

### Two other things worth knowing

**In-band runs can hang the GPU outright**, not just run slow — `ring comp_1.0.1 timeout` → `Ring reset failed`, needing a power cycle. Twice for me.

**The 22.4 GB MoE load failure I reported is the same root cause.** With a 256 MB aperture everything moves via SDMA, which is exactly the engine that timed out. It's not a size limit, it's the window.

Kernel-side reports (the window discard, and the ReBAR renegotiation) are going to `linux-usb`/`linux-pci` and AMD's tracker respectively — nothing here needs changing in llama.cpp.

**tl;dr for anyone on an eGPU:** check `lspci -vv | grep -A2 "Resizable BAR"`. If `current size` is 256 MB on a 32 GB card, that's your cliff, and it isn't RDNA4's fault.

## comment 17: VectorAnvil @ 2026-08-14

Following this RDNA4 work, I isolated a separate Q6_K prefill optimization on HIP; results and patch are here https://github.com/ggml-org/llama.cpp/discussions/27082

## comment 18: JohnTDI-cpu @ 2026-08-16

• Sorry for my long absence from this discussion. I recently returned to testing with the newly released Qwen3.8-27B and ran a new optimization sweep on a single Radeon AI PRO R9700.

  ## Qwen3.8-27B Q8_0 + MTP — single R9700

  ### Environment

  - GPU: AMD Radeon AI PRO R9700, 32 GB, RDNA4/gfx1201
  - OS: Linux Mint 22.3
  - Kernel: 7.0.0-28
  - Driver: Mesa RADV 25.2.8
  - llama.cpp: commit 0d9ceae1e
  - Build: Vulkan, Release, native CPU optimizations
  - Shaders compiled with Vulkan SDK 1.4.357.1
  - Model: unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
  - KV cache: Q4_0
  - GPU power state: auto
  - GPU also driving the Linux desktop

  One important note: compiling the Vulkan shaders with the old system glslc 2023.8 caused a severe performance regression. Rebuilding with Vulkan SDK 1.4.357.1 restored the expected performance.

  ### Baseline

  My initial configuration used:

  ctx-size=65536
  parallel=2
  ubatch=512
  spec-draft-n-max=5

  This produced approximately:

  59.2 tok/s

  ### Best configuration

  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
  export GGML_VK_ALLOW_GRAPHICS_QUEUE=1

  llama-server \
    --model Qwen3.8-27B-Q8_0.gguf \
    --device Vulkan0 \
    --n-gpu-layers 99 \
    --spec-draft-ngl all \
    --ctx-size 49152 \
    --parallel 3 \
    --batch-size 2048 \
    --ubatch-size 288 \
    --flash-attn on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 7 \
    --spec-draft-p-min 0

  This creates three 16,384-token slots. Reducing the per-slot context appears to be particularly beneficial for this hybrid Qwen architecture.

  ### Results

  Two independent 30-request runs, including a full server restart and VRAM cleanup:

   Run   #              Average TG#     Median TG#        Range #        Valid outputs
     
   Run 1                65.2 tok/s      65.2 tok/s       53.5–79.3           30/30
                
   Cold restart      64.7 tok/s       64.5 tok/s      53.0–78.6            30/30

  Combined average:

  64.95 tok/s

  This is approximately a 9.7% improvement over the 59.2 tok/s baseline. A shorter 10-request run reached 68.0 tok/s, but I consider the two longer runs more representative.

  The large range is expected because MTP performance depends heavily on generated content. Structured text and code generally achieve higher acceptance and throughput than less predictable text.

  ### What mattered

  - parallel=3 was faster than 1, 2, 4, or 8.
  - spec-draft-n-max=7 was the best point for this mixed workload.
  - ubatch=288 outperformed 64, 512, 1024, and 2048 for MTP decode.
  - GGML_VK_ALLOW_GRAPHICS_QUEUE=1 provided a small positive gain on RADV in this configuration.
  - Explicit --spec-draft-ngl all ensures the MTP head remains fully offloaded.
  - GPU power state auto remained faster than forcing high.

  I also tested forced/disabled MMVQ, different submit sizes, CPU thread counts, KV formats, and an experimental larger DMMV workgroup for AMD. None improved the final result; the larger workgroup patch was slower and was reverted.

  So, for this model on a single desktop-connected R9700, the repeatable result is currently approximately 65 tok/s with Q8_0 and MTP, with individual workloads ranging from roughly 53 to 79 tok/s.

