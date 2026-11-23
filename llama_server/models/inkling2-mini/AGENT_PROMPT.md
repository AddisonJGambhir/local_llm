Goal: find the fastest engine + settings for Inkling-2-Mini (thinkingmachines, 26.4B total / 3.4B active MoE,
"looped" arch: 24 layers, middle 12 run 3x = 48 passes/token, 128 experts top-8, hybrid local/global attn,
native weights NVFP4 experts + MXFP8, 1M ctx, has mtp.safetensors + unsloth DSpark drafter). Maximize pp and tg
on my AMD box. Current numbers are far too slow for 3.4B active and I want them fixed.

## Hardware (Ubuntu 26.04, kernel 7.0, ROCm 10.0 at /opt/rocm; Ubuntu ROCm 7.1 under /usr)
- ROCm0 = RX 7900 XTX, gfx1100, 24 GB, drives no display. Real usable ~24.5 GB.
- ROCm1 = Radeon AI PRO R9700, gfx1201 (RDNA4, has FP8 WMMA), 32 GB, DRIVES THE DESKTOP (~3 GB used by
  the desktop; KFD "free" ignores that, so leave >=3 GB headroom). Sits in a chipset Gen4 x4 slot (7 GB/s),
  so cross-GPU splits pay a PCIe cost.
- CPU 16C Zen 5, 60 GB RAM (~48 GB free). Disk ~700 GB free.

## What exists
- Models: ~/Desktop/local_llm/llama_server/models/inkling2-mini/
  - Inkling-2-Mini-UD-Q8_K_XL.gguf (unsloth ROOT layout, needs unsloth's llama.cpp fork)
  - Inkling-2-Mini-UD-Q8_K_XL-mainline.gguf (same tensors, inkling2_loop layout for mainline llama.cpp)
  - thinkingmachines_Inkling-2-Mini-Q5_K_S.gguf (bartowski, 19.2 GB, fits XTX at 32K ctx: 23.96 GB used)
  - Inkling-2-Mini-MXFP4.gguf (bouroo, looped/mainline layout, 14.8 GB)
  - bb/ = BetterBench outputs
- Mainline llama.cpp afe77e1 (2026-11-21, includes inkling fix PR #32704): ~/Desktop/local_llm/llama.cpp-inkling
  - build-rocm (gfx1100;gfx1201, ROCm 10) done; build-vulkan in progress/done.
- Unsloth fork prebuilts (b12588-mix-c0c3de8, claims faster decode + DSpark support) DOWNLOADED but NOT
  extracted/run: ~/Desktop/local_llm/unsloth-llama-inkling/*.tar.gz (vulkan, rocm-gfx110X, rocm-gfx120X).
  Running them was blocked by the auto-mode permission classifier. ASK ME before running third-party binaries
  or building third-party forks; don't work around the block.
- vLLM checkouts: ~/Desktop/vllm, ~/Desktop/local_llm/vllm. Docker Desktop is installed (VM capped at 4 GB).
- BetterBench: ~/Desktop/local_llm/tools/BetterBench (.venv), config config/qwen-thinking-xtx.json
  (temp 1.0/top_p .95/top_k 20, prefill 2K/8K/16K).

## Baseline measured (mainline llama.cpp ROCm, -fa on -c 32768 -np 1, single stream, thinking on)
- R9700 alone, UD-Q8_K_XL: tg ~25 tok/s, GPU 97% busy, CPU not the bottleneck.
- XTX alone, Q5_K_S: tg ~33 tok/s.
- R9700 alone, MXFP4 (quick chat test): tg ~30 tok/s.
- pp not measured yet. Q8 compute buffer at ub 512 is ~3.4 GB. Expected tg for this size: ~100+ tok/s.
- In mainline llama.cpp the native FP4/FP8 MMA paths are NVIDIA-only (blackwell_mma_available,
  fp8_mma_hardware_available); AMD dequantizes NVFP4/MXFP4 through the generic MMQ/MMVQ path.

## What to investigate (rank by expected payoff, report numbers for each)
1. Why tg is ~4x below bandwidth-bound: profile (rocprofv3 / GGML_SCHED debug / llama-bench -v) to find the hot
   kernels; check whether the 48 looped passes cause launch overhead (HIP graphs on?), whether NVFP4
   per-expert scale ops run as separate kernels, MMVQ vs MMQ for the expert matmuls, -ub/-b, -fa, KV type.
2. Engines: mainline Vulkan (RADV) vs ROCm; unsloth fork (needs my OK); vLLM ROCm or SGLang ROCm with the
   original NVFP4/MXFP8 checkpoint or an FP8 checkpoint (real FP8 WMMA on the gfx1201 R9700; check
   whether NVFP4 MoE kernels exist for RDNA, else MXFP4 or FP8 variants); TokenSpeed; lemonade
   (~/Desktop/local_llm/lemonade); any upstream llama.cpp PRs/issues for inkling2 perf on AMD.
3. Speculative decoding: DSpark drafter (unsloth dspark-*.gguf, bouroo MXFP4 dspark), and the model's MTP head.
4. GPU combos once something is fast: XTX alone, R9700 alone, both (-sm layer both orders, -ts, -sm tensor,
   -sm row). Q8 does not fit the XTX alone; Q5_K_S does.

## Rules (important)
- Benchmark with BetterBench, not custom scripts: `betterbench run --endpoint http://127.0.0.1:PORT/v1
  --model inkling2-mini --config config/qwen-thinking-xtx.json --no-concurrency --max-model-len 32768`
  (--passes 5 --warmup 1 is fine for screening; record --note gpu=/quant=/engine=). Quick single-request
  smoke tests to check output coherence are fine. No 32K-depth tests.
- Use ports 8097/8098 for test servers. :1234 is my live server (Strata/hipfire): never stop or start it, and
  never kill servers you didn't start. Stop your own test servers when done.
- Do NOT touch GPU power/clock/profile settings.
- Check `uptime` before benches; other jobs on the CPU skew results. Don't run two benches on the same GPU at once.
- Put downloads in their own new dirs; verify sha256 against HF. Don't delete any of my existing models.
- `pkill -f`/`pgrep -f` with the pattern in the same shell line self-matches: use [b]racket patterns.
- Final deliverable: a table of engine x quant x GPU combo -> pp (2K/8K/16K) and tg (median), the winning
  launch command, and what was tried and didn't help.
