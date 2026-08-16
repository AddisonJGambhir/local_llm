# KV Cache Quantization Comparison: TurboQuant, KIVI, KVQuant, and Atom (2024-2025)

## Overview of Methods

| Method | Venue | Primary Target | Key Innovation |
|---|---|---|---|
| **TurboQuant** | ICLR 2026 (arXiv:2504.19874), Google Research + NYU | KV cache only (post-RoPE) | Polar coordinate rotation + QJL 1-bit residual correction; data-oblivious vector quantization |
| **KIVI** | ICML 2024 (arXiv:2402.02750), Princeton/CMU | KV cache only | Per-channel Keys, per-token Values; asymmetric 2-bit with hardware-friendly implementation |
| **KVQuant** | NeurIPS 2024 (arXiv:2401.18079), Berkeley SqueezeAI Lab | KV cache only | Per-channel pre-RoPE Keys, non-uniform datatypes, outlier preservation; targets 10M context |
| **Atom** | MLSys 2024 (arXiv:2310.19102), UW-Madison/Berkley/Amazon | Weights + activations (incl. KV cache) | Mixed-precision fine-grained group quantization with dynamic per-layer bit allocation |

---

## 1. What Each Method Actually Compresses

### Compression Scope

**TurboQuant**: Compresses **both key and value caches simultaneously** as vectors using a two-stage vector quantization approach. The entire KV pair for each head-layer is treated as a set of high-dimensional vectors that are rotated (via PolarQuant) into a domain where scalar quantization is optimal, then corrected with QJL to restore unbiased inner-product estimation. Both K and V go through the same pipeline.

**KIVI**: Compresses **keys per-channel** and **values per-token**. The key insight from their element distribution analysis was that keys exhibit strong channel-wise outliers (requiring one scale factor per channel) while values are more uniformly distributed across tokens (one scale per token suffices). They use asymmetric 2-bit quantization with a configurable residual length R where recent tokens stay in higher precision.

**KVQuant**: Also **per-channel for Keys, per-token for Values**, but with an additional critical distinction: keys are quantized **before RoPE rotation** (pre-RoPE) to avoid amplifying quantization errors during positional encoding. KVQuant also introduces non-uniform datatypes across layers (some layers get 4-bit, others 2-bit based on sensitivity) and explicitly preserves numerical outliers in sparse higher-precision representations.

**Atom**: Broader scope -- compresses **both model weights AND activations (including KV cache)**. Atom uses mixed-precision group quantization where different groups of weights/activations receive different bit-widths based on their sensitivity. It combines per-group quantization with dynamic quantization for activations, meaning the actual precision can vary layer-by-layer and within layers. The fine-grained grouping (e.g., 8-element or 16-element groups) allows better accuracy than coarse per-tensor schemes.

### Quantization Axis Details

| Method | Keys | Values | Notes |
|---|---|---|---|
| TurboQuant | Per-coordinate after PolarQuant rotation | Per-coordinate after PolarQuant rotation | Unified vector quantization; no separate K/V treatment |
| KIVI | Per-channel (one scale per channel dim) | Per-token (one scale per token) | Asymmetric; residual length R for recent tokens |
| KVQuant | Per-channel, pre-RoPE | Per-token with non-uniform precision | Outlier preservation via sparse high-precision storage |
| Atom | Mixed-precision fine-grained group quantization | Dynamic per-token per-layer bit allocation | Group size typically 8-32; bit-width varies by sensitivity |

---

## 2. Reported Quality Degradation at 4-bit and Below

### TurboQuant

- **3.5 bits/channel**: Near-lossless performance on tested benchmarks (LongBench, Needle-in-a-Haystack)
- **~3 bits/channel**: Achieves "zero accuracy loss" relative to FP16 across their test suite; maintains perfect retrieval in Needle-in-a-Haydown even at 81% compression ratio
- At 3-bit (their target), the paper reports performance indistinguishable from FP16 on language modeling and long-context comprehension tasks
- The quality parity threshold is cited as ~3.5 bits per coordinate -- below this, small but measurable perplexity increases appear
- **No calibration data needed** -- fully data-oblivious; no per-model tuning

### KIVI

- **2-bit**: Reported "minimal impact on accuracy" on GSM8K, LongBench, and Needle-in-a-Haystack for Llama-7B/13B, Falcon, and Mistral
- Per-channel key quantization was found critical: dropping below per-channel for keys caused severe degradation at 2-bit because the scale factor mismatch overwhelmed the already-tight bit budget
- Residual length R > 0 is crucial: keeping just 64-128 recent tokens in FP16 while quantizing the rest to 2-bit dramatically improves GSM8K and reasoning task performance; with R = 0, quality drops significantly on complex tasks
- The paper notes that for very long-context generation tasks (128K+), 2-bit starts showing measurable perplexity increases on out-of-distribution contexts

### KVQuant

- **4-bit**: Sub-perplexity-loss reported across LLaMA-7B/13B and LLaMA-2 13B; handles context lengths up to 64K with "negligible" quality degradation
- **Sub-4-bit (down to ~2.5 bits)**: Claims "10 million context length" capability at ultra-low bitrates, but this involves aggressive mixed-precision strategies and outlier preservation rather than uniform quantization
- The pre-RoPE quantization strategy is cited as critical: quantizing after RoPE causes the rotated keys' large-magnitude components to be destroyed by low-bit quantization
- Non-uniform datatypes (4-bit for sensitive layers, 2-bit for robust ones) help keep average bitrate low while maintaining quality
- Outlier channels that exceed a threshold are preserved in higher precision (FP8/16), adding ~0.5-1 bits of overhead on average

### Atom

- **4-bit weights + activations**: "Negligible accuracy loss" across standard benchmarks; the paper shows near-identical perplexity to FP16 on Wikitext and LAMBADA
- Mixed-precision approach means some layers/heads stay at higher precision (FP16 or INT8) while heavily quantized others drop to 2-4 bits, yielding a per-model average of ~4-bit effective precision
- The fine-grained group approach (group size = 8 for weights, with sensitivity-based bit allocation) avoids the catastrophic quality drops seen in uniform low-bit schemes at extreme compression ratios
- However, Atom's primary evaluation targets **serving throughput** rather than long-context generation quality -- it is not specifically benchmarked on Needle-in-a-Haystack or LongBench-style tasks the way KIVI and KVQuant are

---

## 3. Throughput / Memory Gains on Consumer GPUs

### TurboQuant

- **6x KV cache memory reduction** at 3-bit (from FP16)
- Up to **8x faster attention computation** on NVIDIA H100 GPUs
- On consumer GPUs: At 128K context, a Llama 3.1 8B model's KV cache drops from ~16 GB FP16 to ~3 GB at 3-bit -- fitting comfortably into the 24 GB VRAM of an RTX 4090/7900 XTX
- The paper notes that rotation + QJL introduces compute overhead, but this is offset by the much smaller attention matrix (fewer elements to load from memory)
- **Notably beneficial for consumer GPUs**: Running 70B-class models on 24 GB cards becomes feasible because weight quantization alone often doesn't free enough room -- cache compression fills the gap at long contexts

### KIVI

- **2.6x less peak memory** (including model weights) in their benchmarking
- Up to **4x larger batch sizes** achievable from the freed memory
- **2.35x - 3.47x throughput improvement** on real LLM inference workloads
- The hardware-friendly asymmetric quantization maps directly to INT2-like operations available on modern GPU tensor cores
- On a consumer RTX 4090 (24 GB): enables fitting two 13B models in FP16 KV-cache bound scenarios where previously only one would fit

### KVQuant

- **6x KV cache compression** at 4-bit mode
- Enables context lengths up to **10 million tokens** in their most aggressive configuration
- A LLaMA-70B model reportedly runs on **8 GB RAM** with 4-bit KV quantization (notably not VRAM -- this suggests CPU inference is also supported)
- The custom kernel implementation minimizes overhead but requires PyTorch/CUDA integration; the speedup comes primarily from reduced memory bandwidth rather than faster compute
- On consumer GPUs: significant batch-size improvement, but the pre-RoPE + non-uniform dtype overhead means actual throughput gains are more modest (~2-3x) compared to KV cache memory savings

### Atom

- **7.73x end-to-end throughput** vs. FP16 (measured on A100 80GB GPUs)
- **2.53x improvement** over INT8 quantization baseline
- The primary gains come from leveraging **4-bit integer operators** on modern GPUs (NVIDIA Tensor Cores v3+ support INT4/GEMM natively) -- this is Atom's main differentiator vs. 8-bit schemes that must emulate INT4
- On consumer GPUs: the fine-grained mixed-precision adds kernel launch overhead that partially offsets gains; still, a 4x-5x throughput increase over FP16 on comparable hardware (A6000/RTX 4090) is expected for serving-heavy workloads

---

## 4. llama.cpp Support Status

### TurboQuant: IN PROGRESS (Active PRs)

- **Issue #20977**: Open tracking issue on ggml-org/llama.cpp requesting native TurboQuant support
- **PR #21089**: Active pull request introducing two new KV cache types:
  - `tbq3_0`: ~3.0625 bits per element (aggressive)
  - `tbq4_0`: ~4.0625 bits per element (conservative)
- Selectable via `-ctk` and `-ctv` flags (matching existing convention for KV cache type selection)
- **Not yet merged** as of April 2026; community fork (`TheTom/turboquant_plus`) provides a ready-to-build alternative
- Once merged, TurboQuant will work with Ollama, LM Studio, and other llama.cpp-based tools
- **Bottom line for consumers**: Available now via community forks or by building the PR locally; not yet in a stable release

### KIVI: NO NATIVE LLAMA.CPP SUPPORT

- No upstream llama.cpp issue or PR found for native KIVI support
- The closest existing mechanism in llama.cpp is `K_quants` (quantized key cache), which uses block-wise per-channel quantization somewhat similar in spirit to KIVI's per-channel keys, but not identical
- KIVI's asymmetric 2-bit scheme with residual length R would require custom kernel implementation in ggml
- **Workaround**: Use vLLM or the original PyTorch implementation from https://github.com/jy-yuan/KIVI for production use; llama.cpp users must accept existing quantization types

### KVQuant: NO NATIVE LLAMA.CPP SUPPORT

- No llama.cpp integration found. KVQuant's approach is tightly coupled to PyTorch and custom CUDA kernels (pre-RoPE quantization, non-uniform dtype scheduling)
- The framework requires integration into the attention kernel itself for pre-RoPE quantization to work efficiently
- Original implementation is PyTorch-only; no ggml backend exists
- **Workaround**: Use vLLM with KVQuant-style optimizations (vLLM has experimented with per-channel key quantization), or use the standalone PyTorch reference implementation

### Atom: PARTIAL SUPPORT VIA ROCm ATOM

- The most direct llama.cpp parallel is through AMD's **ROCm ATOM** framework, which provides optimized attention and execution backends for AMD Instinct GPUs
- **vLLM-ATOM plugin**: Available as an out-of-tree vLLM plugin; supports AiterBackend or AiterMLABackend with full vLLM compatibility
- This is AMD-specific and targets MI300/MI350 series GPUs (not consumer AMD cards like the 7900 XTX)
- The original Atom paper's serving optimizations are also implemented in **HuggingFace Transformers** via `AutoModelForCausalLM` with quantization configuration -- this works on any GPU through PyTorch
- No native ggml/llama.cpp integration exists; Atom is a serving-layer optimization framework rather than a low-level runtime library

---

## Summary Table

| Criterion | TurboQuant | KIVI | KVQuant | Atom |
|---|---|---|---|---|
| **Compresses** | K + V jointly (vector) | K per-channel, V per-token | K per-channel pre-RoPE, V per-token | Weights + activations + KV cache |
| **Target bitrate** | 3-4 bits/coord | 2-bit (+ residual) | 2-4 bits (mixed) | ~4-bit avg (mixed precision) |
| **Quality at target** | Near-lossless at 3.5+ bits | Minimal degradation at 2-bit + R>0 | Sub-perplexity loss at 4-bit | Negligible accuracy loss |
| **Calibration needed** | None (data-oblivious) | None (tuning-free) | Offline calibration for sensitivity mapping | Per-layer sensitivity analysis at deploy time |
| **Memory reduction** | 6x | 2.6x | 6x (4-bit) | Depends on layer distribution |
| **Throughput gain** | Up to 8x attention speedup | 2.35-3.47x | Modest (~2-3x) | 2.53-7.73x |
| **Consumer GPU fit** | Excellent (24 GB feasible for 70B at long ctx) | Very good (larger batches) | Good (batch size improvement) | Depends on operator support |
| **llama.cpp support** | In PR (#21089), fork available | None | None | ROCm ATOM only (AMD datacenter GPUs) |
| **Best for** | Long-context consumer GPU serving, RAG pipelines | Highest compression ratio, tuning-free deployments | Ultra-long context (millions of tokens), PyTorch stack | Serving throughput maximization on supported hardware |

---

## Recommendations for Consumer GPU Users (e.g., 7900 XTX, 24 GB)

1. **Best immediate option**: Monitor llama.cpp PR #21089 for TurboQuant merge -- this will bring the best consumer-gpu-compatible KV cache compression to ggml's ecosystem with minimal setup overhead.

2. **Workaround today**: Use vLLM (PyTorch-based) with either KIVI or KVQuant implementations for maximum VRAM savings, accepting that you cannot use llama.cpp's GGUF quantized models alongside these techniques without custom integration work.

3. **For ultra-long context** (>100K tokens): KVQuant's pre-RoPE + outlier preservation strategy is specifically designed for this scenario and may outperform TurboQuant at extreme lengths on PyTorch-based stacks.

4. **Note on ATOM**: The MLSys Atom paper's serving optimizations are most relevant in server-side deployments where throughput matters more than single-request latency. On a consumer GPU with VRAM constraints, the fine-grained mixed-precision overhead of Atom may not justify the complexity compared to TurboQuant's simpler two-stage approach once it lands in llama.cpp.