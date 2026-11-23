# Local LLM Workstation

Six local inference engines on one dual-GPU AMD desktop. Each engine is tuned for the models and the GPU it runs best on.
All six serve the same OpenAI-compatible endpoint, **`http://127.0.0.1:1234/v1`, model `local`**, so coding agents
(Pi, Hermes, Qwen Code, DeepSeek Harness) work with whichever engine is running.
Every number here was measured on this machine with [BetterBench](https://github.com/GGZ14/BetterBench) at temp 1.0 / top_p 0.95 / top_k 20,
thinking on, and 3 passes per category unless noted.

```bash
inference            # menu: shows what holds :1234, pick an engine, opens that engine's own profile menu
inference status     # which engine is serving 'local' + VRAM per GPU
inference stop       # stop whatever is serving :1234
inference loom       # jump straight to one engine's menu (llama | vllm | strata | hipfire | loom | radiance)
```

## Hardware

| Device | Chip | Memory | Link | Role |
| --- | --- | ---: | --- | --- |
| Radeon RX 7900 XTX | gfx1100 (RDNA3) | 24 GiB | CPU, PCIe 4.0 x16 | compute |
| Radeon AI PRO R9700 | gfx1201 (RDNA4) | 32 GiB | chipset, PCIe 4.0 x4 (7 GB/s) | compute |
| Ryzen iGPU | gfx1036 | shared | - | **drives the display**, so both dGPUs are free for models |

Ryzen 9 9950X3D (16c/32t), 64 GB DDR5, Ubuntu 26.04, ROCm 10.0 (`/opt/rocm`). Rules that hold for every engine:
- Weights live in VRAM, never in system RAM.
- Vision encoders go on the CPU (or in pinned RAM) wherever the engine allows it.
- Every launcher menu lists **context and KV-cache type** for each option.

## Engines at a glance

| Engine | Model | GPU(s) | Context | KV cache | Decode (tg) | Prefill (pp) | Vision |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| [**Loom**](loom/) (AMD hrx-loom 0.2.0) | Inkling-2-Mini UD-Q4_K_XL (26B MoE) | R9700 | **448K** (drafter) / **576K** (no drafter) | **int8** (our patch) | **127** / 94 | 4,300-4,400 | yes |
| | | XTX | 64K | f16 | 154 | 4,200-4,300 | text only |
| [**radiance**](radiance/) 2.2.0 | Qwen3.8-27B IQ4R-S-V4 | R9700 | 256K / 384K | int8 | 121-127 | 3,040-3,190 | yes (tower in RAM) |
| [**hipfire**](hipfire/) 0.4.7 | Qwen3.8-27B MQ4-XTS | XTX | 156-236K | q8 / fwht3 | 121-125 (DFlash) | 2,150-2,600 | yes |
| | | R9700 | 262K | fp8 / fwht3 | 119-124 | 3,300-4,900 | yes |
| [**Strata**](strata/) 0.2.19 | Qwen3.8-Flash-Next 125B UD-IQ4_XS (MoE) | XTX + R9700 | 256K | int8 | 110 | 986-1,963 | yes (CPU) |
| [**vLLM**](vllm/) 0.28 mixed-RDNA fork | Qwen3.8-27B AWQ-INT4 | XTX + R9700 (TP=2) | 262K | bf16 | 68 (92.6 mean) | 1,250 | yes |
| [**llama.cpp**](llama_server/) b10358+ | Qwen3.8-27B Q8_0 / UD-Q6_K_M | XTX + R9700 | 128-256K | f16 (presets), q8_0 / q5_1 (custom) | 34-51 | 509-947 | yes (CPU) |

Which engine to use:
- **Longest context:** Loom `r9700-max`: 576K with images (575K-token needle found, image after 500K tokens read correctly), 94 tok/s. `r9700-vision` holds 448K at 127 tok/s.
- **Fastest decode for code and agents:** Loom on the XTX (154 tok/s, 64K). For Qwen 27B, hipfire DFlash on the XTX.
- **Best prose with Qwen 27B:** hipfire MTP K=3, or radiance on the R9700.
- **Biggest model:** Strata. Qwen3.8-Flash-Next 125B MoE runs across both cards with experts streamed.
- **Concurrency (many subagents):** vLLM. It has 8 slots in a shared 294K-token KV pool. The other engines serve one request at a time.

Only one engine can hold :1234 at a time. `inference` asks before stopping the current one.
The XTX and the R9700 can also run separate engines on separate ports. For example, Loom on the XTX and radiance on the R9700 measured
no interference.

## Layout

| Path | What |
| --- | --- |
| `bin/` | `inference` (front door), plus the `loom`, `hipfire`, `radiance` launchers. The installed copies are in `~/.local/bin`. `harness-sync` refreshes every harness's OpenRouter catalog from the live API and keeps the local model labeled `local` (Pi, oh-my-pi, OpenCode, OpenClaw, Hermes, Prime, Qwen Code, DeepSeek Harness). |
| [`loom/`](loom/) | Loom proxy, watchdog, and the package patch scripts (RDNA4 port, sampler, **int8 KV**) |
| [`radiance/`](radiance/) | radiance build scripts (with FFmpeg for vision) and the serving configs |
| [`hipfire/`](hipfire/) | the :1234 proxy for hipfire, plus the older `hipfirectl` 0.3.0 tooling |
| [`strata/`](strata/) | Strata launcher and engine configs (the engine itself is upstream Niko1221/Strata) |
| [`vllm/`](vllm/) | vLLM launcher menu, llama.cpp-compat shim, R9700 serve script |
| [`llama_server/`](llama_server/) | llama.cpp launcher, presets, benchmark harness, records |
| `bench-scripts/` | needle tests, context-cap probes, vision long-context tests, BetterBench wrappers |
| `integrations/` | agent tools (local-llm MCP, filtered Firecrawl MCP, agent chatroom, SearXNG config) |

Engine source trees, model weights, venvs and builds are not in git (see `.gitignore`). Each engine README says where they come from.

## Related

- **[loom-inkling2-mini-r9700](https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700)**: the Loom R9700 setup packaged for other single-R9700 owners. It has a one-command install and the int8 KV build.
