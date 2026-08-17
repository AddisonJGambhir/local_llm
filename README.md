# Local LLM Workstation

Local [llama.cpp](https://github.com/ggml-org/llama.cpp) inference on a dual-GPU AMD
workstation: five tuned models, OpenAI-compatible API on `127.0.0.1:1234`,
benchmark harness, and a versioned profile system. Everything here is measured,
not guessed, numbers below come from the 2026-08-16 harness run
(6.2k-token uncached prefill, 256-token decode at ~10k context depth, fixed seed,
Pi sampling: temp 0.6 / top_p 0.95 / top_k 20).

## Hardware

| GPU | Chip | VRAM | Role |
| --- | --- | ---: | --- |
| RX 7900 XTX | gfx1100 | 24 GiB | **Drives the desktop**, must stay under ~24.2 GB or the compositor spills to system RAM and the screen drops to ~3 fps |
| AI PRO R9700 | gfx1201 | 32 GiB | Compute partner |

CPU: Ryzen 9 9950X3D (16c/32t) · 64 GB DDR5 · Ubuntu 26.04

## The llama.cpp situation

| Tree | What it is |
| --- | --- |
| `llama.cpp/` | **Mainline, pinned at b10358, the only engine in use.** Two builds: `build/` (ROCm: `GGML_HIP`, `AMDGPU_TARGETS=gfx1100;gfx1201`, HIP graphs, MMQ/MFMA) and `build-vulkan/` (Vulkan). |
| `llama.cpp-next/` | Next-line build (ROCm, same targets) for testing incoming fixes before touching the pinned tree. |
| ~~`llama.cpp-beellama-032-hip/`~~ | Removed 2026-08-16. BeeLlama's one exclusive feature (KVarN KV) is superseded by f16 KV, and mainline measured within 0.4% of it on MTP decode. |

**Why b10358 is pinned:** b10450 measures ~5% slower (upstream `#26696` removed
`-funsafe-math-optimizations`). The upgrade path being considered: b10450 + a local
one-line patch restoring that flag. All current models are hybrid (SWA, SSM, or
recurrent), so speculative depth and KV precision **do not transfer between models:**
each preset was tuned for its own model/backend/topology.

## Quick start

```bash
llamaserver        # interactive launcher: pick backend/model/KV/spec/context
llamakill          # stop the server
harness            # pick a coding/agent front-end pointed at the local server

llamactl status    # what's running: resolved command + GPU residency + health
llamactl logs      # follow the rotating server log
llamactl command   # print the resolved server command WITHOUT launching
llamactl restart --profile config/profiles/<name>.sh   # start a specific preset
```

OpenAI-compatible API at `http://127.0.0.1:1234/v1`, model alias `local`.
Full operational detail: [`llama_server/README.md`](llama_server/README.md).

## Presets (shipped configurations)

All profiles in `llama_server/config/profiles/` (18 total: `default.sh` + 17 presets).
Start one with `llamactl restart --profile config/profiles/<file>`.
Pre t/s = prompt processing · Dec t/s = decode · Acc = draft acceptance.
¹ measured on the 2026-05 harness (pre-f16-KV era), re-measure before trusting.

### Single-GPU (RX 7900 XTX only)

| Profile | Model | Backend | KV | Spec | Ctx | Pre t/s | Dec t/s | Acc |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `default.sh` | 35B MoE Q8_0 | Vulkan | q8_0 | MTP n2 | 256k | N/A | ~94¹ | N/A |
| `preset-27b-q4km-q8-128k` | 27B MTP Q4_K_M | Vulkan | q8_0 | MTP n4 | 128k | N/A | ~76¹ | N/A |
| `preset-27b-q4km-rocm-q8-128k` | 27B MTP Q4_K_M | ROCm | q8_0 | MTP n2 | 128k | N/A | ~49¹ | N/A |
| `preset-36-27b-q4km-vulkan-xtx-only-128k` | 27B MTP Q4_K_M | Vulkan | q8_0 | MTP n2 | 128k | N/A | , | N/A |

`default.sh` is the canonical single-GPU fallback (q8_0 KV, ubatch 1024); the
xtx-only preset is for when the R9700 is needed elsewhere (batch 2048).

### Dual-GPU (ROCm + Vulkan)

| Profile | Model | Backend | KV | Spec | Ctx | Pre t/s | Dec t/s | Acc | XTX / R9700 GiB |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `preset-36-35b-a3b-q8-rocm-dual-256k-textonly` | **35B MoE Q8_0** | ROCm | f16 | MTP n2 | 256k | **3128** | **94.4** | 80.9% | 21.0 / 30.2 |
| `preset-36-35b-a3b-q8-rocm-dual-q8-256k-mmproj` | **35B MoE Q8_0** + vision | ROCm | f16 | MTP n2 | 256k | 3128 | 94.4 | 80.9% | 21.0 / 30.2 |
| `preset-36-35b-a3b-q8-vulkan-dual-256k-textonly` | 35B MoE Q8_0 | Vulkan | f16 | MTP n2 | 256k | 2072 | 93.7 | N/A | , |
| `preset-36-35b-a3b-q8-vulkan-dual-q8-256k-mmproj` | 35B MoE Q8_0 + vision | Vulkan | f16 | MTP n2 | 256k | 2072 | 93.7 | N/A | , |
| `preset-36-27b-q4km-rocm-dual-256k` | 27B MTP Q4_K_M | ROCm | f16 | MTP n2 | 256k | 1408 | 36.9 | 69.8% | 19.7 / 20.9 |
| `preset-36-27b-q4km-vulkan-dual-256k` | 27B MTP Q4_K_M | Vulkan | f16 | MTP n2 | 256k | N/A | , | N/A | , |
| `preset-38-27b-q8-rocm-dual-q8-256k-mmproj` | Qwen3.8 27B + vision | ROCm | f16 | MTP n3 | 256k | 982 | 38.5 | 42–72% | 22.5 / 29.7 |
| `preset-38-27b-q8-rocm-dual-256k-ppopt` | Qwen3.8 27B + vision | ROCm | q8_0 | MTP n3 | 256k | N/A | , | N/A | , |
| `preset-38-27b-q8-vulkan-dual-q8-256k-mmproj` | Qwen3.8 27B + vision | Vulkan | f16 | MTP n3 | 256k | 825 | 35.0 | 53.8% | 22.2 / 29.4 |
| `preset-muse-glimmer-30b-q8-rocm-dual-128k-dflash` | Muse Glimmer 30B | ROCm | f16 | DFlash n2 | 128k | 986 | 33.2 | 58.5% | 18.2 / 16.5 |
| `preset-muse-glimmer-30b-q8-rocm-dual-256k-yarn` | Muse Glimmer 30B | ROCm | f16 | DFlash n2 | 256k | 891 | 34.1 | N/A | 19.9 / 18.3 |
| `preset-muse-glimmer-30b-q8-vulkan-dual-128k-dflash` | Muse Glimmer 30B | Vulkan | f16 | DFlash n2 | 128k | 743 | 31.7 | 43.2% | 18.2 / 16.5 |
| `preset-nemotron35-30b-a3b-q8-rocm-dual-256k` | **Nemotron 3.5 30B-A3B** | ROCm | f16 | off ² | 256k | 3034 | 94.6 | n/a | 19.9 / 18.6 |
| `preset-nemotron35-30b-a3b-q8-rocm-dual-1m` | **Nemotron 3.5 30B-A3B** | ROCm | f16 | off ² | **1M** | 2555 | 94.7 | n/a | 19.9 / 29.4 |

² Nemotron's built-in MTP hangs ~37% of generations (upstream bug, independent of
KV/context/seed), speculation ships **off**; it's still ~2× the dense models.

**Shape of the table:** the two ~3B-active MoEs (35B-A3B, Nemotron 3.5) read far
fewer bytes per token than the dense 27/30B models, so they sit at 94–95 t/s while
everything dense sits at 32–39 t/s. Nemotron adds 1M context at no decode cost;
the 35B MoE adds vision and working speculation. The `mmproj` profiles pin the
vision tower to the R9700 (`MTMD_BACKEND_DEVICE`) to keep it off the constrained XTX;
their text stats are the text-only siblings'. `ppopt` is the pre-f16 q8_0
text-only/prefill variant, kept for maximum prefill when vision isn't needed.

## Models

| GGUF (`llama_server/models/`) | Size | Type | Role |
| --- | ---: | --- | --- |
| `Qwen3.6-35B-A3B-Q8_0` | 36 GB | MoE (3B active) + MTP | **Dual-GPU primary**, vision, working speculation |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0` | 33 GB | Mamba-2/attention MoE | **Dual-GPU, 1M context**, speculation off (upstream hang) |
| `Qwen3.8-27B-Q8_0` | 28 GB | Dense + MTP + vision | Dual-GPU, f16 KV, ts 28,37 |
| `Muse-Glimmer-30B-Q8_0` | 28 GB | Dense + DFlash sidecar | Dual-GPU, YaRN 256k variant |
| `Qwen3.6-27B-MTP-Q4_K_M` | 16 GB | Dense + MTP | Single-GPU |
| `dflash-kquant` + 2× `mmproj-*` | 3.3 GB | Muse drafter + vision projectors | Sidecars |

## Standing rules (measured, not guessed)

1. **f16 KV everywhere**, it's not a space/speed tradeoff, it *raises speculative
   acceptance* (quantized KV makes drafter and target disagree: Muse 43.6 → 58.5%,
   +19% decode). q8_0 only where f16 genuinely does not fit, and record why in the profile.
2. **XTX ceiling ~24.2 GB**, it drives the desktop; the compositor evicts to GTT
   above ~24.4 GB and the screen freezes. `llamactl status` reports per-process VRAM/GTT.
3. **Tensor splits are not a property of the model**, they depend on what else is
   resident on the XTX. The 35B shipped at `18,22`, moved to `16,24` when the desktop
   grew; re-check splits whenever the desktop footprint changes.
4. **Large batch is ROCm-only**, `--batch-size 6144–12288` is worth +13% prefill on
   ROCm and a measured no-op on Vulkan (confirmed on three models).
5. **Draft depth does not transfer**, per model × backend × topology. On this box
   Muse wants n_max 2 on both backends; upstream's "15" (single-GPU Q5_K_M era)
   measures *worst* (6.5% acceptance).

## Benchmarks & diagnostics

| Tool | What it finds |
| --- | --- |
| `llama-benchmark` | Full backend × model × spec × KV matrix (encode/decode t/s, VRAM) |
| `llama-benchmark-mtp` | MTP `n_max` 2→6 sweep; depth pinned by default (production); `--adaptive` tests adaptive-p |
| `llama-benchmark-ctx` | Max context before VRAM spill, per model, via binary search on fdinfo GTT |
| `llama-watch` / `llama-proxy` | Log watcher + TCP-tee postmortem of 4xx/5xx responses |

All require the GPU free (`llamakill` first) and stream CSV + per-run logs to
`llama_server/output/benchmarks/<stamp>/`.

## Docs

Deep operational reference: [`llama_server/README.md`](llama_server/README.md).
Findings and history in [`llama_server/docs/`](llama_server/docs/):
[2026-08-16 results](llama_server/docs/RESULTS_2026-08-16.md) ·
[2026-08-16 optimizations audit](llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md) ·
[speculation backend audit](llama_server/docs/QWEN36_SPECULATION_BACKEND_AUDIT.md) ·
[live tuning queue](llama_server/docs/TUNING_QUEUE.md) ·
[setup guide](llama_server/docs/SETUP.md).

## Layout

```text
llama_server/        the workstation: profiles, llamactl, benchmarks, diagnostics, models, docs
llama.cpp/           mainline b10358, ROCm (build/) + Vulkan (build-vulkan/)
llama.cpp-next/      next-line build for testing upcoming fixes
integrations/        MCP servers + front-end glue (agent_tools, searxng, …)
```
