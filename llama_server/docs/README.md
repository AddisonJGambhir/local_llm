# llama_server docs

Three documents matter for operating this box. Everything else is evidence.

## Start here

| | |
|---|---|
| **[SETUP.md](SETUP.md)** | Install, build, run the server. Start here if the box is new to you. |
| **[HOW_TO_OPTIMIZE.md](HOW_TO_OPTIMIZE.md)** | Tune and benchmark **safely**. The VRAM ceiling, the guards, the measurement protocol, and the traps that have produced wrong answers here. Read section 0 before running any sweep. |

Launch presets with `llamaserver` (`bin/llama-launch`). The profile headers in
`config/profiles/` carry the measured numbers and the reasoning for every value;
they are the most specific documentation in the repo.

## Active work

Open questions, not yet settled.

- **[active/TUNING_QUEUE.md](active/TUNING_QUEUE.md)** - what to try next
- **[active/SUDO_BACKLOG.md](active/SUDO_BACKLOG.md)** - things blocked on root or a reboot, ranked by expected value
- **[active/AMD_DUAL_GPU_TUNING.md](active/AMD_DUAL_GPU_TUNING.md)** - what worked and what didn't across ~50 knobs, with upstream citations. Shareable.
- **[active/CACHE_INVALIDATION_INVESTIGATION.md](active/CACHE_INVALIDATION_INVESTIGATION.md)** - prompt-cache reprocessing

## Results and evidence

Dated, reproducible result sets. Read the summary, not the raw logs.

| where | what |
|---|---|
| `../records/2026-08-21-pr25051/` | PR #25051 Vulkan tensor-parallel: 72-run matrix, RCCL investigation, knob sweeps, correctness check |
| `../records/2026-08-16-dual-gpu/` | earlier dual-GPU optimization round |
| `../output/benchmarks/2026-08-21-pr25051/` | **raw evidence**: 83 llama-bench logs, `all_runs.csv`, `bench.jsonl`. Every number in the records traces back here. |
| `../archive/model-audits/` | superseded per-model audits (Qwen3.6 speculation, Muse Glimmer DFlash) |
| `../archive/kv-cache/` | superseded KV experiments (KVarN, KV quantization) |
| `../output/benchmarks/archive/models-no-longer-present/` | benchmark runs for models that have since been deleted. Not reproducible; kept as history. |

Everything under `records/` and `archive/` carries a banner marking it
historical. If a statement there contradicts `HOW_TO_OPTIMIZE.md`, the latter wins.

## The short version

Measured on this box (RX 7900 XTX + AI PRO R9700, no P2P):

- **Decode wants ROCm, tensor split, f16 KV.** `-ts 58/42` + `GGML_CUDA_DISABLE_GRAPHS=1`, ~29.6 t/s at 60k depth. ROCm leads Vulkan by ~12% at *every* depth on f16.
- **Prefill wants Vulkan, layer or row split.** ~72s to encode 71k tokens, versus 212s for the ROCm tensor config. Different job, different build.
- **Quantized KV is a capacity tool, not a speed tool.** On ROCm, f16 beats q8_0 by 13% at depth. The "Vulkan beats ROCm at depth" result is q8_0-specific.
- **Never rank configs on MTP throughput.** Draft acceptance correlates 0.918 with it and swings 0.10-0.90. Use raw decode; measure MTP once, at the end.
