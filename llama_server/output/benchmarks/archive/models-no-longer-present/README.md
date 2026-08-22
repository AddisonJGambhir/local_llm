# Benchmarks for models that are no longer on this box

Archived 2026-08-22. Every run in here measured a model that has since been
deleted, so **none of these numbers can be reproduced** without re-downloading
the model first. They are kept as history, not as reference.

## Models these runs used (all absent from `llama_server/models/`)

| model | note |
|---|---|
| `Qwen3.6-27B-MTP-Q4_K_M.gguf` | deleted 2026-08-22 along with its 6 profiles and 3 menu presets |
| `Qwen3.6-27B-IQ4_NL.gguf` | deleted earlier |
| `Qwen3.6-27B-DFlash-IQ4_XS.gguf` | DFlash sidecar; arch `dflash-draft`, which no current engine loads |
| `qwen36-35b-a3b-dflash-IQ4_XS.gguf` | same |
| `Qwen3.6-35B-A3B-MTP-UD-IQ4_NL.gguf` | deleted earlier |
| `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` | deleted earlier |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | deleted earlier |

The 35B MoE lives on as `Qwen3.6-35B-A3B-Q8_0.gguf`, a different quantization.
Numbers here do NOT transfer to it.

## What is in here

| directory | harness |
|---|---|
| `2026-06-21_00-41-12/` | full benchmark matrix |
| `ctx-headroom-*/` | context-headroom sweeps (6 runs, Jun-Aug 2026) |
| `spec-depth-*/` | speculative-depth sweeps (3 runs, Jun 2026) |

## Why the numbers here should not be quoted

Beyond the models being gone, these predate several corrections that changed
what the measurements mean. See `docs/HOW_TO_OPTIMIZE.md`:

- Decode measured at **empty KV** ranks configurations differently than decode at
  real depth. The backend ranking actually inverts with depth on q8_0 KV.
- **MTP throughput cannot rank configurations.** Draft acceptance correlates
  0.918 with it and swings 0.10-0.90 between identical runs.
- Peak VRAM read as **whole-card** rather than server-only is confounded by the
  desktop, which moves 1.7-3.2 GiB during a session.

Current instructions: [How to optimize](../../../docs/HOW_TO_OPTIMIZE.md)
