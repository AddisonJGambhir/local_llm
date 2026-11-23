# vLLM 0.28 (mixed-RDNA fork): Qwen3.8-27B across XTX + R9700 with TP=2

The full vLLM project, including fork patches, Docker images, bisect kit and records, lives in `~/Desktop/vllm`, which is a separate repo.
This folder keeps this machine's front end for it:

| File | What |
| --- | --- |
| `vllm_server.sh` | entry point used by `inference vllm` |
| `bin/vllm-launch` | preset menu (prose / code / json / edit decode, tg, pp, ctx per preset) |
| `bin/llamacompat-shim.py` | `:1234` front: `local` alias, llama.cpp-style `/props` with `n_ctx` and vision for Pi / Hermes |
| `run-qwen38-int4-r9700.sh`, `Dockerfile.quark-w4a16` | single-R9700 AWQ W4A16 serve (Quark fix overlay) |

| Preset | Speculation | Context | KV | prose / code / json / edit | mean | tg | pp |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| **13 (daily)** | DFlash2 k=4 PROB + skinny GEMM | 262K | bf16 | 65.5 / 100.5 / 112.3 / 92.2 | 92.6 | 67.8 | 1,249 |
| 12 | k=3 PROB | 262K | bf16 | 68.5 / 98.0 / 92.0 / 84.1 | 85.7 | 66.9 | 1,234 |
| 6 | none (graphs only) | 131K | bf16 | 45 across the board | 45.1 | 45.4 | 1,238 |

vLLM is the only engine here with real concurrency. The KV pool holds 293,873 tokens at 262K, shared by 8 slots
(8 subagents get about 36K each, and one request gets 262K when it has the box to itself).
