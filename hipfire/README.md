# hipfire 0.4.7: Qwen3.8-27B MQ4-XTS (XTX or R9700)

[hipfire](https://github.com/warpfront/hipfire) is a standalone HIP engine (its own HFQ weight format and daemon) with DFlash and MTP speculation.
It is built from `~/.hipfire/src` (tag v0.4.7) against ROCm 10 (`hipfire setup --gpu-arch gfx1100 --rocm-root /opt/rocm`). Models are in `~/.hipfire/models`.

## Modes (`hipfire` menu, launcher [`../bin/hipfire`](../bin/hipfire))

| Mode | GPU | Context | KV (K / V) | tg (BetterBench) | pp |
| --- | --- | ---: | --- | ---: | ---: |
| `dflash` | XTX | 168K | q8 / q8 | ~121 | 2,150-2,600 |
| **`dflash3`** (default) | XTX | 236K | fwht3 / q8 | ~125 | 2,140-2,460 |
| `dflash4` | XTX | 220K | fwht4 / q8 | ~125 | 2,150-2,480 |
| `mtp` | XTX | 156K | q8 / q8 | 93-110 (best at 8K+ context and on prose) | ~2,290 |
| `mtpmax` | XTX | 158K | q8 / q8 | MTP | ~10% lower |
| `maxctx` | XTX | 195K | q8 / q8 | 46-50, no speculation, no vision | |
| `r9700` | R9700 | 262K | fp8 / fp8 | ~119 | 4,600-4,900 |
| `r9700-fwht3` | R9700 | 262K | fwht3 / q8 | ~124 | 3,300-4,300 |

fwht3 / fwht4 = 3- / 4-bit Walsh-Hadamard-rotated keys with q8 values. They let a longer context fit at the same decode speed.

`local/proxy.py` is the :1234 front (installed at `~/.hipfire/local/proxy.py`). It maps `local` to the hipfire model id, adds `local` to `/v1/models`,
serves `/props` (`n_ctx`, vision), and logs prompt / cached / gen / first-token per request.

## Findings (XTX, 2026-11-19)
- DFlash block 16 wins short-context structured work (BetterBench 137.9 combined).
- MTP K=3 wins at 8K+ context and on prose. DFlash verify cost grows with context.
- No gain from: DDTree, the MQ6 draft, HIPFIRE_GRAPH, the gfx1100 dense gate-up flags, the f16 DFlash2 draft, or the ROCm 7.1 runtime.

`bin/hipfirectl`, `config/` and `docs/NOTES.md` are the original hipfire 0.3.0 tooling (port 11435) and are kept for reference.
