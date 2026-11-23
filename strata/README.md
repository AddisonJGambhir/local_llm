# Strata 0.2.19: Qwen3.8-Flash-Next 125B MoE across both GPUs

[Strata](https://github.com/Niko1221/Strata) splits one large MoE model across the XTX and the R9700, with experts resident in VRAM and in pinned RAM.
The engine checkout is `~/Desktop/strata` (upstream) and the model data is `~/Desktop/Strata-data`. This folder holds this machine's launcher
(`bin/strata`, installed at `~/Desktop/strata/bin/strata` and `~/.local/bin/strata`) and the engine config.

| Config | Model | GPUs | Context | KV | tg | pp 2K / 8K / 16K | Vision |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `iq4xs` | UD-IQ4_XS (88.9% top-token match vs Q8_0) | XTX + R9700, split 26 | 256K | int8 | 110.3 | 986 / 1,740 / 1,963 | yes (CPU sidecar) |

`strata` = menu / start / stop / status / logs / watch (live pp/tg). It refuses to start if :1234 is busy, guards XTX VRAM and system RAM,
and restarts `hermes-gateway` when Hermes's context changes.

Lessons that cost a crash:
- Pinned RAM can't be reclaimed: a resident budget of 34 GiB OOM-killed the engine. 27 GiB is the stable maximum.
- The R9700's chipset x4 link starves an automatic layer split, so use a manual split.
- MTP `--spec 5` beats 4 and 6.
- Benchmark only on a quiet CPU. A parallel test run cut decode to about 44.
