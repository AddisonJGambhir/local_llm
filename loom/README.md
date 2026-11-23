# Loom (AMD hrx-loom 0.2.0): Inkling-2-Mini

[AMD Loom](https://github.com/ROCm/hrx-system) (`pip install hrx-loom`) runs Inkling-2-Mini (26B MoE, 3.4B active,
Unsloth UD-Q4_K_XL) with the DSpark drafter. It is the fastest engine here for this model. llama.cpp manages 25-33 tok/s on it.

Packaging for other single-R9700 owners: **[loom-inkling2-mini-r9700](https://github.com/AddisonJGambhir/loom-inkling2-mini-r9700)**
(one-command install, same int8 build).

## Profiles (`loom` menu, launcher [`../bin/loom`](../bin/loom))

| Profile | GPU | Context | KV | Drafter | tg | pp | Vision |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `xtx` | XTX | 64K | f16 | DSpark depth 3 | 154 | 4,200-4,300 | - |
| `xtx-max` | XTX | 128K | f16 | none | 99 | - | - |
| `r9700` | R9700 | 288K | f16 | depth 3 | 118 | 4,100-4,350 | - |
| `r9700-max` | R9700 | 384K | f16 | none | 77 | - | - |
| **`r9700-vision`** | R9700 | **448K** | **int8** | depth 3 | **127** | 4,300-4,400 | yes |
| `r9700-vision-f16` | R9700 | 288K | f16 | depth 3 | 124 | 4,300-4,500 | yes, to ~253K |
| `vision` | XTX | 16K | f16 | DFlash depth 1 | - | - | yes (uncensored fine-tune) |

The layout is `:1234 proxy.py` (alias `local`, `/props`, retries) → `loom-serve :11437`. The `loom-backend` /
`loom-proxy` / `loom-watchdog` units run under `systemd-run --user` with io_uring blocked, so Loom keeps no
second copy of the weights in RAM. The R9700 profiles run with `--memory_bytes=32256mib` and the model is pinned, so an oversized
request is refused instead of evicting the model.

## Files

| File | What |
| --- | --- |
| `proxy.py` | OpenAI front for Loom. It handles the `local` alias, llama.cpp-style `/props` (`n_ctx` + `modalities.vision` for Pi), and message normalization. It replaces images in older turns with a placeholder so long chats stay cacheable. On "retained state leaves insufficient completion capacity" it frees the held history and retries once. |
| `loom-watchdog.sh` | Polls `/v1/status`. If Loom marks the model `failed`, it restarts the backend and re-pins (recovery takes about 14 s). |
| `port_rdna4.py` | Adds gfx12 (RDNA4) twins of the 4 flash2 attention kernels, plus per-target dispatch. The published package was written for gfx1151 and does not compile on the R9700. |
| `patch_sampler.py` | Sampled top-k: picks the top-k vocabulary chunks from the head's chunk maxima, then scans only k×128 logits. This raised no-drafter decode from 74 to 86 tok/s. |
| `kv8_patch.py` | **int8 paged KV**, q8_0-style (int8 + an f16 scale per 32 values). Each position and plane drops from 2048 to 1088 bytes. 11 kernels are touched; the sliding-window ring and the drafter stay f16. |
| `int8-kv-plan.md` | The original design notes for the int8 change |

## int8 KV results (R9700, 2026-11-23)

| test | f16 (288K card) | int8 (448K card) |
| --- | --- | --- |
| BetterBench decode | 124.1 | 126.9 |
| prefill 2K / 8K / 16K | 4,287 / 4,424 / 4,478 | 4,277 / 4,404 / 4,432 |
| 440K-token prompt | does not fit | 345 s to first token; needle found at 50% and 90% depth |
| 285K-token prompt | 653 s | - |
| image after 300K / 400K tokens of text | fails above ~253K | works (192 s / 303 s, correct descriptions) |

512K loads, but it sits at 32.4 GB VRAM and the answer did not finish, so 448K is the ceiling.

## Known limits (Loom 0.2.0)

- Requests that contain an image are never prefix-cached. An image turn re-processes the whole conversation. The proxy's
  old-image placeholder keeps the *following* text turns cached.
- One request is served at a time (`rows: 1`). More rows are possible but untested here.
- There is no multi-GPU split for one model. The XTX and the R9700 run separate instances.
