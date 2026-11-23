# radiance 2.2.0: Qwen3.8-27B IQR-V4 on the R9700

[radiance](https://codeberg.org/StillDeadcode/radiance) is a C++/HIP engine built for RDNA4. It uses `.rad` model containers with the
DFlash2 drafter and the vision tower merged in. Here it serves Qwen3.8-27B IQ4R-S-V4 on the R9700 only (`ROCR_VISIBLE_DEVICES=1`).
radiance serves the OpenAI API and `/props` itself, so it needs no proxy. Each profile is a YAML file in `configs/`.

## Profiles (`radiance` menu, launcher [`../bin/radiance`](../bin/radiance))

| Profile | Model | Context | KV | tg | pp | Notes |
| --- | --- | ---: | --- | ---: | ---: | --- |
| **`q4s`** | IQ4R-S-V4 4.1 bpw (97.1% top-1) | 256K | int8 | 121-124 | 3,130-3,190 | default, DFlash2 depth 7, batch 4096 |
| `q4s-393k` | IQ4R-S-V4 + YaRN 1.5 | 384K | int8 | 127 | 3,040-3,160 | needle found at 370K (358 s) |
| `q4l` | IQ4R-L-V4 4.8 bpw (98.1%) | 256K | int8 | 98-111 | ~3,000 | |
| `q5` | IQ5R-V4 5.4 bpw (98.6%) | 256K | int8 | not measured | | best quality |

Every profile sets `vision-tower-placement: ram`, which puts the 879 MiB image encoder in pinned host RAM so the VRAM goes to the KV cache.

## Build

```bash
radiance/build-radiance-ffmpeg.sh   # Release, gfx1201, /opt/rocm clang, FFmpeg on (required for image input)
radiance/build-radiance.sh          # same without FFmpeg (text only), then the fast test suite on the R9700
```
The FFmpeg -dev headers are unpacked into `radiance/ffmpeg-dev/` from `apt-get download` debs, so no sudo is needed. Source goes in `radiance/src`
(git clone of upstream, tag v2.2.0). Models go in `radiance/models/` (HF `StillDeadcode/qwen3.8-27b-iqr-v4`).

## Tuning notes
- DFlash2 depth must be 1, 3, 7 or 15. Depth 7 is best (depth 3 = 81, depth 15 = 114).
- `max-num-batched-tokens` 8192 gains only 1-4% pp and costs 15% decode, so 4096 stays.
- `rad-tune` hangs on the `gemm_nt_q` M=2048 candidates. Tuning only `row_topk` and `attn_paged` gave noise-level gains.
- "cannot bind :1234" right after another engine stops is TIME_WAIT (radiance has no SO_REUSEADDR). The launcher waits it out.
