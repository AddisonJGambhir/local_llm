ggml_cuda_init: found 2 ROCm devices (Total VRAM: 57184 MiB):
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 1: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 32624 MiB
| model                          |       size |     params | backend    | ngl | n_ubatch |   main_gpu |     sm |  fa | dev          | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | ---------: | -----: | --: | ------------ | ------------ | --------------: | -------------------: |
| qwen35 27B Q6_K                |  24.13 GiB |    27.32 B | ROCm       | 999 |     1024 |          1 | tensor |   1 | ROCm0/ROCm1  | 70.00/30.00  | pp8192 @ d65536 |        275.45 ± 0.04 |
| qwen35 27B Q6_K                |  24.13 GiB |    27.32 B | ROCm       | 999 |     1024 |          1 | tensor |   1 | ROCm0/ROCm1  | 70.00/30.00  |  tg256 @ d65536 |         24.56 ± 0.27 |

build: 030ebb558 (10358)
