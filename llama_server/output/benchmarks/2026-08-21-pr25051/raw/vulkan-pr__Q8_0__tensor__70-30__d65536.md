WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 2 = AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | n_ubatch |   main_gpu |     sm |  fa | dev          | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | ---------: | -----: | --: | ------------ | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | Vulkan     | 999 |     1024 |          1 | tensor |   1 | Vulkan0/Vulkan1 | 70.00/30.00  | pp8192 @ d65536 |        206.85 ± 1.12 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | Vulkan     | 999 |     1024 |          1 | tensor |   1 | Vulkan0/Vulkan1 | 70.00/30.00  |  tg256 @ d65536 |         11.65 ± 0.01 |

build: 13ac90156 (9834)
