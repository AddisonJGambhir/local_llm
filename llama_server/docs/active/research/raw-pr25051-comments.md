# PR #25051 thread mining (inline)

Fetched 2026-08-2x via gh api. Full comment bodies below.

## pwilkin @ 2026-06-26 (upd 2026-06-26)

Did a few more tests with an A16 and a 4090:

### 4090 + A16

**-sm layer**
```
3.15.215.713 I slot print_timing: id  3 | task 0 | prompt eval time =   35434.16 ms / 28872 tokens (    1.23 ms per token,   814.81 tokens per second)
3.15.215.720 I slot print_timing: id  3 | task 0 |        eval time =  140477.32 ms /  1481 tokens (   94.85 ms per token,    10.54 tokens per second)
```

**-sm tensor**
```
3.25.475.691 I slot print_timing: id  3 | task 0 | prompt eval time =   60539.25 ms / 28872 tokens (    2.10 ms per token,   476.91 tokens per second)
3.25.475.696 I slot print_timing: id  3 | task 0 |        eval time =  126102.68 ms /  1504 tokens (   83.84 ms per token,    11.93 tokens per second)
```

### 2xA16

**-sm layer**
```
4.14.866.003 I slot print_timing: id  3 | task 0 | prompt eval time =   52667.78 ms / 28872 tokens (    1.82 ms per token,   548.19 tokens per second)
4.14.866.009 I slot print_timing: id  3 | task 0 |        eval time =  179656.12 ms /  1233 tokens (  145.71 ms per token,     6.86 tokens per second)
```

**-sm tensor**
```
3.49.116.708 I slot print_timing: id  3 | task 0 | prompt eval time =   57678.27 ms / 28872 tokens (    2.00 ms per token,   500.57 tokens per second)
3.49.116.712 I slot print_timing: id  3 | task 0 |        eval time =  155333.79 ms /  1800 tokens (   86.30 ms per token,    11.59 tokens per second)
```

So as you can see, the 4090 is held down by the A16 and the boost from tensor parallel there is really small, but on 2xA16, the TG boost is almost double while the PP loss is negligible (almost 90% of the original).

## characharm @ 2026-06-26 (upd 2026-06-26)

AMD Radeon RX 9070 XT  & AMD Radeon AI PRO R9700 
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q5_K - Medium       |  18.94 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 |  pp512 @ d60000 |        219.29 ± 1.65 |
| qwen35 27B Q5_K - Medium       |  18.94 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 |  tg128 @ d60000 |         10.68 ± 0.16 |


| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q5_K - Medium       |  18.94 GiB |    27.32 B | Vulkan     |  -1 |   1 |  pp512 @ d60000 |        493.22 ± 2.01 |
| qwen35 27B Q5_K - Medium       |  18.94 GiB |    27.32 B | Vulkan     |  -1 |   1 |  tg128 @ d60000 |         21.79 ± 0.09 |

## wizardeur @ 2026-06-26 (upd 2026-06-26)

## AMD RX7900XTX (2, 4, 8 GPUs)

ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           pp512 |        809.00 ± 2.70 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           tg128 |         34.50 ± 0.07 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           pp512 |       1283.40 ± 7.10 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           tg128 |         41.57 ± 0.37 |


ggml_vulkan: Found 4 Vulkan devices:
ggml_vulkan: 0 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 2 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 3 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           pp512 |        762.29 ± 5.09 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           tg128 |         29.66 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           pp512 |        305.86 ± 1.00 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           tg128 |         14.68 ± 0.90 |


ggml_vulkan: Found 8 Vulkan devices:
ggml_vulkan: 0 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 2 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 3 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 4 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 5 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 6 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 7 = Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           pp512 |        697.64 ± 3.23 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 |  layer |   1 |           tg128 |         11.89 ± 0.31 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           pp512 |         95.61 ± 0.10 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | Vulkan     |  -1 | tensor |   1 |           tg128 |          3.87 ± 1.57 |

## And ROCm for comparison:

ggml_cuda_init: found 2 ROCm devices (Total VRAM: 49120 MiB):
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 1: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           pp512 |       913.54 ± 26.89 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           tg128 |         26.34 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           pp512 |       1535.47 ± 2.33 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           tg128 |         45.35 ± 0.27 |

ggml_cuda_init: found 4 ROCm devices (Total VRAM: 98240 MiB):
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 1: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 2: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 3: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           pp512 |       877.60 ± 31.85 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           tg128 |         23.58 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           pp512 |      2085.89 ± 13.42 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           tg128 |         55.79 ± 0.41 |

ggml_cuda_init: found 8 ROCm devices (Total VRAM: 196480 MiB):
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 1: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 2: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 3: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 4: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 5: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 6: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
  Device 7: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32, VRAM: 24560 MiB
| model                          |       size |     params | backend    | ngl |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           pp512 |       811.63 ± 55.16 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 |  layer |   1 |           tg128 |         21.48 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           pp512 |      2163.08 ± 23.29 |
| qwen35 27B Q4_K - Medium       |  15.48 GiB |    26.90 B | ROCm       |  -1 | tensor |   1 |           tg128 |         37.54 ± 2.63 |

## digitalscream @ 2026-06-27 (upd 2026-06-27)

OK, bit of oddness on my R9700s compared with the 7900XTX above - prefill takes a significant hit compared with `-sm row`:

```
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |    row |   1 |          pp2048 |       1580.58 ± 5.18 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |    row |   1 |          pp4096 |       1556.35 ± 2.32 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |    row |   1 |           tg128 |         29.91 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp2048 |       1125.71 ± 2.82 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp4096 |       1110.65 ± 1.31 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |           tg128 |         36.84 ± 0.38 |
```

Also, even more weirdness - when run as a server, the first request goes through at 36t/s, but the second falls off to 2t/s:

```
0.21.379.838 I srv  proxy_reques: proxying request to model nondraft_tensor_Qwen3.6-27B-Q4_0.gguf on port 47447
[47447] 0.08.608.257 I srv    operator(): Chat format: peg-native
[47447] 0.08.608.379 I slot get_availabl: id  3 | task -1 | selected slot by LRU, t_last = -1
[47447] 0.08.608.381 I srv  get_availabl: updating prompt cache
[47447] 0.08.608.385 I srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
[47447] 0.08.608.388 I srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 16384.000 MiB, 256000 tokens, 17179869184 est)
[47447] 0.08.608.389 I srv  get_availabl: prompt cache update took 0.01 ms
[47447] 0.08.608.423 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0
[47447] 0.08.608.425 I slot process_sing: id  0 | task -1 | saving idle slot to prompt cache
[47447] 0.08.608.426 I slot prompt_clear: id  0 | task -1 | clearing prompt with 0 tokens
[47447] 0.08.608.578 I slot process_sing: id  1 | task -1 | saving idle slot to prompt cache
[47447] 0.08.608.579 I slot prompt_clear: id  1 | task -1 | clearing prompt with 0 tokens
[47447] 0.08.608.723 I slot process_sing: id  2 | task -1 | saving idle slot to prompt cache
[47447] 0.08.608.725 I slot prompt_clear: id  2 | task -1 | clearing prompt with 0 tokens
[47447] 0.08.609.083 I srv  stream_sessi: stream_session_attach_pipe: conv_id= (empty=1)
[47447] 0.08.842.162 I slot create_check: id  3 | task 0 | created context checkpoint 1 of 32 (pos_min = 14, pos_max = 14, n_tokens = 15, size = 149.626 MiB)
[47447] 0.11.928.914 I slot print_timing: id  3 | task 0 | n_decoded =    108, tg =  35.85 t/s, tg_3s =  35.85 t/s
[47447] 0.14.931.024 I slot print_timing: id  3 | task 0 | n_decoded =    217, tg =  36.08 t/s, tg_3s =  36.31 t/s
[47447] 0.17.940.243 I slot print_timing: id  3 | task 0 | n_decoded =    325, tg =  36.02 t/s, tg_3s =  35.89 t/s
[47447] 0.20.946.915 I slot print_timing: id  3 | task 0 | n_decoded =    434, tg =  36.08 t/s, tg_3s =  36.25 t/s
[47447] 0.23.948.536 I slot print_timing: id  3 | task 0 | n_decoded =    542, tg =  36.06 t/s, tg_3s =  35.98 t/s
[47447] 0.26.957.662 I slot print_timing: id  3 | task 0 | n_decoded =    652, tg =  36.14 t/s, tg_3s =  36.56 t/s
[47447] 0.29.983.478 I slot print_timing: id  3 | task 0 | n_decoded =    761, tg =  36.12 t/s, tg_3s =  36.02 t/s
[47447] 0.33.010.837 I slot print_timing: id  3 | task 0 | n_decoded =    870, tg =  36.11 t/s, tg_3s =  36.00 t/s
[47447] 0.36.017.438 I slot print_timing: id  3 | task 0 | n_decoded =    979, tg =  36.12 t/s, tg_3s =  36.25 t/s
[47447] 0.39.044.366 I slot print_timing: id  3 | task 0 | n_decoded =   1087, tg =  36.08 t/s, tg_3s =  35.68 t/s
[47447] 0.40.202.434 I slot print_timing: id  3 | task 0 | prompt eval time =     307.57 ms /    19 tokens (   16.19 ms per token,    61.78 tokens per second)
[47447] 0.40.202.437 I slot print_timing: id  3 | task 0 |        eval time =   31285.98 ms /  1129 tokens (   27.71 ms per token,    36.09 tokens per second)
[47447] 0.40.202.438 I slot print_timing: id  3 | task 0 |       total time =   31593.55 ms /  1148 tokens
[47447] 0.40.202.441 I slot print_timing: id  3 | task 0 |    graphs reused =       1124
[47447] 0.40.202.474 I slot      release: id  3 | task 0 | stop processing: n_tokens = 1147, truncated = 0
[47447] 0.40.202.481 I srv  update_slots: all slots are idle
0.53.000.387 I srv  proxy_reques: proxying request to model nondraft_tensor_Qwen3.6-27B-Q4_0.gguf on port 47447
[47447] 0.40.231.788 I srv    operator(): Chat format: peg-native
[47447] 0.40.231.898 I slot get_availabl: id  2 | task -1 | selected slot by LRU, t_last = -1
[47447] 0.40.231.900 I srv  get_availabl: updating prompt cache
[47447] 0.40.231.901 I srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
[47447] 0.40.231.903 I srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 16384.000 MiB, 256000 tokens, 17179869184 est)
[47447] 0.40.231.905 I srv  get_availabl: prompt cache update took 0.01 ms
[47447] 0.40.231.944 I slot launch_slot_: id  2 | task 1131 | processing task, is_child = 0
[47447] 0.40.231.947 I slot process_sing: id  0 | task -1 | saving idle slot to prompt cache
[47447] 0.40.231.947 I slot prompt_clear: id  0 | task -1 | clearing prompt with 0 tokens
[47447] 0.40.232.096 I slot process_sing: id  1 | task -1 | saving idle slot to prompt cache
[47447] 0.40.232.098 I slot prompt_clear: id  1 | task -1 | clearing prompt with 0 tokens
[47447] 0.40.232.250 I slot process_sing: id  3 | task -1 | saving idle slot to prompt cache
[47447] 0.40.232.564 W srv   prompt_save:  - saving prompt with length 1147, total state size = 187.732 MiB (draft: 0.000 MiB)
[47447] 0.40.465.224 I srv        update:  - cache state: 1 prompts, 337.358 MiB (limits: 16384.000 MiB, 256000 tokens, 256000 est)
[47447] 0.40.465.227 I srv        update:    - prompt 0x614a78ac7710:    1147 tokens, checkpoints:  1,   337.358 MiB
[47447] 0.40.465.227 I slot prompt_clear: id  3 | task -1 | clearing prompt with 1147 tokens
[47447] 0.41.257.685 I slot create_check: id  2 | task 1131 | created context checkpoint 1 of 32 (pos_min = 423, pos_max = 423, n_tokens = 424, size = 149.626 MiB)
0.56.797.744 I srv  proxy_reques: proxying request to model nondraft_tensor_Qwen3.6-27B-Q4_0.gguf on port 47447
[47447] 0.44.039.953 I srv    operator(): Chat format: peg-native
[47447] 0.44.387.907 I slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = -1
[47447] 0.44.387.909 I srv  get_availabl: updating prompt cache
[47447] 0.44.387.911 I srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
[47447] 0.44.387.914 I srv        update:  - cache state: 1 prompts, 337.358 MiB (limits: 16384.000 MiB, 256000 tokens, 256000 est)
[47447] 0.44.387.914 I srv        update:    - prompt 0x614a78ac7710:    1147 tokens, checkpoints:  1,   337.358 MiB
[47447] 0.44.387.915 I srv  get_availabl: prompt cache update took 0.01 ms
[47447] 0.44.387.955 I slot launch_slot_: id  1 | task 1139 | processing task, is_child = 0
[47447] 0.44.387.957 I slot process_sing: id  0 | task -1 | saving idle slot to prompt cache
[47447] 0.44.387.957 I slot prompt_clear: id  0 | task -1 | clearing prompt with 0 tokens
[47447] 0.44.388.080 I slot process_sing: id  3 | task -1 | saving idle slot to prompt cache
[47447] 0.44.388.081 I slot prompt_clear: id  3 | task -1 | clearing prompt with 0 tokens
[47447] 0.44.388.213 I srv  stream_sessi: stream_session_attach_pipe: conv_id= (empty=1)
[47447] 0.45.918.819 I slot create_check: id  1 | task 1139 | created context checkpoint 1 of 32 (pos_min = 132, pos_max = 132, n_tokens = 133, size = 149.626 MiB)
[47447] 0.48.003.905 I slot create_check: id  1 | task 1139 | created context checkpoint 2 of 32 (pos_min = 1144, pos_max = 1144, n_tokens = 1145, size = 149.626 MiB)
[47447] 0.48.888.392 I slot print_timing: id  1 | task 1139 | prompt processing, n_tokens =   1157, progress = 1.00, t =   4.50 s / 257.10 tokens per second
[47447] 0.51.100.466 I slot print_timing: id  2 | task 1131 | prompt eval time =    2231.06 ms /  1452 tokens (    1.54 ms per token,   650.81 tokens per second)
[47447] 0.51.100.469 I slot print_timing: id  2 | task 1131 |        eval time =    8403.82 ms /    12 tokens (  700.32 ms per token,     1.43 tokens per second)
[47447] 0.51.100.470 I slot print_timing: id  2 | task 1131 |       total time =   10634.87 ms /  1464 tokens
[47447] 0.51.100.470 I slot print_timing: id  2 | task 1131 |    graphs reused =       1129
[47447] 0.51.100.552 I slot      release: id  2 | task 1131 | stop processing: n_tokens = 1463, truncated = 0
[47447] 0.51.100.639 I srv  stream_sessi: stream_session_attach_pipe: conv_id= (empty=1)
[47447] 1.30.922.239 I slot print_timing: id  1 | task 1139 | n_decoded =    100, tg =   2.43 t/s, tg_3s =   2.43 t/s
[47447] 1.34.268.373 I slot print_timing: id  1 | task 1139 | n_decoded =    108, tg =   2.43 t/s, tg_3s =   2.39 t/s
[47447] 1.37.595.106 I slot print_timing: id  1 | task 1139 | n_decoded =    116, tg =   2.43 t/s, tg_3s =   2.40 t/s
[47447] 1.40.910.950 I slot print_timing: id  1 | task 1139 | n_decoded =    124, tg =   2.43 t/s, tg_3s =   2.41 t/s
[47447] 1.44.222.279 I slot print_timing: id  1 | task 1139 | n_decoded =    132, tg =   2.43 t/s, tg_3s =   2.42 t/s
1.59.519.602 E srv    operator(): http client error: Connection handling canceled
[47447] 1.47.152.943 W srv          stop: cancel task, id_task = 1139
[47447] 1.47.568.409 I slot print_timing: id  1 | task 1139 | n_decoded =    140, tg =   2.42 t/s, tg_3s =   2.39 t/s
[47447] 1.47.568.417 I slot      release: id  1 | task 1139 | stop processing: n_tokens = 1300, truncated = 0
[47447] 1.47.568.422 I srv  update_slots: all slots are idle
```

## cattivik66 @ 2026-06-27 (upd 2026-06-27)

| | |
|---|---|
| GPU | 2× W7800 48 GB, separate PCIe 4.0 x16 root complexes (no P2P, no XGMI) |
| OS | CachyOS, kernel 7.0.12, Mesa 26.1.2, RADV driver |
| Vulkan | 1.4.348, `KHR_coopmat` detected and active |
| Model | Qwen3.5-122B-A10B, UD-Q4_K_XL, 73 GiB |
| llama.cpp | `9d5d882d8` + this PR |

### Synthetic benchmark (pp4096 / tg128, ~4K context)

__Without this PR:__

| Mode | pp4096 | tg128 |
|---|---|---|
| row | 1229 | 48.4 |
| layer | 1195 | 48.4 |
| **tensor** | **1164** | **29.3** |

__With this PR applied:__

| Mode | pp4096 | tg128 |
|---|---|---|
| row | 1230 | 48.5 |
| layer | 1193 | 48.4 |
| **tensor** | **1450 (+25%)** | **43.6 (+49%)** |

At short context the PR delivers `-sm tensor` pp +25 %, tg +49 %. It becomes the fastest pp mode (1450 vs row's 1230) and tg is only 10 % behind row.

### Real-world big prompt benchmark (88K prompt, 164K context)
I've asked the LLM to provide me a resume of a book which content has been sent in the prompt

__Without this PR:__

| Mode | pp (t/s) | tg (t/s) | wall (s) |
|---|---|---|---|
| row | 669 | 37 | 172 |
| layer | 689 | 37 | 162 |
| **tensor** | **segfault** | **—** | **—** |

__With this PR applied:__

| Mode | pp (t/s) | tg (t/s) | wall (s) |
|---|---|---|---|
| row | 691 | 37 | 164 |
| layer | 693 | 37 | 162 |
| **tensor** | **677** | **9** | **255** |

### The tensor-mode segfault (fixed by this PR)

Without this PR, `-sm tensor` segfaults at any realistic context (`-c 167936` and `-c 204800` both crash). The log shows:

```
llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort
```

followed by `SIGSEGV` during model load. The single-buffer synthetic test at `-p 4096` worked because the KV cache was small enough. The PR's multi-buffer KV-cache fix is **required** for `-sm tensor` to function at usable c>

### The tensor decode regression at long context

With the PR applied, tensor-mode tg drops from 43.6 (short ctx) to 9 (long ctx) — 4.8× worse. Row/layer stay at 37. This appears to be the `-sm tensor` AllReduce decode path with high fixed per-call overhead at batch=1, mult>

For reference, `-sm layer` (1 sync point at the pipeline boundary) and `-sm row` (concatenation-only syncs) don't have this problem.

## marksverdhei @ 2026-06-27 (upd 2026-06-27)

@pwilkin 
As per the contribution guidelines in this repo, 
"This project does not accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity."

I manage a fork which welcomes such contribution, so consider pr-ing there :)
We would definitely like to make our vulkan backend as good as possible there
https://github.com/heiervang-technologies/ht-llama.cpp

Cheers

## maxious @ 2026-06-27 (upd 2026-06-27)

## 2x Intel Arc Pro B60 (Battlemage, BMG G21)

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (Intel open-source Mesa driver) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = Intel(R) Arc(tm) Pro B60 Graphics (BMG G21) (Intel open-source Mesa driver) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
```

`llama-bench -fa 1 -ngl -1 -p 512 -n 128 -r 3`, Vulkan only. Build at PR head `a448deb85`. Both B60s share the same Mesa ANV driver (`Mesa 26.1.3`, driverUUID match), so the native cross-device timeline-semaphore import path is used (not the CPU-proxy fallback).

| model | size | params | backend | ngl | sm | fa | test | t/s |
| --- | ---: | ---: | --- | --: | --- | --: | ---: | ---: |
| llama 1B Q8_0 | 1.22 GiB | 1.24 B | Vulkan | -1 | layer | 1 | pp512 | 6100.11 ± 18.65 |
| llama 1B Q8_0 | 1.22 GiB | 1.24 B | Vulkan | -1 | layer | 1 | tg128 | 187.42 ± 8.38 |
| llama 1B Q8_0 | 1.22 GiB | 1.24 B | Vulkan | -1 | tensor | 1 | pp512 | 8626.08 ± 108.51 |
| llama 1B Q8_0 | 1.22 GiB | 1.24 B | Vulkan | -1 | tensor | 1 | tg128 | 115.82 ± 7.53 |
| llama 8B Q4_0 | 4.33 GiB | 8.03 B | Vulkan | -1 | layer | 1 | pp512 | 1375.81 ± 124.48 |
| llama 8B Q4_0 | 4.33 GiB | 8.03 B | Vulkan | -1 | layer | 1 | tg128 | 50.19 ± 11.48 |
| llama 8B Q4_0 | 4.33 GiB | 8.03 B | Vulkan | -1 | tensor | 1 | pp512 | 2158.55 ± 2.04 |
| llama 8B Q4_0 | 4.33 GiB | 8.03 B | Vulkan | -1 | tensor | 1 | tg128 | 41.42 ± 10.05 |
| qwen35 9B Q8_0 | 9.10 GiB | 9.20 B | Vulkan | -1 | layer | 1 | pp512 | 1090.84 ± 1.86 |
| qwen35 9B Q8_0 | 9.10 GiB | 9.20 B | Vulkan | -1 | layer | 1 | tg128 | 34.85 ± 2.10 |
| qwen35 9B Q8_0 | 9.10 GiB | 9.20 B | Vulkan | -1 | tensor | 1 | pp512 | 1609.03 ± 30.27 |
| qwen35 9B Q8_0 | 9.10 GiB | 9.20 B | Vulkan | -1 | tensor | 1 | tg128 | 38.28 ± 1.08 |
| gpt-oss 20B Q8_0 | 11.27 GiB | 20.91 B | Vulkan | -1 | layer | 1 | pp512 | 1305.84 ± 37.30 |
| gpt-oss 20B Q8_0 | 11.27 GiB | 20.91 B | Vulkan | -1 | layer | 1 | tg128 | 43.37 ± 11.86 |
| gpt-oss 20B Q8_0 | 11.27 GiB | 20.91 B | Vulkan | -1 | tensor | 1 | pp512 | 2013.57 ± 48.94 |
| gpt-oss 20B Q8_0 | 11.27 GiB | 20.91 B | Vulkan | -1 | tensor | 1 | tg128 | 53.73 ± 5.34 |
| qwen35moe 35B.A3B Q3_K - Medium | 12.79 GiB | 35.51 B | Vulkan | -1 | layer | 1 | pp512 | 771.10 ± 12.43 |
| qwen35moe 35B.A3B Q3_K - Medium | 12.79 GiB | 35.51 B | Vulkan | -1 | layer | 1 | tg128 | 44.54 ± 7.42 |
| qwen35moe 35B.A3B Q3_K - Medium | 12.79 GiB | 35.51 B | Vulkan | -1 | tensor | 1 | pp512 | 1111.37 ± 10.68 |
| qwen35moe 35B.A3B Q3_K - Medium | 12.79 GiB | 35.51 B | Vulkan | -1 | tensor | 1 | tg128 | 22.77 ± 7.64 |
| qwen35 27B Q8_0 | 33.31 GiB | 27.32 B | Vulkan | -1 | layer | 1 | pp512 | 321.05 ± 11.59 |
| qwen35 27B Q8_0 | 33.31 GiB | 27.32 B | Vulkan | -1 | layer | 1 | tg128 | 9.32 ± 0.28 |
| qwen35 27B Q8_0 | 33.31 GiB | 27.32 B | Vulkan | -1 | tensor | 1 | pp512 | 517.89 ± 0.90 |
| qwen35 27B Q8_0 | 33.31 GiB | 27.32 B | Vulkan | -1 | tensor | 1 | tg128 | 13.89 ± 0.03 |

### Observations

- **Prefill (pp512):** `-sm tensor` is ~1.4-1.6x faster than `-sm layer` across every model, including the dense 27B Q8_0 that exceeds a single B60's 24 GiB. The pipelined chunked AllReduce does its job on prefill activations, consistent with the NVIDIA results in the PR description.
- **Decode (tg128):** Matches the PR author's characterisation - for models that comfortably fit one B60 (1B, 8B, 9B, dense MoE at Q3), tensor mode is break-even or slightly slower (per-call overhead on small decode tensors dominates the single-shot path). For gpt-oss-20B and the dense 27B Q8_0, the only two that genuinely need TP to fit on 2x24 GiB, tensor mode wins on TG too: gpt-oss 43 -> 54 t/s, and 27B Q8_0 9.3 -> 13.9 t/s - exactly the "balanced GPU pair, TG boost" regime the PR description calls out.
- The 27B Q8_0 result is in the same ballpark as the mixed AMD pair reported above (10.68 tg with tensor on 27B Q5_K), so Intel/Vulkan lands competitively in that workload class.

## pwilkin @ 2026-06-27 (upd 2026-06-27)

@marksverdhei it's a prototype, don't worry :) when it's been decently tested and cleaned up I'll deslopify it :)

## pwilkin @ 2026-06-27 (upd 2026-06-27)

So the damn clanker decided to only implement the proper allreduce for 2 GPUs, because why bother ;) sorry for all the people with 4+ who posted their results, could you please retest with the new commit? (I've tested up to 8 GPUs now for correctness)

## AbdullahMPrograms @ 2026-06-27 (upd 2026-06-27)

Sharing some benchmark results with 2/4/5x Radeon PRO W7900's:
##Vulkan Stock:
./LLM/llama.cpp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm layer
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 |   1 |           pp512 |        786.58 ± 2.38 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 |   1 |           tg128 |         33.94 ± 0.03 |

##Vulkan PR TP (2 GPU):
./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           pp512 |       836.11 ± 92.66 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           tg128 |         29.77 ± 0.41 |

##Vulkan PR TP (4 GPU):
./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1/Vulkan2/Vulkan3
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3 |           pp512 |       413.89 ± 56.70 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3 |           tg128 |         27.29 ± 0.62 |

##Vulkan PR TP (5 GPU):
./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1/Vulkan2/Vulkan3/Vulkan4
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3/Vulkan4 |           pp512 |       341.89 ± 55.39 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1/Vulkan2/Vulkan3/Vulkan4 |           tg128 |         23.23 ± 0.62 |

#For comparison with ROCm:
##ROCm Stock:
./LLM/llama.cpp/rocm/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm layer
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 |   1 |           pp512 |        843.56 ± 3.07 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 |   1 |           tg128 |         28.52 ± 0.04 |

build: 050ee92d0 (9821)
##ROCm TP RCCL (2 GPU):
./LLM/llama.cpp/rocm-rccl/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev ROCm0/ROCm1
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1  |           pp512 |      1429.22 ± 10.58 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1  |           tg128 |         37.50 ± 0.47 |

##ROCm TP RCCL (4 GPU):
./LLM/llama.cpp/rocm-rccl/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev ROCm0/ROCm1/ROCm2/ROCm3
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           pp512 |       1992.92 ± 6.26 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3 |           tg128 |         45.54 ± 1.95 |

##ROCm TP RCCL (5 GPU):
./LLM/llama.cpp/rocm-rccl/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev ROCm0/ROCm1/ROCm2/ROCm3/ROCm4
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3/ROCm4 |           pp512 |      2033.37 ± 13.56 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | ROCm       |  -1 | tensor |   1 | ROCm0/ROCm1/ROCm2/ROCm3/ROCm4 |           tg128 |         48.57 ± 2.89 |

## digitalscream @ 2026-06-27 (upd 2026-06-27)

Could someone else check multi-turn conversations? As noted in my comment above, I'm finding that the first prompt works as expected, but any follow up tanks down to 2t/s.

## pwilkin @ 2026-06-27 (upd 2026-06-27)

@AbdullahMPrograms is this after the multiGPU fix already?

@netrunnereve I risked a proper fix to the real buffer mapping issue.

## AbdullahMPrograms @ 2026-06-27 (upd 2026-06-27)

> @AbdullahMPrograms is this after the multiGPU fix already?
> 
> @netrunnereve I risked a proper fix to the real buffer mapping issue.

@pwilkin This is with commit: 362fdd2b7

```
git log --oneline -n 5
362fdd2b7 (HEAD -> pr-25051) vulkan: generalize -sm tensor AllReduce to >2 devices
a448deb85 fix Windows build
68c289bd7 vulkan: portable CPU-proxy fallback for cross-driver -sm tensor AllReduce
5dc559cdd vulkan: GPU-pipelined multi-GPU AllReduce for -sm tensor
325cceab0 ggml-backend-meta: fix SPLIT_MODE_TENSOR segfault on multi_buffer KV-cache views
```

## AbdullahMPrograms @ 2026-06-27 (upd 2026-06-27)

> Could someone else check multi-turn conversations? As noted in my comment above, I'm finding that the first prompt works as expected, but any follow up tanks down to 2t/s.

I can reproduce this behavior as well, first prompt:
```
0.58.303.871 I slot print_timing: id  3 | task 0 | prompt eval time =     606.59 ms /    17 tokens (   35.68 ms per token,    28.03 tokens per second)
0.58.303.874 I slot print_timing: id  3 | task 0 |        eval time =   34812.30 ms /  1061 tokens (   32.81 ms per token,    30.48 tokens per second)
0.58.303.896 I slot print_timing: id  3 | task 0 |       total time =   35418.89 ms /  1078 tokens
```

follow up prompt:
```
2.09.854.324 I slot print_timing: id  2 | task 1063 | prompt eval time =    4341.21 ms /  1090 tokens (    3.98 ms per token,   251.08 tokens per second)
2.09.854.328 I slot print_timing: id  2 | task 1063 |        eval time =   63484.68 ms /   214 tokens (  296.66 ms per token,     3.37 tokens per second)
2.09.854.329 I slot print_timing: id  2 | task 1063 |       total time =   67825.90 ms /  1304 tokens
```

## digitalscream @ 2026-06-27 (upd 2026-06-27)

> > Could someone else check multi-turn conversations? As noted in my comment above, I'm finding that the first prompt works as expected, but any follow up tanks down to 2t/s.
> 
> I can reproduce this behavior as well, first prompt:
> 
> ```
> 0.58.303.871 I slot print_timing: id  3 | task 0 | prompt eval time =     606.59 ms /    17 tokens (   35.68 ms per token,    28.03 tokens per second)
> 0.58.303.874 I slot print_timing: id  3 | task 0 |        eval time =   34812.30 ms /  1061 tokens (   32.81 ms per token,    30.48 tokens per second)
> 0.58.303.896 I slot print_timing: id  3 | task 0 |       total time =   35418.89 ms /  1078 tokens
> ```
> 
> follow up prompt:
> 
> ```
> 2.09.854.324 I slot print_timing: id  2 | task 1063 | prompt eval time =    4341.21 ms /  1090 tokens (    3.98 ms per token,   251.08 tokens per second)
> 2.09.854.328 I slot print_timing: id  2 | task 1063 |        eval time =   63484.68 ms /   214 tokens (  296.66 ms per token,     3.37 tokens per second)
> 2.09.854.329 I slot print_timing: id  2 | task 1063 |       total time =   67825.90 ms /  1304 tokens
> ```

Thank you! At least I know I'm not holding it wrong ;)

(EDIT: ...or we both are)

## AbdullahMPrograms @ 2026-06-27 (upd 2026-06-27)

I've also encountered some gibberish in llama-server with TP on 5 GPU's:
<img width="802" height="1046" alt="image" src="https://github.com/user-attachments/assets/f5cca2f2-08d1-4dc7-8db0-10173c4ab885" />
2-4 GPU's does not have gibberish, launch command: 
```
./LLM/llama.cpp-vulkantp/vulkan/bin/llama-server -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -c 32768 -ngl 999 -fa on -fit off -sm tensor --host 0.0.0.0 --port 7001 -dev Vulkan0,Vulkan1,Vulkan2,Vulkan3,Vulkan4
```
above on commit e578ca2c8

## wizardeur @ 2026-06-27 (upd 2026-06-27)

It's faster now on TP=4 and TP=8 (RX7900XTX). Shorter table this time:

| Backend | GPUs | sm     | pp512 Run 1 | pp512 Run 2 | tg128 Run 1 | tg128 Run 2 |
| ------- | ---: | ------ | ----------: | ----------: | ----------: | ----------: |
| Vulkan  |    2 | layer  |      809.00 |      794.62 |       34.50 |       33.40 |
| Vulkan  |    2 | tensor |     1283.40 |     1286.12 |       41.57 |       41.68 |
| Vulkan  |    4 | layer  |      762.29 |      757.42 |       29.66 |       29.58 |
| Vulkan  |    4 | tensor |      305.86 |      981.02 |       14.68 |       37.50 |
| Vulkan  |    8 | layer  |      697.64 |      701.80 |       11.89 |       12.34 |
| Vulkan  |    8 | tensor |       95.61 |      492.38 |        3.87 |       21.10 |
| ROCm    |    2 | layer  |      913.54 |      915.04 |       26.34 |       26.13 |
| ROCm    |    2 | tensor |     1535.47 |     1532.96 |       45.35 |       45.42 |
| ROCm    |    4 | layer  |      877.60 |      879.33 |       23.58 |       23.63 |
| ROCm    |    4 | tensor |     2085.89 |     2070.25 |       55.79 |       55.40 |
| ROCm    |    8 | layer  |      811.63 |      807.94 |       21.48 |       21.55 |
| ROCm    |    8 | tensor |     2163.08 |     2165.96 |       37.54 |       36.99 |

And for a multiturn conversation, I don't see any significant degradation.
1st message:
```
2.14.960.380 I slot print_timing: id  3 | task 0 | prompt eval time =    4516.41 ms /  4618 tokens (    0.98 ms per token,  1022.49 tokens per s
econd)
2.14.960.383 I slot print_timing: id  3 | task 0 |        eval time =   76061.21 ms /  1390 tokens (   54.72 ms per token,    18.27 tokens per s
econd)
```

2nd message:
```
8.48.047.849 I slot print_timing: id  3 | task 1395 | prompt eval time =    2080.61 ms /  1926 tokens (    1.08 ms per token,   925.69 tokens per second)
8.48.047.852 I slot print_timing: id  3 | task 1395 |        eval time =  125606.23 ms /  2336 tokens (   53.77 ms per token,    18.60 tokens per second)
```

3rd message:
```
23.52.991.932 I slot print_timing: id  3 | task 3735 | prompt eval time =    3239.29 ms /  3366 tokens (    0.96 ms per token,  1039.12 tokens per second)
23.52.991.936 I slot print_timing: id  3 | task 3735 |        eval time =   57636.44 ms /  1098 tokens (   52.49 ms per token,    19.05 tokens per second)
```

## pwilkin @ 2026-06-27 (upd 2026-06-27)

Looking into the multiGPU degradation, as for the slowdown, can one of you possibly capture a perf profile?

## digitalscream @ 2026-06-28 (upd 2026-06-28)

> Looking into the multiGPU degradation, as for the slowdown, can one of you possibly capture a perf profile?

Sure - not done that before, though. Totally revealing my lack of knowledge here, but can you point me to documentation for doing that?

## pwilkin @ 2026-06-28 (upd 2026-06-28)

> > Looking into the multiGPU degradation, as for the slowdown, can one of you possibly capture a perf profile?
> 
> Sure - not done that before, though. Totally revealing my lack of knowledge here, but can you point me to documentation for doing that?

Sure: https://perfwiki.github.io/main/tutorial/

It's actually pretty easy: you run `perf record <command>`, then you upload the perf results eg. to Huggingface.

## pwilkin @ 2026-06-28 (upd 2026-06-28)

All right - a prototype D2D implementation for AMD cards has landed, could anyone with a multi-AMD GPU setup here give it a try?

EDIT: for the D2D you have to run with `GGML_VK_COMM_D2D=1`

## digitalscream @ 2026-06-28 (upd 2026-06-28)

> All right - a prototype D2D implementation for AMD cards has landed, could anyone with a multi-AMD GPU setup here give it a try?

Previous result:
```
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp2048 |       1580.58 ± 5.18 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp4096 |       1556.35 ± 2.32 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |           tg128 |         29.91 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp2048 |       1125.71 ± 2.82 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp4096 |       1110.65 ± 1.31 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |           tg128 |         36.84 ± 0.38 |
```

After:

```
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp2048 |      1542.28 ± 38.78 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp4096 |       1536.07 ± 1.12 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |           tg128 |         29.78 ± 0.26 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp2048 |       1119.14 ± 2.83 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp4096 |       1106.15 ± 2.13 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |           tg128 |         36.47 ± 0.09 |

```

Results largely within noise. This is on x8/x8 PCIE 4.0 connections, mind. Would you expect to see a performance difference?

## pwilkin @ 2026-06-28 (upd 2026-06-28)

@digitalscream sorry, clever me forgot to mention you have to run with `GGML_VK_COMM_D2D=1` :/

## AbdullahMPrograms @ 2026-06-28 (upd 2026-06-28)

> All right - a prototype D2D implementation for AMD cards has landed, could anyone with a multi-AMD GPU setup here give it a try?

Results seem to be significantly worse with GGML_VK_COMM_D2D=1 enabled:

GGML_VK_COMM_D2D Disabled:
```./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1```
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           pp512 |       836.11 ± 92.66 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           tg128 |         29.77 ± 0.41 |

GGML_VK_COMM_D2D Enabled:
```GGML_VK_COMM_D2D=1 ./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1```
| model                          |       size |     params | backend    | ngl |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           pp512 |        488.69 ± 3.43 |
| qwen35 27B Q4_K - Medium       |  17.62 GiB |    27.32 B | Vulkan     |  -1 | tensor |   1 | Vulkan0/Vulkan1 |           tg128 |         13.67 ± 0.08 |

## digitalscream @ 2026-06-28 (upd 2026-06-28)

Same here, except worse - with D2D, power draw drops from 250-260W previously to 130W during prefill and 75W during decode, and performance falls by 90% across the board.

```
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon Graphics (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp2048 |      1532.42 ± 36.69 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |          pp4096 |       1527.48 ± 7.08 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 |  layer |   1 |           tg128 |         29.75 ± 0.14 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp2048 |        156.48 ± 0.67 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |          pp4096 |        156.19 ± 0.80 |
| qwen35 27B Q4_K - Medium       |  15.92 GiB |    27.32 B | Vulkan     |  99 |     1024 | tensor |   1 |           tg128 |          2.32 ± 0.00 |

```

## pwilkin @ 2026-06-28 (upd 2026-06-28)

Interesting... would really love those perf dumps, I'm getting to the limit of what I can do without data from the actual hardware :)

## digitalscream @ 2026-06-28 (upd 2026-06-28)

> Interesting... would really love those perf dumps, I'm getting to the limit of what I can do without data from the actual hardware :)

Would it help if you had direct access to my AI box? I can't be of much use regarding code, but if contributing time on there will help, I'm happy to do it.

## AbdullahMPrograms @ 2026-06-28 (upd 2026-06-28)

> Interesting... would really love those perf dumps, I'm getting to the limit of what I can do without data from the actual hardware :)

Heres the perf dump with command ```GGML_VK_COMM_D2D=1 ./LLM/llama.cpp-vulkantp/vulkan/bin/llama-bench -m /home/ultimis/LLM/Models/bartowski/Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_L.gguf -ngl -1 -fa 1 -sm tensor -dev Vulkan0/Vulkan1```: 
https://gist.github.com/AbdullahMPrograms/e9155e3094e4c603b2abfc771c336709

## pwilkin @ 2026-06-28 (upd 2026-06-28)

> > Interesting... would really love those perf dumps, I'm getting to the limit of what I can do without data from the actual hardware :)
> 
> Would it help if you had direct access to my AI box? I can't be of much use regarding code, but if contributing time on there will help, I'm happy to do it.

If you'd be willing to provide such access then of course I'd be grateful, my contact details are in my profile.

## digitalscream @ 2026-06-28 (upd 2026-06-28)

> > > Interesting... would really love those perf dumps, I'm getting to the limit of what I can do without data from the actual hardware :)
> > 
> > 
> > Would it help if you had direct access to my AI box? I can't be of much use regarding code, but if contributing time on there will help, I'm happy to do it.
> 
> If you'd be willing to provide such access then of course I'd be grateful, my contact details are in my profile.

Sure - check your email :)

## netrunnereve @ 2026-06-28 (upd 2026-06-28)

> @AbdullahMPrograms is this after the multiGPU fix already?
> 
> @netrunnereve I risked a proper fix to the real buffer mapping issue.

I didn't test it but I think your method of using the address should work, that's much better than storing a list of tensors somewhere. By the way it might be worth just submitting this crash fix separately to get tensor parallel working and then work on the performance stuff later.

## TheBlueMatt @ 2026-06-28 (upd 2026-06-28)

Hum, `build/bin/llama-cli -hf unsloth/Qwen3.5-122B-A10B-GGUF:Q4_K_XL -sm tensor -fa 1 --no-mmap -st --prompt "What is the capital of france?" -c 8192 -n 500` outputs chinese characters for me on this branch on 4xB60s (`-sm layer -fa 1` works fine on this branch).

FWIW, in rebasing my existing tensor-par branch I noted that a patch my clanker wrote to do vulkan graph caching was a 25% improvement on top of tensor-par, so there's plenty of room to grow with nontrivial patches later, at least compared to the CUDA stuff that does graph caching.

## pwilkin @ 2026-06-28 (upd 2026-06-28)

@TheBlueMatt I did run correctness checks, but not on a 122B model, so it's possible that the F16 pipeline is causing overflows. Can you reproduce this on smaller models as well?

## TheBlueMatt @ 2026-06-28 (upd 2026-06-28)

The same command generates sensible output for `unsloth/Qwen3.6-27B-GGUF:BF16` and `unsloth/gemma-4-31B-it-GGUF:BF16`, but also generates gibberish on `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_XL`. (apologies, had posted a previous comment and somehow managed to ^R and swap `tensor` for `layer` parallelism in testing).

## 0cc4m @ 2026-06-29 (upd 2026-06-29)

> FWIW, in rebasing my existing tensor-par branch I noted that a patch my clanker wrote to do vulkan graph caching was a 25% improvement on top of tensor-par, so there's plenty of room to grow with nontrivial patches later, at least compared to the CUDA stuff that does graph caching.

Is that similar to #24720?

## pwilkin @ 2026-06-29 (upd 2026-06-29)

Thanks to @digitalscream I managed to nail the slowdown issue for decode, the patch is in :)

## En3Tho @ 2026-06-29 (upd 2026-06-29)

@jeffbolznv Hey! Sorry to ping here but am I right to understand that Vulkan does not allow p2p transfers without an active device group even if pice allows it? But device group usually cannot be created without a dedicated link like sli/crossfire? If so then maybe p2p in vulkan can be revised somehow?

@pwilkin A quick question: what happents to mtp/draft head with tensor splitting? Is it a subject to splitting too or it can reside within a single gpu memory?

## 0cc4m @ 2026-06-29 (upd 2026-06-29)

> Thanks to @digitalscream I managed to nail the slowdown issue for decode, the patch is in :)

Assuming that other drivers don't have this issue, that's not a fix, but a workaround. It should be reported to RADV.

## pwilkin @ 2026-06-29 (upd 2026-06-29)

@JohannesGaessler 

> A proper implementation that would also fix Vulkan would require some refactoring of the ggml graph allocator.

Could you elaborate a bit on this?

## JohannesGaessler @ 2026-06-30 (upd 2026-06-30)

In the graph allocator there is currently a strict 1:1 mapping between abstract ggml backend buffer types and physical ggml backend buffers. This works for the conventional buffers per ggml backend. But for the meta backend there should instead be a mapping from one ggml backend buffer type to multiple physical buffers.

## pwilkin @ 2026-06-30 (upd 2026-06-30)

Aight, that one I'll have to take a clanker-less look at :)

## TheBlueMatt @ 2026-06-30 (upd 2026-06-30)

Presumably such a conversion would also address the current over-allocation of compute buffers in TP?

## TheBlueMatt @ 2026-06-30 (upd 2026-06-30)

> Is that similar to https://github.com/ggml-org/llama.cpp/pull/24720?

Let me test this on top of @pwilkin's eventual outcome. IIRC the graph caching didn't make a huge impact in layer parallelism but much more in TP. It might also have been because of the clanker's lazy vulkan TP implementation, I haven't found the time to look that deeply at it.

## JohannesGaessler @ 2026-06-30 (upd 2026-06-30)

>Aight, that one I'll have to take a clanker-less look at :)

If you do, please make a separate PR.

>Presumably such a conversion would also address the current over-allocation of compute buffers in TP?

Yes.

## pwilkin @ 2026-06-30 (upd 2026-06-30)

@JohannesGaessler yeah that is obviously out-of-scope here, for this one I just wanted to see if I can get a quick win with improving Vulkan TP performance.

## AbdullahMPrograms @ 2026-07-27 (upd 2026-07-27)

any updates on enabling TP on Vulkan? This work was really promising

## leapdragon @ 2026-08-16 (upd 2026-08-16)

As a user, I'd love to see us make progress here. Vulkan has some unique benefits on AMD hardware, being about 35% faster for generation on old builds using row splits across multiple cards, at the cost of prefill. But there are times when you want generation to go a lot faster and you don't need to prefill as much.

Now it's down to running quite old builds at this point or just sticking with ROCm tensor split mode all the time.

## pwilkin @ 2026-08-16 (upd 2026-08-16)

Once I'm back from vacation I'll revisit this.
