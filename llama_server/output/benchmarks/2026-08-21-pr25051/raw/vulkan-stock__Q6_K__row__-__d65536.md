WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 2 = AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
llama_bench: error: failed to load model '/home/addison-gambhir/Desktop/local_llm/llama_server/models/Qwen3.8-27B-Q6_K.gguf'
| model                          |       size |     params | backend    | ngl | n_ubatch |   main_gpu |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | ---------: | -----: | --: | ------------ | --------------: | -------------------: |
