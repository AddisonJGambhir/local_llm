# Dual-GPU llama.cpp optimization report

**Assessment date:** 2026-08-16  
**Scope:** Qwen 3.8 27B on the local dual-AMD-GPU llama.cpp server. This is an analysis and test plan only; it makes no runtime or configuration changes.

## Executive conclusion

The current deployment is already using the right fundamental topology for one large interactive model across these two cards: layer split, one server, Flash Attention, a 256K context target, and the R9700 as the main/vision GPU. llama.cpp documents layer split as the default and most compatible multi-GPU mode; it minimizes cross-GPU transfer and is preferred when prefill/batch throughput matters. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) The local prefill profile implements that topology with ROCm0 = RX 7900 XTX and ROCm1 = Radeon AI PRO R9700. [current prefill profile](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:2)

The useful optimization is not a blanket switch to tensor parallelism. It is keeping two deliberately different profiles: the existing Q8-KV profile for maximum 256K prompt-ingest headroom, and the existing F16-KV 28,37 layer split for better interactive code-generation quality/speed when its memory headroom is acceptable. The local measurements report Q8 KV at 1,1 as 1,067 prompt tokens/s and 31.3 generation tokens/s, while the F16 28,37 profile reports 982 prompt tokens/s and a 38.5-token/s five-seed median at 256K. [measured comparison](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68) [F16 measurement set](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:91)

## Assessment snapshot

| Item | Observed state | Evidence |
|---|---|---|
| GPU order used by the server | ROCm0 is the 24 GiB RX 7900 XTX; ROCm1 is the 32 GiB Radeon AI PRO R9700. | [device mapping audit](/home/addison-gambhir/Desktop/local_llm/llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md:178) |
| PCIe links | The R9700 was operating at PCIe 5.0 x16 (32.0 GT/s); the 7900 XTX at PCIe 4.0 x16 (16.0 GT/s). | [R9700 link speed](/sys/class/drm/card0/device/current_link_speed) [R9700 link width](/sys/class/drm/card0/device/current_link_width) [XTX link speed](/sys/class/drm/card1/device/current_link_speed) [XTX link width](/sys/class/drm/card1/device/current_link_width) |
| Active model mode | The current server log shows a 262,144-token slot, Q8 KV cache, layer split over ROCm0/ROCm1, and a three-token MTP draft limit. | [startup log](/home/addison-gambhir/.local/state/local-llm/llama-server.log:157079) [active profile arguments](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:93) |
| Recent long-context performance | A logged 135,390-token prompt evaluated at 491.28 prompt tokens/s; the subsequent 1,966-token generation ran at 22.11 tokens/s. This is a real deep-context datapoint, not a short-prompt estimate. | [live timing log](/home/addison-gambhir/.local/state/local-llm/llama-server.log:157146) |
| Present caveat | The active multimodal server disables cache reuse, so its configured cache-reuse value is not currently delivering a reuse benefit. | [live server log](/home/addison-gambhir/.local/state/local-llm/llama-server.log:157081) |

The physical asymmetry is meaningful: the XTX has roughly 24 GiB of VRAM and the R9700 roughly 32 GiB, so equal model distribution is not automatically the best decode configuration. The tested F16 profile shifts more model responsibility toward the R9700 with tensor-split 28,37 and reserves about 2.9 GiB on each card in its measurement. [hardware/profile rationale](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:17) [tested 28,37 memory result](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:85)

AMD’s current Radeon compatibility documentation lists both Radeon 9000-series (RDNA 4) and selected Radeon 7000-series (RDNA 3) hardware under current ROCm support, and specifically lists llama.cpp among supported Linux inference frameworks. That makes the heterogeneous pair a supportable ROCm target, although it does not make every multi-GPU backend path equally mature. [AMD ROCm Radeon compatibility](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html)

## Why the present split mode should remain layer split

llama.cpp defines layer split as the default pipeline approach: contiguous layers are assigned to GPUs and each GPU owns the KV cache for the layers it runs. Its guide says that this minimizes cross-GPU transfer, maximizes batch/prefill throughput, and is the broadly compatible choice when interconnect bandwidth is limited relative to GPU memory bandwidth. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)

That description matches the local objective. The current profile explicitly uses --split-mode layer, --device ROCm0,ROCm1, --tensor-split 1,1, Flash Attention, and a large batch size for long-prompt ingestion. [active prefill profile](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:123) The measured Q8 configuration also obtains its extra prompt-ingest speed by keeping the cache quantized, not by depending on an unvalidated ROCm tensor-parallel path. [Q8 profile result](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68)

Tensor split is therefore not a production recommendation for this model. Upstream labels it experimental, states that it needs Flash Attention and F32/F16/BF16 KV cache rather than quantized KV cache, and warns that it is not implemented for hybrid state-space architectures. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) The local Qwen profile identifies this model as hybrid/recurrent and deliberately enables context checkpoints for that architecture. [hybrid-model settings](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:49) It would also trade the current Q8-KV capacity profile for F16-or-higher KV cache requirements. This is an inference from the upstream limitation and the local profile, not a claim that every future llama.cpp build will fail. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) [Q8 capacity rationale](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:17)

RCCL is the ROCm analogue of the collective-communication path described by llama.cpp, but upstream says its HIP/RCCL build option is disabled by default because it is not universally beneficial. It should only be treated as a separately compiled, benchmarked experiment after the higher-confidence profile work below. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)

## The two profiles have different jobs

| Workload | Recommended profile | Why | Evidence |
|---|---|---|---|
| Large documents, RAG, long prompt ingestion, or maximum 256K reliability | preset-38-27b-q8-rocm-dual-256k-ppopt.sh | Q8 KV makes the 1,1 split viable at 256K and the recorded prefill result is higher. | [profile purpose and result](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:2) [memory explanation](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:17) |
| Interactive coding and output-heavy work where generation quality/speed matters more than peak prefill | preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh, whose target KV settings are F16 | The tested 28,37 F16 split has a higher multi-seed decode median and higher reported speculation acceptance than the Q8 result. | [F16 target cache settings](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:116) [comparison](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68) |
| Vision-enabled work | Keep mmproj on ROCm1 and leave margin on both cards | Local testing found text performance unchanged while moving permanent and transient vision allocation off the tighter XTX; image prefill remains a tradeoff. | [vision placement measurement](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:31) |

The filename of the second profile is historical; its explicit --cache-type-k f16 and --cache-type-v f16 arguments, rather than its filename, determine its target-cache type. [F16 arguments](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:116)

The F16 profile is not simply a more aggressive split: it keeps --parallel 1 because the hybrid recurrent buffer grows substantially with parallel slots, places the main GPU on ROCm1 for headroom, and uses the same 6,144-token batch size that local testing identified as the observed prompt-processing peak. [parallel/headroom rationale](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:18)

## What not to change casually

- Do not move to --split-mode tensor for the active Q8-KV hybrid model. Its cache and architecture restrictions conflict with the current large-context design, and no local tensor-mode validation result exists in this assessment. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) [active Q8/hybrid settings](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:49)

- Do not use the locally explored 26,39 F16 split as a daily profile. It put the R9700 at about 96% utilization and was recorded as unsafe, whereas 28,37 maintained the measured margin. [split safety notes](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:85)

- Do not increase --parallel above one for this 256K hybrid configuration without measuring recurrent-state allocation and user-visible latency. The existing profile records a much larger recurrent buffer with additional parallel slots. [parallel allocation note](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:38)

- Do not count --cache-reuse 256 as a current optimization while vision/mmproj is enabled: the live server reports it disabled for multimodal execution. [live server log](/home/addison-gambhir/.local/state/local-llm/llama-server.log:157081)

## Prioritized optimization plan

### 1. Complete the deep-context speculation sweep first

Run n-max values 0, 1, 2, and 3 at the same prompt length and output length, twice: once near the normal operating prompt size and once near 256K. Record prompt tokens/s, generation tokens/s, speculative acceptance, and total time. The existing audit explicitly identifies a deep-context n-max sweep as the next high-value measurement, and the active profile currently uses n-max 3. [local benchmark plan](/home/addison-gambhir/Desktop/local_llm/llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md:203) [active draft setting](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:99)

Decision rule: retain the depth with the best end-to-end generation time at the relevant context length, not the depth with the best short-context acceptance. This is an inference from the audit’s request to test at both shallow and deep contexts and the observed 22.11-token/s generation rate after a 135K prompt. [benchmark plan](/home/addison-gambhir/Desktop/local_llm/llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md:203) [deep-context timing](/home/addison-gambhir/.local/state/local-llm/llama-server.log:157146)

### 2. Make profile selection operational

Keep the present Q8 prefill profile as the default for document-heavy work. Launch the F16 28,37 profile for coding/output-heavy sessions after confirming both cards retain its recorded safety margin and that the session does not need the Q8 profile’s extra prefill/capacity headroom. This recommendation follows the local measured tradeoff rather than a theoretical assumption. [Q8-versus-F16 result](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68) [F16 memory margin](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:85)

Before judging either profile, restart after image tests or collect clean text-only and image-enabled samples separately. The prefill profile records a roughly 40% prompt-processing regression after images and documents restart as the tested workaround. [post-image regression note](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:142)

### 3. Validate the implementation version before tuning around it

The local audit identifies a regression around the b10450 llama.cpp build compared with b10358 in its own measurements, including weaker prompt processing, generation, and acceptance. Treat that as a version-specific local observation: benchmark the current build against the known-good revision with the same model, profile, prompt, and seeds before choosing a version for daily use. [local build comparison](/home/addison-gambhir/Desktop/local_llm/llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md:80)

Do this one variable at a time. ROCm support for these GPU generations is current, but an AMD compatibility listing is not evidence that a new runtime is faster or more stable with this exact heterogeneous split. [AMD ROCm Radeon compatibility](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html)

### 4. Only then run one constrained Q8 split experiment

If the Q8 default still causes XTX desktop pressure, test one slightly R9700-biased layer split at a time under a hard guard: abort on display instability, GTT/spill growth, or a material loss in generation speed. llama.cpp defines --tensor-split as proportions in --device order, so any experiment must document which share belongs to ROCm0 and ROCm1. [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) The locally validated F16 profile already demonstrates why the bigger card can take a larger share; it is the starting evidence, not proof that the same ratio is ideal for Q8 KV. [tested F16 split](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-q8-256k-mmproj.sh:147)

This is lower priority than the speculation sweep because the present Q8 1,1 profile already has a recorded performance result and the active application is currently functioning. [Q8 baseline](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68)

### 5. Consider display offload as a hardware/desktop decision, not a llama.cpp flag

The local profile estimates roughly 1.7–3 GiB of XTX desktop use, which directly reduces the margin available to the smaller VRAM card. Moving displays to the integrated GPU or the R9700 could free XTX margin, but it changes the workstation’s display topology and should be evaluated as a reversible desktop decision before any server-tuning conclusion is drawn. [desktop-memory estimate](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:17)

## Measurement contract

For each comparison, keep model quantization, mmproj state, context target, batch size, split, build, and prompt fixed; run at least five different seeds for interactive decode measurements. The F16 profile’s reported median is based on a five-seed set and shows why a single run can be misleading when speculative acceptance varies. [five-seed measurements](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:71)

Record the following together:

1. prompt tokens/s at short, normal, and near-target context;
2. generation tokens/s plus end-to-end time at each context;
3. speculative acceptance and chosen draft depth;
4. VRAM used/free on both cards, any GTT/spill, and desktop stability;
5. whether vision was enabled and whether images had already been processed in that server instance.

The last field matters because the local profile observed a persistent post-image prefill regression; without it, a later benchmark could falsely attribute a vision-state effect to the GPU split or cache type. [post-image regression note](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:142)

## Read-only checks for future runs

~~~bash
ps -eo pid,ppid,cmd | rg '[l]lama-server'
tail -n 250 /home/addison-gambhir/.local/state/local-llm/llama-server.log
cat /sys/class/drm/card0/device/{mem_info_vram_total,mem_info_vram_used,current_link_speed,current_link_width}
cat /sys/class/drm/card1/device/{mem_info_vram_total,mem_info_vram_used,current_link_speed,current_link_width}
/usr/bin/rocminfo
~~~

The DRM card-to-ROCm mapping must be checked before interpreting these counters because the local audit maps card0 to the R9700 and card1 to the XTX, while the llama.cpp device order is ROCm0 = XTX and ROCm1 = R9700. [mapping audit](/home/addison-gambhir/Desktop/local_llm/llama_server/docs/AUDIT_2026-08-16_OPTIMIZATIONS.md:178) [profile device order](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:2)

## Bottom line

Keep layer split and the current Q8 1,1 profile for long-context prefill. Use the tested F16 28,37 profile when interactive generation is the priority and its measured headroom remains available. Spend the next benchmark budget on deep-context MTP depth and build-version validation; do not spend it first on tensor parallel or broad split churn. This is the highest-confidence path from the local measurements and llama.cpp’s documented multi-GPU constraints. [local profile results](/home/addison-gambhir/Desktop/local_llm/llama_server/config/profiles/preset-38-27b-q8-rocm-dual-256k-ppopt.sh:68) [llama.cpp multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
