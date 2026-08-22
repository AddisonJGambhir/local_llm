WARNING: radv is not a conformant Vulkan implementation, testing use only.
ggml_vulkan: Found 3 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV NAVI31) (radv) | uma: 0 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 1 = AMD Radeon AI PRO R9700 (RADV GFX1201) (radv) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
ggml_vulkan: 2 = AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO) (radv) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
| model                          |       size |     params | backend    | ngl | n_ubatch |   main_gpu |  fa | dev          | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | ---------: | --: | ------------ | ------------ | --------------: | -------------------: |
/home/addison-gambhir/Desktop/local_llm/llama.cpp/src/llama-context.cpp:1713: GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed
[New LWP 871018]
[New LWP 871017]
[New LWP 871016]
[New LWP 871015]
[New LWP 871014]
[New LWP 871013]

This GDB supports auto-downloading debuginfo from the following URLs:
  <https://debuginfod.ubuntu.com>
Enable debuginfod for this session? (y or [n]) [answered N; input not from terminal]
Debuginfod has been disabled.
To make this setting permanent, add 'set debuginfod enabled off' to .gdbinit.
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1".
__syscall_cancel_arch () at ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S:56
⚠️ warning: 56	../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S: No such file or directory
#0  __syscall_cancel_arch () at ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S:56
56	in ../sysdeps/unix/sysv/linux/x86_64/syscall_cancel.S
#1  0x00007516de6a061c in __internal_syscall_cancel (a1=<optimized out>, a2=<optimized out>, a3=<optimized out>, a4=<optimized out>, a5=0, a6=0, nr=61) at ./nptl/cancellation.c:49
⚠️ warning: 49	./nptl/cancellation.c: No such file or directory
#2  __syscall_cancel (a1=<optimized out>, a2=<optimized out>, a3=<optimized out>, a4=<optimized out>, a5=a5@entry=0, a6=a6@entry=0, nr=61) at ./nptl/cancellation.c:75
75	in ./nptl/cancellation.c
#3  0x00007516de71cd4f in __GI___wait4 (pid=<optimized out>, stat_loc=<optimized out>, options=<optimized out>, usage=<optimized out>) at ../sysdeps/unix/sysv/linux/wait4.c:30
⚠️ warning: 30	../sysdeps/unix/sysv/linux/wait4.c: No such file or directory
#4  0x00007516de9464f3 in ggml_print_backtrace () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libggml-base.so.0
#5  0x00007516de9466a6 in ggml_abort () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libggml-base.so.0
#6  0x00007516ddd007c8 in llama_context::decode(llama_batch const&) () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libllama.so.0
#7  0x00007516ddd00a12 in llama_decode () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libllama.so.0
#8  0x00007516ded8c1cb in test_prompt(llama_context*, int, int, int) () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libllama-bench-impl.so
#9  0x00007516ded9f3e4 in llama_bench(int, char**) () from /home/addison-gambhir/Desktop/local_llm/llama.cpp/build-vulkan/bin/libllama-bench-impl.so
#10 0x00007516de62a601 in __libc_start_call_main (main=main@entry=0x5b492858c2f0 <main>, argc=argc@entry=33, argv=argv@entry=0x7ffc84a2ca18) at ../sysdeps/nptl/libc_start_call_main.h:59
⚠️ warning: 59	../sysdeps/nptl/libc_start_call_main.h: No such file or directory
#11 0x00007516de62a718 in __libc_start_main_impl (main=0x5b492858c2f0 <main>, argc=33, argv=0x7ffc84a2ca18, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7ffc84a2ca08) at ../csu/libc-start.c:360
⚠️ warning: 360	../csu/libc-start.c: No such file or directory
#12 0x00005b492858c325 in _start ()
[Inferior 1 (process 871012) detached]
