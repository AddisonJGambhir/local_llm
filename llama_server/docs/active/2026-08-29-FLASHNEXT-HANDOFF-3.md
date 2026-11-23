# Handoff 3: Qwen3.8-Flash-Next IQ3_XXS + KVarN now runs, long-context validation remains

Supersedes `2026-08-29-FLASHNEXT-HANDOFF-2.md`. Read that document for the
earlier benchmark tables and safety incident, but use this document for current
state. Several claims in Handoff 2 are now stale: mmap has been measured, the
Q4 model was deleted, tensor split was tested more carefully, and KVarN has
been integrated far enough to run Qwen's sparse QSA path.

## CURRENT STATUS -- READ THIS FIRST

The last KVarN run **worked**. It loaded IQ3_XXS with K6/V5 at 16K context,
served two complete 256-token requests, and was processing the real 9,060-token
prompt when the user asked for this handoff. It was stopped intentionally by
exact PID; it did not crash. No test `llama-server` was left running.

This is not ready to ship yet. The completed shallow runs were only:

    KVarN K6/V5, ctx 16384, pass 1: 28.34 t/s (256 tokens)
    KVarN K6/V5, ctx 16384, pass 2: 28.99 t/s (256 tokens)

The real prompt reached 3,072 tokens before intentional shutdown. Its first
2,048 prompt tokens ran at about 512 t/s. No deep-context decode number, final
prefill number, long-context VRAM measurement, or quality comparison exists
yet. Treat the implementation as a promising runtime proof, not validated.

The user's current requirements are explicit:

- Use the **IQ3_XXS model body**. Make this one good before considering another
  body quant.
- Keep **all experts in VRAM**. Only the 26.8 GiB PLE/n-gram table belongs on
  CPU via mmap.
- Ordinary Q4 KV is rejected. Do not revisit it. Q8 is the minimum ordinary KV
  quality; **KVarN K6/V5 is acceptable to evaluate**.
- The objective is materially more context than the current 82K Q8 setup.
- MTP was already a roughly 2x loss despite 99% acceptance. Leave it off.
- Independently measure claims. Do not trust either prior agent's arithmetic.

## MANDATORY BEFORE EVERY MODEL LAUNCH

    pgrep -x llama-server
    ss -ltnp | grep -E ':(1234|8080|TEST_PORT) '
    free -g
    cat /sys/class/drm/card{0,1}/device/mem_info_vram_used

`pgrep -x llama-server` must be empty. If the user's server exists, stop and
ask; a different port does not isolate RAM or VRAM. Require at least 30 GiB
MemAvailable. Port 8080 is currently a Docker searxng listener, not a model.

The 7900 XTX is card1 and drives the desktop. Keep its settled usage below
24,200 MiB. Startup spikes can exceed this temporarily; do not kill a run based
only on a load spike. Kill only the exact test PID. Never use `pkill -f` or
`pkill -x bash`.

## MODEL AND STORAGE STATE

The complete model is:

    llama_server/models/Qwen3.8-Flash-Next/UD-IQ3_XXS/
    Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf

Shard sizes were verified as 10,946,624; 49,567,921,344; and 32,382,955,968
bytes. The directory consumes about 77 GiB.

The Q4 body model was deliberately deleted at the user's request and recovered
about 88 GiB. Do not download it again. `UD-Q3_K_XL` remains but is about 84
GiB and cannot hold all experts in VRAM. The IQ3_XXS body minus its fixed PLE
table is about 49.6 GiB and does fit across the two GPUs.

The PLE tensor is one `per_layer_token_embd.weight` tensor, measured from
instrument dumps at 27,466 MiB. `-ot 'per_layer_token_embd\.weight=CPU'` plus
`--load-mode mmap --tensor-read-lazy on` leaves its settled process RSS near
1.0 GiB. It is genuinely file-backed; physical disk reads after load were only
about 174 MiB across the benchmark requests. SSD offload is therefore not
streaming PLE lookups at generation time; mmap lets the page cache own the
table.

## INDEPENDENT Q8 BASELINE (MMAP VERIFIED)

Build `llama.cpp-qwen4exp` at `aa1b06d0c`, IQ3_XXS, ctx 81,920, layer split
385/615, ubatch 1024, batch 2048, q8_0 K/V, FA, PLE on CPU:

    shallow pass 2: 41.488 t/s
    depth 9,060:    27.976 t/s
    prefill 9,060: 640.792 t/s
    settled RSS:   1,021,104 KiB
    startup VmHWM: 35,055,256 KiB

The startup high-water mark is large even though settled RSS is small. Do not
mistake VmHWM for the steady PLE footprint.

The IQ3_S Vulkan register-spill patch was also tested in
`llama.cpp-qwen4exp-iq3spill` (`48a9046e2` plus test-only `c98958507`): shallow
41.048, depth-9,060 29.426, prefill 646.54 t/s. The deep +5.2% result is not
conclusive because generation diverged; shallow was slightly worse.

## KVarN SOURCE TREE -- UNCOMMITTED, PRESERVE IT

Current tree:

    llama.cpp-beellama-044/
    branch: codex/qwen4exp-kvarn
    upstream HEAD: 3746fb185faa06104e0413236030a5e385bc510c
    build: llama.cpp-beellama-044/build-vulkan/bin/llama-server

This is BeeLlama v0.4.4 from `https://github.com/Anbeeld/beellama.cpp` with
Qwen4Exp integrated. BeeLlama natively supports KVarN on Vulkan for head widths
128/256/512 and K/V bit widths 2,3,4,5,6,8. K6/V5 is its conservative quality
choice and respects the user's 5/6 floor.

Two Qwen commits were applied with `git cherry-pick -n`, so the imported Qwen
changes are staged but deliberately uncommitted:

    6c84c7d5d  Qwen4Exp architecture implementation (PR #27742)
    aa1b06d0c6 graph split reduction (PR #27880)

Do not reset, clean, or re-cherry-pick this tree. `git status` intentionally
shows a large staged Qwen import plus smaller unstaged KVarN/QSA adaptations.
No commit was made because BeeLlama's AGENTS.md says not to commit unless asked.

### Local integration changes already made

1. `llama_memory_hybrid_idx` can construct either a standard attention cache
   or `llama_kv_cache_kvarn`, passing KVarN/tail/batch parameters through
   `llama_model::create_memory`.
2. Its batch path prepares attention metadata once and makes the QSA indexer
   mirror the exact same slot layout.
3. Qwen's custom sparse `build_attn_qsa` now runs the KVarN attention planner,
   chooses native versus materialized reads, applies the required WHT query
   rotation, tags the FlashAttention domain, and reverses value rotation.
4. Pruned exact-tail index inputs are skipped safely when the specialized QSA
   graph does not allocate them.
5. `llama_kv_cache_context::get_prev_tokens` is now virtual; the KVarN wrapper
   delegates to its actual metadata context. This fixes PLE's null dereference.
6. `ggml-backend.cpp` contains temporary unsupported-node diagnostics. They are
   useful but not semantically required and can be removed after validation.
7. State restore for hybrid-index KVarN is not finished: the earlier conflict
   resolution preserves ordinary inference, but mirrored layout restore is not
   validated. Do not test prompt-cache state restoration before auditing it.

The build currently succeeds:

    cmake --build llama.cpp-beellama-044/build-vulkan -j 12 --target llama-server

## FAILURE PROGRESSION (DO NOT REPEAT THE DIAGNOSIS)

All logs are preserved under `/tmp` for this boot/session.

1. Initial K6/V5 launch aborted during graph assignment:

       unsupported node ... op=FLASH_ATTN_EXT ...

   Log: `/tmp/flashnext-kvarn65-diagnostic/server.log`

2. Bee's built-in `GGML_KVARN_DEBUG_ROUTES=1` exposed:

       kvarn-k=1 kvarn-v=1 domain=0 rotated=0 tail-ok=1

   Qwen's custom QSA path bypassed the KVarN planner and left the domain AUTO.
   Log: `/tmp/flashnext-kvarn65-route-debug/server.log`

3. After adding the QSA planner, warmup aborted because `self_tail_idxs` existed
   as a graph object but had been pruned and had no buffer. The set-input guard
   fixed it. Log: `/tmp/flashnext-kvarn65-qsa-fix/server.log`

4. The next run segfaulted in:

       llama_kv_cache::get_prev_tokens
       llm_graph_input_ple::set_input

   GDB proved PLE was calling a non-virtual inherited method whose KVarN wrapper
   base pointer was null. The virtual delegation fix described above fixed it.
   Log: `/tmp/flashnext-kvarn65-qsa-fix2/server.log`

5. Final run loaded and generated successfully. Route diagnostics showed
   `portable-native` K6/V5 on both Vulkan0 and Vulkan1 for 17-token prefill,
   single-token decode, and 1,024-token real-prompt chunks. Log and responses:

       /tmp/flashnext-kvarn65-qsa-fix3/server.log
       /tmp/flashnext-kvarn65-qsa-fix3/d0-pass1.json
       /tmp/flashnext-kvarn65-qsa-fix3/d0-pass2.json

The Q8 control on the same merged Bee/Qwen build worked at 40.97 and 41.69 t/s,
proving the basic Qwen port was sound before the KVarN-specific fixes:

       /tmp/flashnext-bee-q8-control/server.log

## EXACT REPRODUCTION HARNESS

The temporary harness is `/tmp/flashnext_mmap_verify.sh`. It performs the
mandatory process/RAM/VRAM checks, launches an exact PID, runs two shallow
requests and the real 9,060-token prompt, captures RSS/I/O/VRAM snapshots, and
cleans up only its PID.

Use this for the present proof configuration:

    GGML_KVARN_DEBUG_ROUTES=1 \
    FLASHNEXT_BIN=/home/addison-gambhir/Desktop/local_llm/llama.cpp-beellama-044/build-vulkan/bin/llama-server \
    FLASHNEXT_PORT=18250 \
    FLASHNEXT_RUN=/tmp/flashnext-kvarn65-validation \
    FLASHNEXT_CTX=16384 \
    FLASHNEXT_CTK=kvarn6 \
    FLASHNEXT_CTV=kvarn5 \
    /tmp/flashnext_mmap_verify.sh

The effective launch flags are:

    -c CTX -ngl 99 --fit off --device Vulkan0,Vulkan1
    --split-mode layer --tensor-split 385,615
    -ub 1024 -b 2048 --flash-attn on
    --cache-type-k kvarn6 --cache-type-v kvarn5
    --load-mode mmap --tensor-read-lazy on
    -ot 'per_layer_token_embd\.weight=CPU'

`--fit off` is intentional. Bee's automatic fit probe aborted on the new graph
before the QSA work, and placement is already known. `--device` is mandatory;
Vulkan2 is the CPU device and must not be included.

## NEXT TASKS, IN ORDER

1. Re-run the 16K validation to completion without interruption. Keep route
   debug for this one. Record both shallow outputs, full 9,060-token pp/tg,
   settled RSS, both settled VRAM values, and whether output is coherent.
2. Compare fresh deterministic Q8 and K6/V5 outputs for gross correctness. They
   need not match bit-for-bit, but KVarN must not be gibberish or lose PLE
   history. The native route currently reports `tail_stride=0`; audit that the
   intrinsic exact 128-token suffix is actually served from KVarN stage storage
   as Bee intends for this specialized QSA path.
3. Sweep context upward with K6/V5 while preserving all expert placement and
   mmap PLE. Suggested probes: 96K, 104K, 112K, then bracket the cliff. Do not
   jump straight to 128K: Q4 KV at 128K had only about 36 MiB XTX margin, and
   K6/V5 stores 11 total bits versus Q4's 8, so 128K likely exceeds the desktop
   safety ceiling. Measure rather than extrapolate.
4. Once the maximum stable context is found, benchmark depth near the target,
   not only depth 9K. KVarN shallow decode is currently about 29 t/s versus Q8's
   41.5 t/s, so quantify the context-versus-speed trade before shipping it.
5. Remove or retain the scheduler diagnostic deliberately, run focused tests,
   then create a clean commit only if the user asks. The user-facing preset
   should not be changed until KVarN passes long-prompt and context-cliff tests.

## OTHER CLOSED PATHS

- Ordinary Q4 KV at 128K technically ran but is below the user's quality floor
  and used about 24,164 MiB on the XTX, leaving only ~36 MiB. Do not ship it.
- Tensor split with a 32 GiB Vulkan suballocation block worked for a 27B probe,
  but IQ3_XXS still aborts with `multi buffers are not supported by the meta
  backend` because its sharded tensors exceed the path's allocation contract.
- Row split is unsupported on RADV. ROCm does not fit. Moving weights off the
  XTX overloads the R9700. Keep the known 385/615 layer placement.
- The PLE table is mmap/file-backed already. There is no useful SSD streaming
  flag left to discover; async PLE prefetch would require implementation work.
- MTP is a net loss on this hardware and is unrelated to the context objective.

## PRIMARY REFERENCES

- BeeLlama v0.4.4: `https://github.com/Anbeeld/beellama.cpp`
- KVarN algorithm/benchmarks: `https://anbeeld.com/articles/kvarn-kv-cache-implementation-and-benchmarks`
- Qwen4Exp PR: `https://github.com/ggml-org/llama.cpp/pull/27742`
- Graph split PR: `https://github.com/ggml-org/llama.cpp/pull/27880`
- Qwen reference: `https://github.com/QwenLM/Qwen3.8-Flash-Next`
- Performance discussion: `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3`
