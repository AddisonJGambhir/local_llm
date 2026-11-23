# HANDOFF: PR #25051 full test matrix (to run before the MR handoff)

Owner: main agent. Purpose: generate as much clean, depth-tagged data as possible
for the llama.cpp PR #25051 Vulkan tensor-parallel MR, so the maintainer sees a
complete shallow-vs-deep comparison across every split mode.

Do NOT commit or post anything. Run, collect, and write numbers into the data
tables below. When done, feed the completed tables back so the results doc and
PR comment can be finalized.

---

## Exact links

- PR (the one we are testing): https://github.com/ggml-org/llama.cpp/pull/25051
  - Title: `vulkan: add allreduce function with cross-device CPU proxy and fix Tensor Parallel crash [EXPERIMENTAL]`
  - Author: pwilkin, state: open, draft: yes, last updated 2026-08-16, ~48 comments.
- PR comments (full thread, paginate):
  `gh api repos/ggml-org/llama.cpp/issues/25051/comments --paginate`
  (30 comments confirmed via `gh api .../issues/25051/comments --jq 'length'`.)
- Related upstream context (cite in the MR, do not re-test):
  - #19378 (merged) tensor mode as "meta backend", CUDA-only real allreduce (NCCL).
  - #26490 (active) tensor mode for more arches.
  - #27334 weak-PCIe 4xRX580 Vulkan tensor 8 t/s vs 36 layer (not representative; note only).
  - #22197 multi buffers unsupported -> Vulkan segfault.
  - #24219 (RFC, closed) Intra-Prompt Pipeline Scheduling.

The full local results already gathered: `docs/PR25051_AMD_DUAL_GPU_RESULTS.md`.
The superseded Q6-only doc (do not delete, do not edit): `docs/Q6_VULKAN_TENSOR_25051_RESULTS.md`.
The draft PR comment: `docs/PR25051_COMMENT_DRAFT.md`.

---

## Hard rules (do not skip)

1. EVERY number must carry a context-depth provenance tag. Legend (use verbatim):
   - `ctx 0`     = llama-bench pp8192, context allocated but KV empty. Allocation overhead.
   - `pp8192 @ c N` = llama-bench pp8192 with N-token context allocated, KV empty.
   - `d65536`   = marginal prefill rate with context already at depth 65536 (real tokens in KV).
   - `live`     = llama-server serving, MTP on, fixed prompts.
2. ALL tests run WITH and WITHOUT MTP. See "MTP protocol" below: llama-bench CANNOT
   run speculation, so without-MTP = llama-bench; with-MTP = live llama-server probe.
3. AT LEAST 3 TRIALS per config combo.
   - llama-bench: use `-r 3`.
   - live MTP probe: 3 samples (3 fixed prompts, or 1 prompt x 3 passes), record each.
4. No em dashes anywhere in results.
5. Do not compare MTP-on numbers against MTP-off numbers, and do not compare MTP-on
   numbers across configs taken on different days (draft acceptance is stochastic at
   temp 1.0 and ranges wide). MTP-on is indicative of the serving path only.
6. Record peak VRAM per GPU for every live run (XGT/XGMI ceiling matters on the XTX:
   keep GPU0 at or under ~24,000 MiB or it silently collapses, see known-issues).

---

## Environment (verify before starting)

- GPU 0: RX 7900 XTX (NAVI31, gfx1100) 24560 MiB, drives desktop. CPU-attached
  (root port `00:01.1`, Gen5 x16 capable); the card is Gen4, so it trains at 16 GT/s x16.
- GPU 1: AI PRO R9700 (GFX1201, gfx1201) 32624 MiB. Behind the chipset
  (`00:02.1` -> two cascaded Promontory switches). The CPU uplink is PCIe 4.0 x4 and
  that is the hard ceiling for everything on this card. The board (X870E AORUS MASTER)
  has only ONE CPU-attached x16 slot and no bifurcation, so a second compute card
  necessarily lands on this same x4 path.
- GPU 2: Ryzen iGPU (granite ridge) -- leave unused.
- CPU Ryzen 9 9950X3D (16C/32T), 64 GB RAM.
- Interconnect -- MEASURED 2026-09-05 (`hipMemcpyPeer`, `HIP_VISIBLE_DEVICES=0,1`).
  **P2P WORKS**: `hipDeviceCanAccessPeer` YES both directions, `hipDeviceEnablePeerAccess`
  succeeds. (This corrects an earlier claim here that there was no P2P and that all
  allreduce round-trips through system memory -- it does not.) No xGMI; PCIe only
  (`rocm-smi --showtopo`: 2 hops, weight 40).

  | path | measured |
  | --- | ---: |
  | XTX <-> host (Gen4 x16, direct) | 28.2 GB/s |
  | R9700 <-> host (Gen4 x4, chipset) | 7.0 GB/s |
  | GPU <-> GPU peer copy | 6.9 GB/s |
  | 10 KiB peer copy (latency) | 22.8 us |

  6.9 GB/s is 88% of Gen4 x4 theoretical -- there is no tuning headroom left.
  **Bandwidth is NOT the limiter for decode.** A 27B TP=2 all-reduce moves 10 KiB
  (n_embd 5120, bf16); 130 per token = 1.3 MB/token = ~78 MB/s at 60 tok/s, i.e. 88x
  headroom. The cost is LATENCY: only 1.5 us of that 22.8 us is transfer. Chained,
  ~4.0 ms of comm per token (upper bound: sequential blocking copies; real RCCL fuses),
  a ~247 tok/s ceiling. Prefill is where the x4 actually bites: a 2048-token chunk
  moves 2.66 GB of allreduce = ~389 ms.
  Benchmarks: `src/benchmarks/p2p_bw.hip` (streaming) and `src/benchmarks/p2p_lat.hip`
  (small-payload latency + simulated TP=2 allreduce chain). Build and run:
  ```
  /opt/rocm/bin/hipcc --offload-arch=gfx1100 --offload-arch=gfx1201 -O2 \
      llama_server/src/benchmarks/p2p_lat.hip -o /tmp/p2p_lat
  HIP_VISIBLE_DEVICES=0,1 /tmp/p2p_lat   # 0,1 excludes the iGPU
  ```
- OS Ubuntu 26.04, kernel 7.0.0-29-generic. Vulkan driver RADV (prints
  "not a conformant Vulkan implementation, testing use only" at startup: expected, ignore).
- ROCm 7.1.52801, HIP/clang 21, `GGML_HIP_ARCHITECTURES=gfx1100;gfx1201`, RCCL OFF by default.

### Builds and binaries (use exactly these)

| Backend | Binary | Notes |
| --- | --- | --- |
| Vulkan stock (no tensor) | `llama.cpp/build-vulkan/bin/llama-bench` | `-ts` is SILENTLY IGNORED here; every weight lands on device 0. Use only for layer/row. |
| Vulkan PR (tensor-capable) | `llama.cpp-vulkan-tp/build/bin/llama-bench` | Branch `pr25051`, HEAD `13ac90156`. MUST set `LD_LIBRARY_PATH=$PWD/llama.cpp-vulkan-tp/build/bin` or it dies on missing `libllama-bench-impl.so`. This is the one that actually runs #25051. |
| ROCm | `llama.cpp/build/bin/llama-bench` | Commit `030ebb558`. layer/row/tensor all supported. |

Set once per Vulkan PR run:
```
export LLM_ROOT=/home/addison-gambhir/Desktop/local_llm
export LD_LIBRARY_PATH="$LLM_ROOT/llama.cpp-vulkan-tp/build/bin:$LD_LIBRARY_PATH"
```

### Models
- Q6_K: `$LLM_ROOT/llama_server/models/Qwen3.8-27B-Q6_K.gguf` (21.30 GiB, imatrix). Full 262144 ctx fits.
- Q8_0: `$LLM_ROOT/llama_server/models/Qwen3.8-27B-Q8_0.gguf` (26.9 GiB). Vulkan tensor caps at c=200704; Q6_K buys full 262144.

### MTP protocol (the only way to get with-MTP numbers)

llama-bench has no speculation flags. With-MTP must be a live `llama-server` (or
`llama-cli` spec) run with this fixed protocol, so runs are comparable:
- Env (from the shipped profiles): `LLAMA_SPEC_MODE=mtp`, `LLAMA_SPEC_DRAFT_N_MAX=3`,
  `LLAMA_SPEC_DRAFT_TYPE_K=q8_0`, `LLAMA_SPEC_DRAFT_TYPE_V=q8_0`, `LLAMA_DRAFTER=""`.
  MTP-off = the exact same server with `LLAMA_SPEC_MODE` unset/empty.
- Fixed protocol: 3 fixed natural-language prompts, 400 tokens generated each,
  MTP n3, record t/s and acceptance for each of the 3 samples (3 trials).
- Keep prompts IDENTICAL across every MTP-on config so they are the same workload.
  Reuse the four-prompt set referenced in the Q6 ROCm tensor profile
  (`preset-38-27b-q6k-rocm-dual-f16-256k-mmproj-tensor-decode.sh`, the "fixed four
  natural prompts x 400 tokens x two passes" block). If those prompt files are not
  on disk, pick 3 stable 300-500 token prompts and store them in
  `llama_server/config/mtp_prompts/` so the run is reproducible, then use only those 3.
- Record: t/s, acceptance %, mean accepted tokens, peak GPU0 MiB, peak GPU1 MiB.

---

## Test matrix to run

Axes:
- backend: {vulkan-stock, vulkan-pr, rocm}
- split:   {layer, row, tensor}
- quant:   {Q6_K, Q8_0}
- ctx:     {0, 65536}
- mtp:     {off, on}
- trials:  3 (bench `-r 3`; MTP 3 samples)

Split-mode support (what is actually runnable, "if possible" resolved):

| backend \ split | layer | row | tensor |
| --- | --- | --- | --- |
| vulkan-stock | yes | yes | NO (-ts silently ignored; skip, do not fake it) |
| vulkan-pr    | yes | yes | YES (this is #25051) |
| rocm         | yes | yes | yes |

Split ratios to use (keep it canonical; note ratio in the tag):
- tensor: 50/50 (primary, pp peak). Add 70/30 as a second tensor point for the tg peak.
- layer: use the shipped ratios already in the profiles:
  - Q6_K: ROCm layer 38/27; Vulkan layer 28/37.
  - Q8_0: ROCm layer 1/1 (main-gpu 0) ; Vulkan layer 28/37.
- row: use main-gpu 1 (R9700) to mirror the tensor/layer convention, and also try
  main-gpu 0 for Q8_0 Vulkan to match how row is typically offloaded. If row is
  flaky or unsupported on a given build, record "n/m" with the reason, do not guess.

### Canonical llama-bench command (MTP-off, the with-out-MTP half)

Run both ctx=0 and ctx=d65536 for each config:
```
# depth 0 (KV empty): allocate context c, prefill 8192, decode 256, 3 trials
$BENCH -m "$MODEL" -sm $SPLIT -ts $RATIO -mg 1 -ngl 999 -c 8192 \
  -p 8192 -n 256 -d 0 -b 2048 -ub 1024 -ctk f16 -ctv f16 -r 3

# depth 65536 (real tokens in KV at 65536): use -d 65536, keep -c large enough
$BENCH -m "$MODEL" -sm $SPLIT -ts $RATIO -mg 1 -ngl 999 -c 65536 \
  -p 8192 -n 256 -d 65536 -b 2048 -ub 1024 -ctk f16 -ctv f16 -r 3
```
Where `$BENCH` is one of the three binaries above and `$SPLIT`/`$RATIO` per the table.
For vulkan-pr set `LD_LIBRARY_PATH` first. For row use `-sm row` (drop `-ts`).
For vulkan-stock tensor: DO NOT RUN (unsupported).

Use `-ub 1024` for Q6_K and the 200k Vulkan tensor Q8_0 (matches shipped); note if you
change it. For Q8_0 Vulkan tensor, cap `-c` at 200704 (ceiling); for the d65536 row
that cap is fine (65536 << 200704).

### With-MTP half (live probe)

For each config, start `llama-server` with the matching profile flags (or the shipped
profile, with only split/quant/ctx changed), `LLAMA_SPEC_MODE=mtp`, `LLAMA_SPEC_DRAFT_N_MAX=3`,
then hit it with the 3 fixed prompts x 400 tokens x 3 samples. Record t/s, acceptance,
peak VRAM both GPUs. Do the same with `LLAMA_SPEC_MODE` unset for the MTP-off live baseline
if you want a live (not bench) without-MTP number; optional but preferred for parity.

---

## Data tables to fill in (paste results here as you go)

Tag every cell. `ctx 0` = llama-bench pp8192 KV empty. `d65536` = marginal at depth 65536.
`live` = llama-server MTP n3. `n/m` = not measured (give reason). 3-trial means; record
min-max if spread > 5%.

### Q6_K

| backend | split | ratio | pp ctx 0 | pp d65536 | tg ctx 0 | tg d65536 | tg MTP-off live | tg MTP-on live | acceptance | peak G0 | peak G1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vulkan-pr | tensor | 50/50 | 889.6 (have) | TODO | 23.2 (have) | TODO | 43.41 (have) | 43.4 | ~53%? confirm | 22,782 | 20,792 |
| vulkan-pr | tensor | 70/30 | 729.8 (have) | TODO | 29.2 (have) | TODO | TODO | TODO | | | |
| vulkan-pr | layer | 28/37 | TODO | TODO | TODO | TODO | TODO | TODO | | | |
| vulkan-pr | row | mg1 | TODO | TODO | TODO | TODO | TODO | TODO | | | |
| vulkan-stock | layer | 28/37 | TODO | TODO | TODO | TODO | TODO | n/m (no TP) | | | |
| vulkan-stock | row | mg1 | TODO | TODO | TODO | TODO | TODO | n/m (no TP) | | | |
| rocm | layer | 38/27 | TODO | 353.3 (d131k have) | TODO | 18.1 (have) | 32.231 (have) | 32.2 | 51.62% | TODO | TODO |
| rocm | row | mg1 | TODO | TODO | TODO | TODO | TODO | TODO | | | |
| rocm | tensor | 1/1 | 294.5 (d65k have) | 294.5 (have) | 27.3 (have) | TODO | 43.576 (have) | 43.6 | 51.01% | 22,211 | TODO |

### Q8_0

| backend | split | ratio | pp ctx 0 | pp d65536 | tg ctx 0 | tg d65536 | tg MTP-off live | tg MTP-on live | acceptance | peak G0 | peak G1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vulkan-pr | tensor | 50/50 | 946.7 (have) | 561.7 (have) | 21.4 (have) | TODO | TODO | 40.29 (have) | 54.5% | 23,538 | TODO |
| vulkan-pr | tensor | 70/30 | 801.1 (have) | TODO | 25.9 (have) | TODO | TODO | TODO | | | |
| vulkan-pr | layer | 28/37 | 825 (6.2k harness, have) | TODO | n/m | TODO | 35.0 (have) | 35.0 | TODO | TODO | TODO |
| vulkan-pr | row | mg1 | TODO | TODO | TODO | TODO | TODO | TODO | | | |
| vulkan-stock | layer | 28/37 | TODO | TODO | TODO | TODO | TODO | n/m (no TP) | | | |
| vulkan-stock | row | mg1 | TODO | TODO | TODO | TODO | TODO | n/m (no TP) | | | |
| rocm | layer | 1/1 | 1099 (d0 have) | 509.4 (have) | TODO | 18.4 (have) | 34.352 (have) | 34.4 | TODO | TODO | TODO |
| rocm | row | mg1 | TODO | TODO | TODO | TODO | TODO | TODO | | | |
| rocm | tensor | 45/55 | 275.2 (d65k have) | 275.2 (have) | 22.6 (have) | TODO | 44.813 (have) | 44.8 | TODO | 24,487 | TODO |

"(have)" = already in `docs/PR25051_AMD_DUAL_GPU_RESULTS.md`; keep, do not re-measure
unless you are filling a `TODO` in the same row.

---

## Priorities if you must cut scope (do not cut before trying all of these)

1. vulkan-pr tensor 50/50 and 70/30, both quants, ctx 0 + d65536, MTP on + off.
   (This is the actual PR; it is the whole point.)
2. rocm tensor, both quants, ctx 0 + d65536, MTP on + off. (the comparison baseline)
3. layer and row for all three backends, Q6_K and Q8_0, ctx 0 + d65536.
4. vulkan-stock tensor: skip (unsupported), but log one line of proof that `-ts` is
   ignored (show `ggml_vulkan` device weights all on device 0) so the MR can state it.

## Known issues to watch (record, do not "fix")

- XTX GTT cliff: at c=262144 ts 55/45 the XTX hits 24,180 MiB -> silent collapse to
  ~140 pp / 27 tg (not an OOM, not a crash). Keep GPU0 <= ~24,000 MiB. If a config
  peaks above, note peak MiB and mark the run as collapsed, do not report its speed.
- Q8_0 Vulkan tensor context ceiling is 200704 (225280 fits but with 178 MiB spare, rejected).
- RADV non-conformant warning at Vulkan startup is expected; ignore.
- MTP draft acceptance is stochastic at temp 1.0; report mean + range over the 3 samples,
  never a single-sample acceptance as the headline.

## Definition of done

- Every `TODO` in both tables is either a measured number (with tag + 3-trial min-max)
  or an explicit `n/m` with a one-line reason.
- Every config has an MTP-on and MTP-off number (or a stated reason a side is impossible).
- Peak VRAM both GPUs recorded for every live run.
- Results appended to `docs/PR25051_AMD_DUAL_GPU_RESULTS.md` (or a new dated file) and
  `docs/PR25051_COMMENT_DRAFT.md` updated to match. Nothing committed, nothing posted.
