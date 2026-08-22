# Adaptive MTP + Vulkan tensor split — 2026-08-22

Qwen3.8-27B Q8_0, RX 7900 XTX + AI PRO R9700, Vulkan, `-c 98304`, f16 KV, ub 1024.
Server-method: real prompts through `/completion`, 2048 tokens, `temp 0` + `top_k 1`
+ `ignore_eos`, agentic prompt (1958 tok of repeated tool calls). Full data in
`records/MASTER_RESULTS.csv` (regenerate: `src/benchmarks/build_master_results.py`).

## Headline

| build | split | spec | tg | pp | peak XTX |
|---|---|---|---|---|---|
| **merged-tp-adaptive** | tensor 50/50 | **draft-mtp-adaptive** | **81.67** | 622.7 | 20,388 |
| 7c80213c3-rebased | layer 28/37 | draft-mtp-adaptive | 73.20 | 828.2 | 17,866 |
| 13ac90156-pr25051 | tensor 50/50 | draft-mtp n3 | 57.06 | — | 19,674 |
| merged-tp-adaptive | tensor 50/50 | draft-mtp n3 | 56.99 | 622.0 | 19,652 |
| 13ac90156-pr25051 | layer 28/37 | draft-mtp n3 | 53.26 | — | 17,556 |
| 7c80213c3-rebased | layer 28/37 | draft-mtp n3 | 52.95 | 829.6 | 17,575 |

**Adaptive MTP is worth +43% decode on tensor split and costs nothing on prefill**
(622.7 vs 622.0 pp) for +736 MiB on the XTX. Layer split confirms it independently
at +38.3% (n=7 each, exact permutation p=0.00058, no overlap between groups).

## What was built

`llama.cpp-tp-adaptive` = PR #25051 (Vulkan tensor allreduce) with PR #27210
(adaptive MTP) ported onto it. Six conflicts in `common/speculative.cpp`:

- enum/name maps: added `MTP_ADAPTIVE` only; `DFLASH`/`DSPARK` do not exist in this base
- init: kept this base's `LOG_INF` (`SPC_TRC`/`SPC_DBG` are newer master macros)
- **137-line conflict was master API churn**, not the feature — `common_base_params_to_speculative`,
  `common_speculative_init_from_params` etc. are referenced 0 times by `server-context.cpp`,
  `arg.cpp`, or the 92-line `speculative-adaptive.h`. Dropped entirely; builds clean.
- registration: kept explicit style, `COMMON_SPECULATIVE_TYPE_COUNT` 9 -> 10

Do NOT apply `vk-queue-unique-ptr-port.patch` to this tree — that exists only because
rebasing onto master changed Vulkan queues to `unique_ptr`. This base uses raw pointers.

## Dead ends (do not retry)

- **Rebasing #25051 onto master b21e4de74 is worthless and harmful.** Decode is identical
  to the old base (layer n3: 52.95 vs 53.26, within noise), and `-sm tensor` aborts at
  `ggml-backend-meta.cpp:1156: multi buffers are not supported by the meta backend`.
  Master's allocator hands the meta backend multiple buffers; #25051 cannot take them.
  Worth reporting upstream.
- Earlier speculation sweep (13 configs x 2 content x 3 reps): nothing beat `n3/p_min 0`.
  Every `p_min > 0` lost on both content types; `ngram-mod` chaining lost.

## Caveats

- `novel` content measured for adaptive on layer only ([79.67,77.88,79.98,53.87]) and
  never re-tested. Treat as provisional.
- **This box does not produce deterministic output.** Three runs of one config at
  `temp 0`/`top_k 1` gave three different SHA-256 hashes. Output-diffing cannot be used
  to verify a speculator is doing full verification; use acceptance counters instead.
- Adaptive was not verified to preserve output. It is byte-identical to n3 at 512 tokens
  but diverges at 2048 — consistent with the nondeterminism above rather than a bug,
  but not proven.
- Not yet sized for 256k f16 context; all numbers above are `-c 98304`.
