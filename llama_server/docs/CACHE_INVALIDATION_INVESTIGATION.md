# Investigation brief: intermittent full-context reprocessing on llama-server

Copy everything below this line as the prompt for the investigating agent.

---

## Task

A local `llama-server` deployment intermittently throws away a 120k–190k token
KV cache and reprocesses the entire prompt from scratch, costing ~96 seconds.
Find the root cause and propose a fix. Several plausible explanations have
already been eliminated by measurement, do not re-derive them, and do not
propose a fix that any of the listed evidence already contradicts.

## Environment

- **llama.cpp**: build **10358** (`030ebb558`), pinned deliberately. Upstream
  10450 (`ece963f41`) was built and benchmarked on this host and is **slower**
  (-4.8% prefill, -4.0% decode, MTP acceptance 61.9% → 55.7%), most likely from
  `ggml-hip: remove -funsafe-math-optimizations` (#26696). Source tree is at
  `/home/addison-gambhir/Desktop/local_llm/llama.cpp`.
- **GPUs**: ROCm0 = RX 7900 XTX (gfx1100, 24 GiB, **also drives the desktop**),
  ROCm1 = AI PRO R9700 (gfx1201, 32 GiB). Host RAM 59 GiB, ~30 GiB available.
- **Model**: `Qwen3.6-35B-A3B-Q8_0.gguf`, arch `qwen35moe`, 41 layers, MoE
  (256 experts / 8 active), **hybrid attention** (`full_attention_interval = 4`,
  so ~10 of 41 layers keep context-scaling KV; the rest are recurrent/linear),
  built-in MTP (`nextn_predict_layers = 1`), native context 262144.
- **Client**: the Pi coding agent (`pi -p`), which sends
  `temperature 0.6`, `seed 4294967295` (random), via `/v1/chat/completions`.

### Server config (all values benchmark-derived)

```
-c 262144  --parallel 1  --ubatch-size 1024  --batch-size 12288
--tensor-split 18,22  --main-gpu 1  --split-mode layer
--cache-type-k f16 --cache-type-v f16
--cache-reuse 256   --cache-ram 16384
--spec-type draft-mtp --spec-draft-n-max 2
--timeout 3600  --flash-attn on  --no-mmap
```

**Hard constraint:** the XTX must stay under ~24200 MiB. It drives the desktop;
at 24388 MiB the compositor was evicted to GTT and the screen dropped to ~3 fps.
This is *not* an OOM boundary, llama.cpp will load past it and take the session
down. Predict memory before loading any new config.

## Symptom

Normal operation reuses the cache and reprocesses a handful of tokens:

```
get_availabl: selected slot by LCP similarity, f_sim_best = 0.999, f_keep = 1.000
print_timing: prompt eval time = 776.88 ms / 134 tokens
```

Intermittently it does this instead:

```
release:      t5670 | stop processing: n_tokens = 119064
get_availabl: t-1   | selected slot by LRU, t_last = 25082363673
print_timing: t5721 | prompt eval time = 96393.37 ms / 119389 tokens (1238.56 t/s)
```

119,389 tokens reprocessed, 96 seconds. `t_last` is a real timestamp (not `-1`),
so this is **not** a server restart. Observed repeatedly, minutes apart.

Observed `f_sim_best` values over one session, in order:
`1.000, 1.000, 0.993, 0.969, 0.962, 0.618, 0.930, 0.838`, and occasionally the
LRU fallback, which means `f_sim` fell below the `0.100` threshold.

## Already ruled out, with evidence

1. **A changing prompt prefix** (timestamp, cwd, git state, injected reminder).
   Captured every prompt with `--log-prompts-dir` and diffed consecutive pairs.
   Within a conversation they are **100% pure appends**, e.g. prev 750,712
   chars, new 751,793 chars, common prefix 750,712 (100.0%). Nothing changes at
   the front.

2. **Pi rewriting conversation history.** Same diff evidence.

3. **Context switching per se.** Direct A → B → A test at ~39k tokens per
   context: switching back cost **4 tokens**, i.e. the RAM prompt cache saved and
   restored the whole context correctly. (A larger 120k-per-context repeat of
   this test was in flight when this brief was written, check
   `scratchpad/bench/ctxswitch_large.log` for the result before assuming.)

4. **The hybrid-model checkpoint rewind.** That is a *different, milder* failure
   also present on this box: a mid-history edit rewinds to the previous
   checkpoint (`--checkpoint-min-step`, default 8192), producing e.g. a
   5,242-token reprocess at `f_sim 0.858`. The 119k event is a reprocess from
   zero, not a rewind.

5. **A server restart.** `t_last = -1` marks those; the events in question have
   real `t_last` values.

## Relevant source (build 10358)

- `tools/server/server-context.cpp:1591`, `get_available_slot()`; LCP-similarity
  selection at ~1606-1646, LRU fallback at ~1657-1677.
- `tools/server/server-context.cpp:1680-1700`, on **either** path, if
  `update_cache` holds it calls `prompt_save()` then `prompt_load()`, and only
  calls `prompt_clear()` (⇒ full reprocess) when `prompt_load` returns false.
- `tools/server/server-context.cpp:281`, `prompt_load()` wraps
  `prompt_cache.load(...)` and logs `"failed to load prompt from cache"` on
  failure (SLT_WRN).
- `tools/server/server-task.cpp:1664`, `server_prompt_cache::alloc()`. Note it
  **silently skips** entries larger than the whole budget:
  `"prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping"`
  → returns `nullptr`, so that context is never cached at all.
- `tools/server/server-task.cpp:1706`, `"making room for prompt cache entry,
  removing oldest entry (size = %.3f MiB)"`.
- `tools/server/server-task.cpp:1746`, `server_prompt_cache::load()`.

## The central contradiction to resolve

On the LRU path `update_cache` is set to `true`, so `prompt_save()` +
`prompt_load()` should run. Yet in the server log at verbosity 3 there are
**zero** occurrences of both `"failed to load prompt from cache"` and
`"exceeds cache size limit"`, while full reprocesses demonstrably happen.

So one of these must be true, and determining which is the crux of the task:

- `update_cache` is false at that moment (why? `prompt_cache` null? task type not
  `SERVER_TASK_TYPE_COMPLETION`?), or
- `prompt_load()` succeeds but restores only a short prefix, so the reprocess is
  "expected" behaviour given a cache miss on content, or
- the save never happened earlier (so there is nothing to load, and `load()`
  reports success trivially), or
- those SRV_WRN/SLT_WRN lines are not reaching the log at the configured
  verbosity, **verify this first**, since the whole argument rests on their
  absence.

## Leading hypothesis (unconfirmed)

`--cache-ram 16384` is too small for *this* workload. A cached state includes the
full KV plus checkpoints. With f16 KV on this model, measured KV is 5.44 GiB at
262144 tokens, so ~3.9 GiB at 190k, plus up to 32 checkpoints
(`--ctx-checkpoints`, default 32) of recurrent state (~251 MiB each) ≈ 8 GiB.
One large context ≈ 12 GiB fits; **two do not**, and Pi demonstrably juggles
several, captured prompt sizes went 442k → 198k → 4k → 73k chars, i.e. distinct
concurrent contexts.

Note this hypothesis is quantitative, not structural. Confirm it numerically
before acting on it.

## What a good answer contains

1. The actual mechanism, demonstrated from source **and** a reproduction.
2. Whether it is a configuration problem or an upstream bug. If upstream, the
   specific condition and a minimal patch or issue report.
3. A concrete config change with predicted and measured effect, respecting the
   24200 MiB XTX ceiling.
4. If the fix is `--cache-ram`, the right value derived from measured state
   sizes, not a guess. Instrument the actual sizes (raise verbosity to catch the
   trace lines, or add temporary logging).
5. Explicitly state anything you could not verify.

## Useful tooling already in place

- `--log-prompts-dir <path>`, captures every prompt; diff consecutive files.
- `GET /slots`, reports `n_prompt_tokens` and `n_prompt_tokens_cache` live.
- `scratchpad/bench/ctxswitch.py`, A → B → A reproduction harness.
- `scratchpad/bench/cache_watch.sh`, tails the log for `selected slot by LRU`
  (ignoring `t_last = -1` restarts) and auto-diffs the prompts around it.
- `scratchpad/bench/xtx_predict.py`, predicts XTX peak MiB before loading.
- Server is managed by `llama_server/bin/llamactl {start,stop,restart,status}`;
  profiles live in `llama_server/config/profiles/`.
