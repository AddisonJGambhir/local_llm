# Benchmark prompts

Committed so tg numbers are reproducible. This directory exists because the
81.67 t/s adaptive-MTP figure of 2026-08-22 could NOT be reproduced: RESULTS.md
describes its prompt only as "agentic prompt (1958 tok of repeated tool calls)"
and that prompt was never saved. Re-running the identical build, split, context
and sampler with `matrix.SHORT_PROMPT` gave 46.27 t/s (sd 5.43, n=7).

- `agentic_toolcalls.txt` -- ~1,960 tokens of repeated tool_call/tool_response
  pairs. Deterministic (seed 7). This is the HIGH-ACCEPTANCE regime: it is an
  upper bound on speculative tg, not a realistic figure. Label any number from
  it as such.

Any tg record claim must name the prompt file it used.

- `repetitive_sentence.txt` -- one sentence repeated to ~1,958 tokens. This is
  the DEGENERATE ceiling: acceptance approaches 1.0 and tg approaches the
  no-verification limit. Useful only as an upper bound on what speculation can
  ever deliver. Never quote it as a throughput figure.
