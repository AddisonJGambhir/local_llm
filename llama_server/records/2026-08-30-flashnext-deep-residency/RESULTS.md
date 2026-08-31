# Flash-Next populated-context residency correction — 2026-08-30

## The correction

Allocating a 196K or 262K context at server startup is not a populated-context
test. On Vulkan, the server may start with apparently acceptable VRAM and about
2 GiB of system/GTT memory, run quickly at shallow depth, and migrate buffers
only after it has actually read tens of thousands of tokens. Startup GTT by
itself is normal and is not the tipping point.

From now on:

- every reported performance result must have at least 10,000 live tokens;
- a high-context safety or residency claim requires actually reading 100K+
  tokens while recording both VRAM and GTT over time;
- allocated context and live prompt depth must always be reported separately;
- `--spec-type none` is explicit unless speculation is the experiment;
- desktop choppiness is a stop condition, not something to benchmark through.

## Hardware and common configuration

- model body: UD-Q2_K_XL
- KV: q8_0 K and q8_0 V, flash attention on
- backend: patched Vulkan-fusions tree at detached `aa1b06d0c`
- devices: `Vulkan0,Vulkan1`, layer split
- only embeddings on CPU:
  `per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU`
- all experts in VRAM
- load: mmap, lazy tensor reads on
- guard after ready: R9700 32,500 MiB, XTX 24,200 MiB, three strikes

The harness change in `llama_server/bin/flashnext_sweep.sh` makes a prompt file
mandatory, rejects results below 10K tokens by default, forces speculation off,
and records five-second VRAM/GTT telemetry.

## Valid 10K-depth baselines

These runs used exactly 11,547 prompt tokens. They are valid speed probes at
that depth, but are not evidence about residency at 100K+.

| allocated ctx | split | ub | pp | median tg | run |
|---:|---:|---:|---:|---:|---|
| 131,072 | 400,600 | 512 | ~810-816 | 39.4571 | `depth10k-ctx131k-ub512-none` |
| 196,608 | 400,600 | 512 | ~806-810 | 39.3773 | `depth10k-ctx196k-ub512-none` |
| 196,608 | 375,625 | 512 | ~807-810 | 39.2950 | `depth10k-ctx196k-ts375` |

The split change is effectively free at 11.5K depth, but moving one whole
layer from the desktop XTX to the headless R9700 gives the XTX about 1.1 GiB
more headroom.

## Reproduced failure: 262K allocation, split 400

Run: `/tmp/fn-sweep/deep100k-ctx262k-ub512-none`

The test attempted a real ~104K-token prefill. During prefill, the desktop XTX
repeatedly exchanged roughly 0.8-1.0 GiB between VRAM and GTT; one observed
transition was 24,053 MiB VRAM / 2,456 MiB GTT to 23,072 MiB VRAM / 3,265 MiB
GTT ten seconds later. The desktop became severely choppy, so the run was
stopped immediately. No throughput number is reported for this aborted run.

This is the actual Vulkan tipping behavior. It did not appear at initialization
and could not have been discovered by the earlier 8,948-token probes.

## First trustworthy 100K-depth baseline

Run: `/tmp/fn-sweep/deep100k-ctx196k-ts375`

Configuration:

```text
allocated ctx 196,608
tensor split 375,625
ubatch 512, batch 2048
prompt_n 104,769; n_predict 256
q8_0 K/V; --spec-type none
```

Results:

| metric | result |
|---|---:|
| complete prompt evaluation | 104,769 tokens / 312.443 s |
| end-to-end pp | 335.3218 t/s |
| decode at 104,769 depth | 18.0006 t/s |
| final live tokens | 105,024 |

Decode stayed near 17.8-18.1 t/s. The desktop XTX remained below its guard and
the machine stayed usable.

### Exact prefill cliff

| live tokens | elapsed seconds | cumulative pp |
|---:|---:|---:|
| 90,112 | — | 548 |
| 92,160 | 169.02 | 545.25 |
| 94,208 | 199.50 | 472.21 |
| 96,256 | — | 433 |
| 98,304 | — | 406 |
| 100,352 | — | 383 |
| 102,400 | — | 363 |
| 104,253 | — | 344.85 |
| 104,765 | — | 336.18 |

The 2,048-token block from 92,160 to 94,208 took 30.48 seconds, about 67.2
tokens/s, versus roughly 450-500 tokens/s immediately before it. Later blocks
ran about 90-103 tokens/s.

The cliff correlated with migration on the headless R9700: its VRAM fell from
about 32,005 to 31,004 MiB while GTT rose from about 843 to 1,080 MiB. The XTX
stayed around 22.5-22.7 GiB. After prefill and decode, residency moved back to
VRAM. Shutdown also reported compute-buffer totals smaller than expected,
consistent with the backend changing residency during the request.

## What is and is not established

- Split 375 is materially safer for the desktop than split 400 and improves
  the historical ~9-10 t/s around 100-123K to 18.00 t/s at 104.8K.
- It still suffers a sharp migration cliff around 92-94K, now on the R9700.
- The old 49-50 tg and 855 pp figures remain shallow-depth measurements only.
- The claim that 262K all-GPU is a fast, populated-context configuration is
  withdrawn.
- A normal ~2 GiB startup system/GTT footprint is not evidence of spill. The
  evidence is the change while actually populating context.

## Next discriminating experiment

Use the same split 375 and 104,769-token prompt with a 160K allocated context.
That remains well over 128K while freeing KV allocation. If the 92-94K cliff
moves or disappears, allocation headroom is the immediate lever. If it stays
at the same live depth, investigate Vulkan buffer residency and tensor-parallel
patches rather than further split or ubatch guessing.

## 200K populated-context frontier

The next experiments read the entire `/tmp/bigprompt-200k.txt` rather than a
slice. The tokenizer reported `prompt_n = 199,320`; each successful run then
generated 256 tokens. Every result below used split 375,625, Q8_0 K/V, every
expert in VRAM, and explicit `--spec-type none`.

| allocated ctx | ub | result at 199,320 live prompt tokens |
|---:|---:|---|
| 212,992 | 128 | **338.9899 pp / 13.4109 tg**, clean |
| 212,992 | 256 | migration late in prefill; guard killed |
| 200,704 | 256 | migration late in prefill; guard killed |
| 200,704 | 192 | **347.2632 pp / 13.0885 tg**, clean |

Run directories:

```text
/tmp/fn-sweep/deep200k-ctx213k-ts375-ub128-none
/tmp/fn-sweep/deep200k-ctx213k-ts375-ub256-none
/tmp/fn-sweep/deep200k-ctx200704-ts375-ub256-none
/tmp/fn-sweep/deep200k-ctx200704-ts375-ub192-none
```

The ub128 run remained resident throughout. Its telemetry ranged only
31,732-31,738 MiB R9700 VRAM, 22,282-22,533 MiB XTX VRAM, 19-45 MiB R9700
GTT, and 1,753-1,928 MiB XTX GTT. Net GTT growth from ready was 9 MiB.

Both ub256 runs were faster until they failed, but each reproduced the same
R9700 migration signature late in the prompt: roughly 1.0-1.1 GiB VRAM lost
while roughly 0.8-0.9 GiB appeared in GTT. Reducing unused context allocation
by 12,288 tokens freed about 200 MiB on the R9700 and did not prevent the
migration. Therefore ub256 itself is not safe for a populated 200K request.

Ub192 completed cleanly and improved pp by only 2.4%, while tg fell by 2.4%
and safety margin shrank. Ub128 remains the better all-round production
choice; ub192 is a minor prefill-first alternative.

The guard now detects the actual migration pattern after ready: a correlated
512 MiB VRAM loss plus 512 MiB GTT rise, sustained for three five-second
samples. This deliberately ignores an initially high but stable GTT footprint.
Telemetry now includes exact `n_prompt_tokens` at each sample.
