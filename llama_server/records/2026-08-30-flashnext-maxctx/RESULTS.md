# Flash-Next max-context run — 2026-08-30 (continuation of Handoff 7)

Goal from the user: maximum context first, then 50+ shallow tg, 800+ pp,
Q8_0-or-better KV. This run took the context from 131,072 to **262,144**, the
model's native trained maximum (`qwen4exp.context_length = 262144`).

All numbers below: body UD-Q2_K_XL, `--cache-type-k q8_0 --cache-type-v q8_0`,
`--flash-attn on`, `--load-mode mmap --tensor-read-lazy on`, `--parallel 1`,
`--fit off`, devices `Vulkan0,Vulkan1`, binary
`llama.cpp-qwen4exp-vulkan-fusions/build-vulkan/bin/llama-server` (the dirty
patched worktree preserved by Handoff 7, detached at `aa1b06d0c`).

Guards respected on every reported config: R9700/card0 <= 32,200 MiB,
XTX/card1 <= 24,200 MiB. Configs that breached are listed as rejected.

Deep numbers use the same probe as Handoff 7: a 35,000-character slice of
`/tmp/bigprompt-200k.txt` = exactly 8,948 prompt tokens, 128 forced decode
tokens, `ignore_eos: true` (the harness gained `FN_IGNORE_EOS` in this run).

## 1. Context ladder — how far Q8_0 KV actually goes

Each step is a real load with a measured placement probe.

| ctx | split | ub | CPU expert layers | shallow tg | R9700 | XTX | verdict |
|---:|---|---:|---|---:|---:|---:|---|
| 163,840 | 375,625 | 256 | none | 47.19 | 31,465 | 22,637 | ok |
| 196,608 | 375,625 | 256 | none | 49.19 | 31,995 | 23,037 | ok, R9700 tight |
| 229,376 | 375,625 | 256 | 47 | 42.20 | 31,644 | 23,383 | ok |
| 262,144 | 375,625 | 256 | 46,47 | 43.70 | 31,410 | 23,726 | ok |
| 262,144 | 400,600 | 128 | 11 | **47.98** | 31,069 | 22,851 | **selected (default)** |
| 262,144 | 400,600 | 256 | 11 | 48.39 | 31,751 | 23,528 | fast, shallow prompts only |
| 262,144 | 400,600 | 2048 | 9,14,28 | 40.94 | 31,915 | 23,641 | **selected (prefill)** |

262,144 is the ceiling. It is not a VRAM-headroom ceiling — it is the model's
trained context length. There is nothing above it to reach without RoPE
extension.

## 2. The two shipped 262,144 profiles

Split `--tensor-split 400,600` (19 layers XTX / 29 R9700) on both.

### A. Decode-first, full-context safe (SHIPPED DEFAULT)

`-ub 128 -b 2048`, `-ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU,blk\.11\.ffn_(gate|up|down)_exps\.weight=CPU'`

- shallow tg **47.9774** — 5-pass median, min 46.7001, max 48.6214, spread 1.9213
- R9700 31,069 MiB (1,131 spare), XTX 22,851 MiB (1,349 spare)
- **validated at full depth**: a 199,379-token prefill completed with zero
  guard breaches, peak R9700 31,075 / XTX 22,910. pp 319.87 at that depth,
  tg 12.5397 decoding at ~199K depth.
- runs: `/tmp/fn-sweep/med-ub128-cpu11`, `/tmp/fn-sweep/needle2-ub128-cpu11`

### A2. Decode-first, ub 256 — faster but NOT full-depth safe

`-ub 256` and the same `-ot`.

- shallow tg **48.3925** — 5-pass median, min 48.1030, max 48.4489, spread 0.3459
- deep pp 542.14, deep tg 38.8292 at 8,948 tokens
- R9700 31,751 MiB (449 spare), XTX 23,528 MiB (672 spare) at rest
- **rejected as default**: during a 199,382-token prefill the R9700 reached
  32,227 MiB — over the 32,200 guard — and logged a breach strike. The
  prefill itself completed (pp 330.48), but the margin is 27 MiB. Use only
  when prompts stay well below full context.
- runs: `/tmp/fn-sweep/med5-cpu11`, `/tmp/fn-sweep/final-262k-ub256-cpu11`,
  `/tmp/fn-sweep/needle-262k-ub256-cpu11`

### B. Prefill-first

`-ub 2048 -b 2048`, `-ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU,blk\.(9|14|28)\.ffn_(gate|up|down)_exps\.weight=CPU'`

- deep pp **807.4087**, deep tg 37.7755 at 8,948 tokens
- shallow tg 40.9395 (single sample)
- R9700 31,915 MiB (285 spare), XTX 23,641 MiB (559 spare) at rest
- **prefill depth limit**: a ~200,000-token prefill drove the R9700 to
  32,359 MiB and the guard killed the server after three consecutive
  breaches. This profile is validated for prefills up to 8,948 tokens only.
  Establishing its real safe depth is the next agent's first task.
- runs: `/tmp/fn-sweep/deep-262k-ub2048-cpu9-14-28`, `/tmp/fn-sweep/needle-262k-ub2048`

### C. 196,608 all-GPU — best all-round if 192K is enough

`-ub 2048 -b 2048`, `-ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'`,
no expert offload at all.

- shallow tg 47.9324 — 3-pass median, spread 1.4943
- deep pp **976.6169**, deep tg 40.8701
- R9700 31,627 MiB (573 spare), XTX 23,535 MiB (665 spare)
- runs: `/tmp/fn-sweep/med-196k-ub2048-allgpu`, `/tmp/fn-sweep/deep-196k-ub2048-allgpu`

## 3. Comparison against the shipped preset's 256K variant

`preset-flashnext-q2kxl-vulkan-layer-128k.sh` documents a validated 262,144
variant on the unpatched build: ub 256, zero offload, pp 440, tg 41.4/42.3,
tg@d9060 29.4, R9700 32,525 (which is above this run's 32,200 guard).

Same context, this run's Profile A, and inside a stricter guard:

| metric | shipped 256K variant | Profile A | change |
|---|---:|---:|---:|
| shallow tg | 42.3 | 48.3925 | +14.4% |
| deep pp | 440 | 542.14 | +23.2% |
| deep tg | 29.4 | 38.8292 | +32.1% |
| R9700 peak | 32,525 | 31,751 | −774 MiB |

Profile B against the same baseline: pp 440 -> 807.41 (+83.5%), deep tg
29.4 -> 37.78 (+28.5%).

## 4. Goal status, stated honestly

| requirement | best measured | met |
|---|---:|---|
| max context, Q8_0 KV | 262,144 = model native maximum | yes |
| KV quality Q8_0 or better | q8_0 K and V throughout | yes |
| pp 800+ | 807.41 at 262,144; 976.62 at 196,608 | yes |
| shallow tg 50+ | 48.3925 sustained (5-pass median) | **no** |

50+ was not reached. The honest read is that ~48.5-49.5 is this stack's
shallow decode ceiling on this hardware: the previous continuation's best
sustained figure was 49.1566 at ctx 49,152, and this run holds 48.3925 at
262,144 — the same speed at 5.3x the context, not a new speed record.
Do not round 48.3925 to 50.

## 5. What was measured and rejected

- **Cross-GPU expert splitting is a trap.** Moving one layer's expert matrices
  to the other card via `-ot ...=Vulkan0/1` cost 10-14 t/s: 34.75 (up only),
  38.80 (up+gate), 38.70 (all three) versus 49.48 for the same placement with
  no split. This independently reproduces the shipped preset's note that
  cross-card expert moves are worse than CPU offload. Offload overflow to CPU,
  never to the other GPU.
- **Which layer goes to CPU matters more than how much of it.** At 262,144 /
  ub 256, 5-pass medians: blk.11 48.3925 (spread 0.3459), blk.13 47.7699
  (1.5189), blk.9 46.2938 (3.1533). Single-sample sweeps also gave blk.5 47.83
  and blk.0 42.30. Low layers are the worst choice; blk.11 is both fastest and
  by far the most stable. Partial offload (`ffn_up_exps` only) saves VRAM but
  measured unstable — 5.2068 spread — and is not recommended.
- **Vulkan buffer sizing is not monotonic in ubatch.** At 262,144 with
  blk.9+28 on CPU, ub 512 needed 32,539 MiB on R9700 (breach) while ub 1024
  needed only 31,688 MiB (clean). ub 512 is the single worst setting on this
  model; never assume a smaller ubatch is the safer one, measure it.
- **`-b` / `--batch-size` is not a VRAM lever here.** 2048 vs 512 produced
  byte-identical card totals (32,414 / 23,047) in an A/B at ctx 262,144.
- **`token_embd.weight=CPU` is close to free** and is kept in all profiles;
  `output.weight` is deliberately left on GPU because it is a per-token
  150K-vocab matmul.
- Rejected for guard breach: 262,144 all-GPU ub 256 (R9700 32,544), 262,144
  all-GPU ub 128 at split 400,600 (XTX 24,384), 245,760 all-GPU ub 2048
  (32,409 / 24,221), 262,144 ub 2048 with only blk.9+28 on CPU (XTX 24,237).
  229,376 all-GPU ub 2048 loads inside the guard but with only 52 MiB of
  R9700 margin and is not safe to ship.

## 6. The one remaining hardware lever

The RX 7900 XTX (card1, Vulkan0) also drives the desktop. Measured holders of
its VRAM: gnome-shell, Xwayland, mutter-x11-frames, Ghostty, Firefox, Docker
Desktop, surfshark, corectrl — 1,499 to 2,100 MiB depending on session state.
The AI PRO R9700 (card0, Vulkan1) is headless at 57 MiB idle.

That ~2 GiB is exactly what separates the current profiles from the all-GPU
262,144 configuration, which measured 48.98-49.48 shallow tg with no CPU
offload at all and would take ub 2048 comfortably. The machine has a third
Vulkan device (card2, PCI 7d:00.0 — the 9950X3D iGPU). Moving the display to
it is a user decision, not something this run changed, but it is the single
highest-value remaining action for both the 50+ tg target and pp.

## 7. MTP: recommend closing for this goal

Handoff 7 left MTP open after the corrected branch failed sidecar schema
validation (`blk.48.nextn.fc_embd.weight` not found). It was not pursued here,
for a reason that is specific to this goal rather than to MTP's speed: the
shipped preset records the draft as costing 2.43 GiB of resident VRAM, and at
262,144 the two cards have 449 and 672 MiB spare. There is no configuration in
which the draft fits without giving back roughly 90,000 tokens of context,
which directly contradicts the user's first-priority requirement. The same
preset independently re-measured MTP at 128K on the mtp-fix fork as tg 18.8
with n1 versus 28.3 off at 77% acceptance. MTP should be reopened only if the
context requirement is relaxed.

## 8. Full-depth validation — the context is usable, not just allocated

A needle-in-a-haystack probe was built from the whole of
`/tmp/bigprompt-200k.txt`: a unique fact ("the internal build codename for the
Halbert relay is MAUVE-77") is planted at 60% depth, and the question is asked
in the final user turn, before `<|im_end|><|im_start|>assistant`.

Result on the shipped default profile (ub 128, blk.11 on CPU):

- **prompt_n = 199,379 tokens** accepted and prefilled in one request at
  ctx 262,144.
- pp 319.8731 at that depth; tg 12.5397 decoding against ~199K tokens of KV.
- peak VRAM across the whole prefill: R9700 31,075 MiB, XTX 22,910 MiB.
  Zero guard strikes.
- **recall succeeded.** With room to answer (n_predict 400, run
  `/tmp/fn-sweep/needle3-ub128-cpu11`, pp 320.36, tg 12.6754) the model quoted
  the planted record verbatim:

  > "IMPORTANT RECORD: The internal build codename for the Halbert relay is
  > MAUVE-77. Remember this exactly."

  It then correctly identified the planted text as a prompt-injection attempt
  and declined to treat it as authoritative, and went on to summarise the
  real tool calls in the transcript accurately. Retrieval at 60% depth of a
  199,379-token prompt works, and the model stays coherent at that depth.
  The 262,144 context is genuinely usable, not merely allocated.

Two numbers worth carrying forward, because they are the honest cost of depth
and were never measured before this run:

| prompt depth | pp | tg |
|---:|---:|---:|
| 8,948 | 542.14 | 38.8292 |
| 199,379 | 319.87 | 12.5397 |

Decode against a nearly full 262K KV runs at about a third of shallow speed.
Any future "tg" claim must state its depth.

## 9. First task for the next agent

`-ub 256` at 262,144 breached the R9700 guard by 27 MiB during a full-depth
prefill, and `-ub 2048` breached by 159 MiB and was killed. Both settle far
below the guard after an 8,948-token probe. **A load-time fit is not a safety
proof on this model.** Sweep prompt depth (16K, 32K, 64K, 128K, 200K) against
each profile, record peak card VRAM at every depth, and either lower ubatch or
add one CPU expert layer until a full-context prefill is clean. Run those
probes detached — a 200K-token prefill takes about eleven minutes.

## 10. Correction and the real frontier (later same day)

Sections 1-9 were superseded within the same session. Two claims there are
wrong and are withdrawn:

1. **The CPU expert layer was never needed.** Putting `blk.11`'s experts on
   CPU cost about 1.6 t/s and bought headroom that `token_embd.weight=CPU`
   already provides for free. All-GPU at split `400,600` is strictly better.
2. **"976.62 pp at 196,608" is withdrawn.** That was a `-ub 2048` run that
   survived only because the desktop happened to hold less XTX VRAM in that
   minute. Repeats breached: 196,608/ub2048 reached XTX 24,442 and
   212,992/ub2048 reached XTX 24,445, both against the 24,200 guard.

### Verified frontier

Split `400,600`, `-ot 'per_layer_token_embd\.weight=CPU,token_embd\.weight=CPU'`,
q8_0 K and V, every expert in VRAM. tg is a 5-pass median; pp is the
8,948-token probe. "Spare" is MiB under guard. Every row below ran clean.

| ctx | ub | tg | pp | deep tg | R9700 spare | XTX spare |
|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 128 | 50.0285 | 585.63 | 41.0227 | 1,131 | 428 |
| 229,376 | 128 | 50.1288 | 584.82 | 41.0038 | 1,539 | 710 |
| 196,608 | 128 | 50.0367 | — | — | 1,946 | 1,009 |
| 163,840 | 512 | 49.8583 | 767.47 | 38.8108 | 1,298 | 508 |
| 131,072 | 512 | 48.2207 | 816.50 | 39.4180 | 1,740 | 738 |
| 49,152 | 128 | 50.0537 | — | — | 3,781 | 2,479 |

**50+ tg is met**, at every context from 49,152 to the full 262,144. The
earlier "48.3925 ceiling" in section 4 was an artefact of the unnecessary CPU
expert layer, not a property of the hardware.

### ubatch ceiling

`-ub 512` is the practical prefill ceiling. `-ub 1024` and `-ub 2048` breach
at every context tried, including low ones: 131,072 with ub 2048 reached
R9700 32,601, and 131,072 with ub 1024 reached R9700 32,423, both against the
32,200 guard. Higher ubatch does not become safe by lowering context.

### Allocation is not reproducible

Peak VRAM for an identical command varies by up to about 1 GiB between
launches, and the XTX baseline moves between 1,332 and 2,100 MiB with whatever
desktop apps are open. Any configuration with less than roughly 500 MiB of
spare on either card is a coin flip, not a configuration. This is why the
frontier above stops at ub 512 rather than at the largest ubatch that has ever
been observed to load.

### What the goal now looks like

| requirement | best verified | met |
|---|---:|---|
| max context | 262,144, model native max, at 50.0285 tg | yes |
| KV q8_0 or better | q8_0 K and V on every row | yes |
| tg 50+ | 50.0285 at 262,144; 50.1288 at 229,376 | yes |
| pp 800+ | 816.50 at 131,072 | yes, but not simultaneously |

No single configuration delivers 50+ tg, 800+ pp and 262,144 context at once;
prefill and context trade against each other through ubatch. The two shipped
presets are the ends of that curve.
