> **SUPERSEDED by `2026-08-31-FLASHNEXT-HANDOFF-9.md`.** The successor contains
> actual 199K-populated measurements, the corrected preset, current online PR
> research, and the live same-prompt A/B. Keep this file as historical context.
>
> **SUPERSEDED AGAIN: the final profile was not validated with populated
> context.** A 262,144-token allocation plus an 8,948-token prompt does not
> reveal Vulkan's later VRAM-to-GTT migration. A real 104,769-token read found
> a sharp residency cliff at 92-94K and only 18.00 tg afterward. Read
> `llama_server/records/2026-08-30-flashnext-deep-residency/RESULTS.md` first.
>
> **NUMBERS IN THIS FILE ARE STALE.** They were measured before three method
> errors were corrected (peak-during-load is not a valid VRAM reading; the
> headless R9700 guard was too tight at 32,200; CPU expert offload hurts
> prefill as well as decode). The authoritative result is
> `llama_server/records/2026-08-30-flashnext-maxctx/RESULTS.md` **section 11**,
> and the shipped config is
> `llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-fusions-262k.sh`.
>
> Final: ctx **262,144**, ts 400,600, `-ub 512`, q8_0 K/V, every expert in
> VRAM -> tg **49.5362** (9-pass median), pp **855.2231**, deep tg 41.6166,
> R9700 32,419/32,624 and XTX 23,521/24,560 while generating. A `-ub 128`
> variant gives tg **50.1408** / pp 585.63. Sections 1-7 below are kept only
> for the hardware, guard and worktree rules.

# Handoff 8: 262,144 context reached and shipped; 50+ tg not reached

Supersedes `2026-08-30-FLASHNEXT-HANDOFF-7.md`. Handoff 7's hardware notes,
guard rules and worktree-preservation rules all still apply unchanged — reread
its sections 1 and 3 before touching anything.

Authoritative measurements for this continuation:
`llama_server/records/2026-08-30-flashnext-maxctx/RESULTS.md`.

## 1. Headline

Context went from 131,072 to **262,144** — the model's trained maximum
(`qwen4exp.context_length = 262144`). There is nothing above it to reach
without RoPE extension, so the user's first-priority requirement is now
capped by the model, not by this machine.

At that context, inside a stricter R9700 guard than the shipped preset used:

| metric | shipped 256K variant | now | change |
|---|---:|---:|---:|
| shallow tg | 42.3 | 48.3925 | +14.4% |
| deep pp (8,948 tok) | 440 | 542.14 | +23.2% |
| deep tg (8,948 tok) | 29.4 | 38.8292 | +32.1% |
| R9700 peak | 32,525 | 31,751 | −774 MiB |

A prefill-first variant reaches **807.41 pp** at the same 262,144 context, and
a 196,608 all-GPU config reaches **976.62 pp** with 47.9324 tg.

## 2. Goal status — read this before claiming success

| requirement | best measured | met |
|---|---:|---|
| max context | 262,144, model native max | yes |
| KV Q8_0 or better | q8_0 K and V everywhere | yes |
| pp 800+ | 807.41 at 262K; 976.62 at 196K | yes |
| shallow tg 50+ | **48.3925** 5-pass median (ub 256); 47.9774 shipped | **no** |

Do not round 48.3925 to 50 and do not quote the 49.48 or 49.55 single samples
as sustained. Handoff 7's best sustained figure was 49.1566 at ctx 49,152;
this run holds 48.3925 at 262,144. That is the same decode speed at 5.3x the
context — a capacity win, not a speed win. ~48.5-49.5 looks like this stack's
shallow decode ceiling on this hardware.

## 3. What ships

Both profiles are documented in
`llama_server/config/profiles/preset-flashnext-q2kxl-vulkan-fusions-262k.sh`.
Both use split `400,600`, q8_0 K/V, and the patched worktree binary.

- **Decode-first (preset default):** `-ub 128`, one expert layer (`blk.11`)
  on CPU. 47.9774 tg 5-pass median, R9700 31,069 / XTX 22,851.
  **The only 262,144 profile proven at full depth**: a 199,379-token prefill
  ran clean with zero guard strikes, peak 31,075 / 22,910, pp 319.87 and
  tg 12.5397 at that depth.
- **Decode-first, ub 256 (shallow prompts only):** 48.3925 tg / 542 pp /
  38.83 deep tg at 8,948 tokens, but a 199,382-token prefill reached
  32,227 MiB on the R9700 — over guard.
- **Prefill-first (documented variant):** `-ub 2048`, `blk.9,14,28` experts on
  CPU. 807.41 pp / 37.78 deep tg / 40.94 shallow tg.
- **196,608 all-GPU (documented variant):** `-ub 2048`, no expert offload.
  47.9324 tg / 976.62 pp / 40.87 deep tg, with 573 and 665 MiB spare.

## 4. Rules learned this run — do not relearn them the hard way

1. **Never move experts across cards.** `-ot ...=Vulkan0/1` on one layer's
   expert matrices cost 10-14 t/s (34.75 / 38.80 / 38.70 versus 49.48 for the
   same placement unsplit). Overflow goes to CPU or nowhere. This reproduces
   the shipped preset's own note independently.
2. **Which layer goes to CPU matters more than how much of it.** 5-pass
   medians at 262,144/ub256: blk.11 48.3925 (spread 0.3459), blk.13 47.7699,
   blk.9 46.2938 (spread 3.1533), blk.0 42.30. Low layers are worst. Partial
   offload (`ffn_up_exps` only) is unstable — 5.2068 spread.
3. **Vulkan buffer sizing is not monotonic in ubatch.** ub 512 needed 32,539
   MiB on R9700 and breached; ub 1024 needed 31,688 and did not. Never assume
   a smaller ubatch is safer — measure it.
4. **`--batch-size` is not a VRAM lever.** 2048 vs 512 gave byte-identical
   card totals.
5. **A load-time fit is not a safety proof.** See section 5.

## 5. Open issue, and the exact next task

**RESOLVED for the default profile; still open for the others.**
Deep prefill grows VRAM past the load-time figure. The prefill-first
profile (ub 2048) sits at 31,915 MiB on the R9700 after an 8,948-token probe,
but a ~200,000-token prefill drove it to 32,359 MiB and the guard killed it
after three consecutive breaches:

```text
/tmp/fn-sweep/needle-262k-ub2048/guard.log
13:31:10 KILLING pid=2425999 sustained breach r9700=32359MiB xtx=23803MiB
```

So every profile in section 3 is validated for prefills up to 8,948 tokens
only. The next agent's first task is to establish the real safe prefill depth
for each, by sweeping prompt length (16K, 32K, 64K, 128K, 200K) and recording
peak card VRAM at each depth, then either lowering ubatch or adding one more
CPU expert layer until a full-context prefill is clean. A ~200K-token prefill
takes well over ten minutes, so run it detached, not in a foreground shell
with a timeout.

This was resolved for the shipped default by dropping to `-ub 128`: the
prefill-time KQ mask is ctx x ubatch x 4 bytes per attention-hosting card, so
it is 128 MiB at ub 128, 256 MiB at ub 256 and about 2 GiB at ub 2048. At
ub 128 a 199,379-token prefill peaked at 31,075 / 22,910 with zero strikes.
`-ub 256` reached 32,227 (27 MiB over guard) and `-ub 2048` reached 32,359 and
was killed. Both still need a depth sweep before they can be recommended for
anything but short prompts.

The same probe also proves the context is *usable* and not merely allocated:
a unique fact is planted at 60% depth and the model is asked to recall it in
the final user turn. Details and the depth-vs-speed table are in `RESULTS.md`
section 8. The headline cost of depth: pp 542 -> 320 and tg 38.83 -> 12.54
going from 8,948 to 199,379 prompt tokens. Every future tg claim must state
its depth.

## 6. The one remaining hardware lever

The RX 7900 XTX (card1 / Vulkan0) also drives the desktop and holds
1,499-2,100 MiB before llama-server starts — measured holders: gnome-shell,
Xwayland, mutter-x11-frames, Ghostty, Firefox, Docker Desktop, surfshark,
corectrl. The AI PRO R9700 (card0 / Vulkan1) is headless at 57 MiB idle.

That ~2 GiB is exactly the gap between the shipped profiles and the all-GPU
262,144 configuration, which measured 48.98-49.48 shallow tg with zero CPU
offload and would take ub 2048 comfortably. The host has a third Vulkan
device (card2, PCI 7d:00.0 — the 9950X3D iGPU). Driving the display from it is
a user decision; it is the single highest-value remaining change for both the
50+ tg target and pp, and it is more valuable than any further software
tuning found in this run.

## 7. MTP — recommend closing for this goal

Not pursued here, for a reason specific to this goal rather than to MTP's
speed. The shipped 128k preset records the draft as costing 2.43 GiB resident.
At 262,144 the two cards have 449 and 672 MiB spare. There is no configuration
where the draft fits without giving back roughly 90,000 tokens of context,
which contradicts the first-priority requirement. That same preset also
independently re-measured MTP at 128K on the mtp-fix fork as tg 18.8 with n1
versus 28.3 off at 77% acceptance. Handoff 7's sidecar schema failure
(`blk.48.nextn.fc_embd.weight` not found) is still unresolved and is still the
correct first step *if* the context requirement is ever relaxed.

## 8. Harness change

`llama_server/bin/flashnext_sweep.sh` gained `FN_IGNORE_EOS` (default `false`).
Set it to `true` for forced-length decode probes; without it a deep probe can
return `tg=0.0` because the model emits EOS immediately.
