# hipfire on this host

Set up 2026-08-16. hipfire 0.3.0 (`8510ca5f247d`, master).

This is a **separate inference engine**, not a llama.cpp backend. Different
weight format (HFQM: `.mq4` / `.mq4r` / `.hf4` / `.mq6`), different daemon
protocol, different port. It gets its own tooling here rather than being bolted
into `llama_server/`.

## Quick start

```bash
cd ~/Desktop/local_llm/hipfire
bin/hipfirectl doctor      # verify env + kernel cache
bin/hipfirectl start       # daemon on 127.0.0.1:11435
bin/hipfirectl status
bin/hipfirectl bench       # benchmark the active profile's model
bin/hipfirectl stop
```

**Never call the bare `hipfire` binary.** It fails on this host without the
environment in `config/env.sh`. Use `bin/hipfirectl exec <subcommand>` for
anything not wrapped.

## Install gotchas, all four are load-bearing

The install took five attempts. Every failure was an environment mismatch, not
a hardware problem. If you ever reinstall or upgrade, you will hit these again.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `multiple GPU architectures found: gfx1100, gfx1201, gfx1036` | hipfire compiles kernels for **one** arch | `--gpu-arch gfx1201` |
| 2 | `Installation cancelled.` | interactive `Continue? [Y/n]` with no stdin | `--yes` |
| 3 | **45/45 kernels fail**: `cannot find ROCm device library` | hipcc gets `--rocm-path=/usr`, but the amdgcn bitcode is under the LLVM-21 resource dir | `HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode` |
| 4 | kernels compile, then `hipModuleLoad: device kernel image is invalid` (HIP 200) | compiled for gfx1201, validated against HIP device 0 = the **XTX** (gfx1100) | `HIP_VISIBLE_DEVICES=1 ROCR_VISIBLE_DEVICES=1` |

Number 3 is the one that will catch you again: **an LLVM major-version upgrade
moves that bitcode path**. Check with

```bash
ls /usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode/ocml.bc
```

llama.cpp is immune because CMake's first-class HIP language support resolves
the bitcode itself.

Number 3 also applies at **runtime**, not just install, `hipfire profile`
recompiles kernels and fails without the flag.

## Hardware mapping

| HIP idx | Card | Arch | VRAM | Notes |
|---|---|---|---|---|
| 0 | RX 7900 XTX | gfx1100 | 24560 MiB | **drives the desktop**, ~24200 MiB hard ceiling |
| 1 | AI PRO R9700 | gfx1201 | 32624 MiB | what we target |
| 2 | iGPU (Raphael) | gfx1036 | N/A | ignore |

We target the R9700: it has room for a 17 GiB model and does not drive the
display. Retargeting the XTX means setting `HIPFIRE_GPU=0` **and**
`HIPFIRE_TARGET_ARCH=gfx1100` **and** re-running `hipfirectl setup`, the
kernel cache is per-arch.

`HIPFIRE_ALLOW_MIXED_ARCH` exists in the source but is untested here.

## Multi-GPU: don't expect anything

Read before wiring the second card in. From hipfire's own `docs/multi-gpu.md`:

- **PP (pipeline parallel)** is wired only for Qwen3.5/3.6 HFQ (`arch_id` 5
  dense, 6 MoE/A3B) via `load_qwen35_pp`. **Qwen3.8 is not on that list.**
- **EP (`--tp`)** is wired only for MiniMax-M2 (arch 10) and DeepSeek V4 Flash
  (arch 9). None of our models.
- `admissions.yml` has **no multi-GPU records** at schema v2; the only earned
  row is single-GPU `pp=tp=1`.
- Upstream, verbatim: *"PP is not tensor-parallel serving… **No speedup is
  promised** for models that already fit on one card; current historical
  measurements on one 2× gfx1100 station were slower under sequential PP=2."*
- `params.pp` is not even a first-class CLI flag, it needs a raw daemon JSONL
  `load` message.

So hipfire here is a **single-GPU engine**. That matches where its performance
claim comes from anyway.

## Quantization: there is no 8-bit path

This is the constraint that decides whether hipfire is usable for real work.

Registry entries for our model:

| tag | file | size | VRAM |
|---|---|---|---|
| `qwen3.8:27b` | `.mq4` | 15.66 GB | 17 GB |
| `qwen3.8:27b-fast` | `.mq4r` | 14.98 GB | 16 GB |

Both 4-bit. Converting our own GGUFs is narrower still: `hipfire quantize` with
**GGUF input accepts only `hf4`, `hf6`, `mq4`, `mq6`**. `q8`/`q8f16` are
safetensors-input only and documented as *"reference / debug"*.

The llama.cpp side of this box runs Qwen3.8-27B at **Q8_0 with f16 KV**. So any
hipfire-vs-llama.cpp comparison is **not like-for-like**, it is 4-bit on one
card against 8-bit on two. State that whenever quoting a number.

Registry notes both SKUs are *"Validated on gfx1100/gfx1201"*.

## Coexistence with llama-server

Ports do not clash (llama.cpp 1234, hipfire 11435) but **VRAM does**. The R9700
has 32624 MiB and hipfire wants ~16-17 GiB of it; a loaded llama-server
routinely holds 28-30 GiB there.

`hipfirectl start` refuses to launch below 18000 MiB free and tells you to stop
llama-server first. That is deliberate, an OOM partway through a load is worse
than a refusal.

## Gotcha inherited from the llama.cpp side

`llamactl command` has been observed reporting a **completely different config
from what is actually running** (it resolves a mutable `interactive-profile.sh`).
Ground truth is `/proc/<pid>/cmdline`. `hipfirectl status` prints the daemon's
real cmdline for exactly this reason, do not add a cached-config shortcut.

## Not yet measured

Nothing. The model download was still in flight when this was written. No
hipfire throughput number exists for this host yet.

The comparison to run, both on the R9700 alone, back to back:

| engine | config | expected |
|---|---|---|
| hipfire | `qwen3.8:27b-fast` MQ4R | unknown |
| llama.cpp | Qwen3.8-27B **Q8_0**, `-sm none` | 698 pp / 19.24 tg (measured 2026-08-16) |

4-bit should beat 8-bit on bandwidth alone, so hipfire winning is not by itself
evidence its kernels are better. The honest bar: if MQ4R does **not** clearly
beat llama.cpp's Q8_0 on the same card, the engine is buying nothing and the
quality drop is unjustified.
