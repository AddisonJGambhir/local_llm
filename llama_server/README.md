# Local LLM Workstation (`llama_server/`)

Local llama.cpp serving, benchmarking, and diagnostics across **five models** on a
dual-GPU AMD workstation:

| GPU | Model | VRAM | Role |
| --- | --- | --- | --- |
| RX 7900 XTX (gfx1100, 24 GB) | N/A | 24 GB | Drives the desktop |
| AI PRO R9700 (gfx1201, 32 GB) | N/A | 32 GB | Compute partner |

## Models in use

| Model | Size | Arch | Specs | Dec t/s | Pre t/s | GPU |
| --- | ---: | --- | --- | ---: | ---: | --- |
| **Qwen3.6 35B-A3B** Q8_0 | 36 GB | MoE (3B active) | MTP n_max=2, f16 KV, dual-GPU | 93.7 vk / **94.4** rocm | **2072** / **3128** | dual |
| **Nemotron 3.5 Lightning** 30B-A3B Q8_0 | 33 GB | Mamba-2/attention MoE | spec off (upstream MTP hang), f16 KV | 94.6 (256k) / 94.7 (1M) | 3034 (256k) / 2555 (1M) | dual (ROCm) |
| **Qwen3.8 27B** Q8_0 | 28 GB | Dense + vision | MTP n_max=3, f16 KV, dual-GPU | 35.0 vk / **38.5** rocm | 825 / **982** | dual |
| **Muse Glimmer 30B** Q8_0 | 28 GB | Dense | DFlash n_max=2, f16 KV, dual-GPU | 31.7 vk / **33.2** rocm | 743 / **986** | dual |
| Qwen3.6 27B-MTP Q4_K_M | 16 GB | Dense | MTP, q8_0 KV, single-GPU | ~73–76 (May harness) | N/A | single |

**Profile selection is workload-driven (2026-08-16 numbers):**

- Long generation (chat, agentic, reasoning): **35B MoE ROCm dual** (94.4 t/s, 3128 prefill).
- 1M context: **Nemotron 3.5** at 2555 prefill, 94.7 decode (speculation off; upstream MTP hang).
- Vision + coding: **35B MoE ROCm dual with `--mmproj`** (vision tower pinned to the R9700).
- Text-only, maximum prefill: **Qwen3.8 27B ROCm `ppopt`** (q8_0 KV, ts 1,1).
- Dense quality, single GPU: **27B MTP Q4_K_M** 128k presets.

## Quick start

These are shell functions/PATH entries added by `~/.bashrc` (open a new terminal
or `source ~/.bashrc` after changes):

```bash
llamaserver                  # interactive launcher: pick backend/model/KV/spec/context
llamakill                    # stop the server (alias for: llamactl stop)
harness                      # pick a coding/agent front-end pointed at the local server

llamactl status              # what's running: resolved command + GPU residency + health
llamactl logs                # follow the rotating server log
llamactl command             # print the resolved server command WITHOUT launching
```

The server listens on `http://127.0.0.1:1234/v1` (OpenAI-compatible), model alias `local`.

> **Always free the GPU before benchmarking** (`llamakill`), each profile
> may load onto either a single GPU or both.

---

## Architecture

```text
llamactl  ──sources──►  src/server/common.sh        (load + validate profile, build LLAMA_CMD)
   │                         ▲
   │                         │ reads vars from
   │                    config/profiles/*.sh          (LLAMA_* settings)
   │
   └──launches──►  src/server/supervisor.py           (run server, rotate log, forward signals)
                          │
                          └──spawns──►  ../llama.cpp*/build*/bin/llama-server
```

- **`llamactl`**, the real CLI. Resolves a profile, builds the exact
  `llama-server` command, starts it under the supervisor, polls `/health`, then
  prints the resolved command and a GPU-memory/spill report. Also does
  stop/restart/status/health/logs/command.
- **`src/server/common.sh`**, `llama_load_profile` (defaults + validation) and
  `llama_build_command` (turns `LLAMA_*` vars into the argv).
- **`src/server/supervisor.py`**, owns the child process and the rotating log
  (`llama-server.log`, 50 MB × 5). PID files let `llamactl` manage only *its own*
  server, never an unrelated process on the port.
- **`config/profiles/default.sh`**, the canonical production profile (35B MoE, Vulkan, single-GPU).
- **`config/profiles/preset-36-35b-a3b-*-dual-*.sh`**, 35B MoE dual-GPU profiles (ROCm and Vulkan).
- **`config/profiles/preset-38-27b-*-dual-*.sh`**, Qwen3.8 27B dual-GPU profiles.
- **`src/server/sync_clients.py`**, best-effort push of alias/context into client
  configs after a healthy start.

### Front-ends (thin wrappers)

| Command | File | What it does |
| `llamaserver` | `llama_server.sh` → `bin/llama-launch` | interactive decision-tree launcher; writes a generated profile then `llamactl restart` |
| `llamakill` | (bashrc) | `llamactl stop` |
| `llamactl` | `llamactl` → `bin/llamactl` | the supervisor CLI |

---

## Commands

### `llamactl <action> [--profile PATH]`

| Action | Description |
| `start` | Start the profile, wait for health, print resolved command + GPU report |
| `stop` | Stop **only** the server this setup owns (PID-verified) |
| `restart` | `stop` then `start` |
| `status` | Profile, endpoint, PIDs, **resolved command of the running process**, GPU residency, health |
| `health` | Query `/health` |
| `logs` | `tail -F` the rotating server log |
| `command` / `dry-run` | Validate the profile and print the exact command **without launching** |

Every `start`/`restart` prints, before launching:

```text
── final configuration (resolved server command) ──
<the full llama-server argv, %q-quoted>
```

and after health:

```text
gpu memory [whole card]: VRAM 20.10/23.98 GiB (3.88 free) · GTT 0.20 GiB
  server process: 18.86 GiB resident in VRAM, 0.14 GiB in SYSTEM RAM (GTT)
```

> The `status` **`profile:`** line reads the (mutable) profile file; the
> **`command:`** line is read from `/proc/<pid>` and is the ground truth of what
> is actually running. Trust `command:`.

### `llamaserver`, interactive launcher

A top-level launcher with three paths:

```text
1. ROCm           custom decision tree
2. Vulkan         custom decision tree
3. Presets        advised Vulkan + MTP configurations (single-GPU)
```

#### Preset summary (single-GPU, Vulkan / ROCm)

| Profile | Model | Backend | KV | Ctx | Spec | n_max | Dec t/s |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `default.sh` | 35B MoE Q8_0 | Vulkan | q8_0 | 256k | MTP | 2 | ~94¹ |
| `preset-27b-q4km-q8-128k` | 27B MTP Q4_K_M | Vulkan | q8_0 | 128k | MTP | 4 | ~76¹ |
| `preset-27b-q4km-rocm-q8-128k` | 27B MTP Q4_K_M | ROCm | q8_0 | 128k | MTP | 2 | ~49¹ |
| `preset-36-27b-q4km-vulkan-xtx-only-128k` | 27B MTP Q4_K_M | Vulkan | q8_0 | 128k | MTP | 2 | N/A |

¹ 2026-05 harness (pre-f16-KV era); re-measure before trusting. `default.sh` is the
single-GPU fallback (ubatch 1024, q8_0 KV); the xtx-only preset is for when the
R9700 is needed elsewhere (batch 2048).

#### Dual-GPU presets (ROCm + Vulkan)

All dual profiles use `HIP_VISIBLE_DEVICES=0,1`, `LLAMA_CACHE_RAM=24576`,
`LLAMA_TIMEOUT=3600`, `MTMD_BACKEND_DEVICE=ROCm1` (or `Vulkan1`) for vision towers.
Large batch `6144–12288` (ROCm only; no-op on Vulkan).

| Profile | Model | Backend | KV | Ctx | Spec | n_max | Pre t/s | Dec t/s | Acc | XTX/R9700 GiB |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `preset-36-35b-a3b-q8-rocm-dual-256k-textonly` | 35B MoE Q8_0 | ROCm | f16 | 256k | MTP | 2 | **3128** | **94.4** | 80.9% | 21.0 / 30.2 |
| `preset-36-35b-a3b-q8-rocm-dual-q8-256k-mmproj` | 35B MoE Q8_0 + vision | ROCm | f16 | 256k | MTP | 2 | 3128 | 94.4 | 80.9% | 21.0 / 30.2 |
| `preset-36-35b-a3b-q8-vulkan-dual-256k-textonly` | 35B MoE Q8_0 | Vulkan | f16 | 256k | MTP | 2 | 2072 | 93.7 | N/A |, |
| `preset-36-35b-a3b-q8-vulkan-dual-q8-256k-mmproj` | 35B MoE Q8_0 + vision | Vulkan | f16 | 256k | MTP | 2 | 2072 | 93.7 | N/A |, |
| `preset-36-27b-q4km-rocm-dual-256k` | 27B MTP Q4_K_M | ROCm | f16 | 256k | MTP | 2 | 1408 | 36.9 | 69.8% | 19.7 / 20.9 |
| `preset-36-27b-q4km-vulkan-dual-256k` | 27B MTP Q4_K_M | Vulkan | f16 | 256k | MTP | 2 | N/A |, | N/A |, |
| `preset-38-27b-q8-rocm-dual-q8-256k-mmproj` | Qwen3.8 27B + vision | ROCm | f16 | 256k | MTP | 3 | 982 | 38.5 | 42–72% | 22.5 / 29.7 |
| `preset-38-27b-q8-rocm-dual-256k-ppopt` | Qwen3.8 27B + vision | ROCm | q8_0 | 256k | MTP | 3 | N/A |, | N/A |, |
| `preset-38-27b-q8-vulkan-dual-q8-256k-mmproj` | Qwen3.8 27B + vision | Vulkan | f16 | 256k | MTP | 3 | 825 | 35.0 | 53.8% | 22.2 / 29.4 |
| `preset-muse-glimmer-30b-q8-rocm-dual-128k-dflash` | Muse Glimmer 30B | ROCm | f16 | 128k | DFlash | 2 | 986 | 33.2 | 58.5% | 18.2 / 16.5 |
| `preset-muse-glimmer-30b-q8-rocm-dual-256k-yarn` | Muse Glimmer 30B | ROCm | f16 | 256k | DFlash | 2 | 891 | 34.1 | 3/3 @149k | 19.9 / 18.3 |
| `preset-muse-glimmer-30b-q8-vulkan-dual-128k-dflash` | Muse Glimmer 30B | Vulkan | f16 | 128k | DFlash | 2 | 743 | 31.7 | 43.2% | 18.2 / 16.5 |
| `preset-nemotron35-30b-a3b-q8-rocm-dual-256k` | Nemotron 3.5 30B | ROCm | f16 | 256k | off² | N/A | 3034 | 94.6 | n/a | 19.9 / 18.6 |
| `preset-nemotron35-30b-a3b-q8-rocm-dual-1m` | Nemotron 3.5 30B | ROCm | f16 | **1M** | off² | N/A | 2555 | 94.7 | n/a | 19.9 / 29.4 |

² Nemotron's built-in MTP hangs ~37% of generations (upstream bug). Speculation
ships **off**; decode still hits 94+ t/s. The `mmproj` profiles pin the vision
tower to the R9700 (`MTMD_BACKEND_DEVICE`); text stats = text-only sibling.
`ppopt` is the pre-f16 q8_0 text-only variant (ts 1,1); kept for max prefill.

### Model menu (llamaserver)

The launcher offers **4 models**:

| Key | Model | Arch | MTP |
| --- | --- | --- | --- |
| 1 | Qwen 3.6 35B-A3B MoE MTP, Q8_0 | MoE (A3B) | yes |
| 2 | Qwen 3.8 27B MTP, Q8_0 (vision) | dense | yes |
| 3 | Muse Glimmer 30B, Q8_0 (DFlash sidecar) | dense | no |
| 4 | Qwen 3.6 27B MTP, Q4_K_M | dense | yes |

It writes `~/.local/state/local-llm/interactive-profile.sh`, sourcing either the
selected preset or `default.sh` plus the custom overrides, then runs
`llamactl restart`. Custom profiles are named
`interactive-<arch>-<backend>-<kv>-<spec>-<ctx>k` (MTP tagged `mtp-n<N>`).
`llamaserver --dry-run` prints the resolved command to a throwaway profile and
never clobbers the live one.

---

## Profiles

Profiles are sourced shell files of `LLAMA_*` variables. Key ones (see
`config/profiles/default.sh` and `src/server/common.sh` for the full set + defaults):

| Variable | Meaning |
| `LLAMA_BINARY` / `LLAMA_MODEL` | server binary + GGUF path |
| `LLAMA_BACKEND` / `LLAMA_DEVICE` | `vulkan` (`Vulkan0`) or `rocm` |
| `LLAMA_CONTEXT` | context length in tokens |
| `LLAMA_CACHE_TYPE_K` / `_V` | KV cache type: `q8_0`, `f16`, `q5_1`, `q5_0`, `q4_0` |
| `LLAMA_SPEC_MODE` | `none` \| `mtp` \| `dflash` |
| `LLAMA_SPEC_DRAFT_N_MAX` | MTP/DFlash draft-depth ceiling |
| `LLAMA_SPEC_DRAFT_TYPE_K` / `_V` | draft KV type |
| `LLAMA_DRAFTER` | DFlash drafter GGUF |
| `LLAMA_TEMPERATURE` / `TOP_P` / `TOP_K` / `MIN_P` / … | sampling |
| `LLAMA_REASONING` | `on` / `off` / `auto` |
| `LLAMA_HOST` / `LLAMA_PORT` | bind (loopback default; non-loopback requires an API-key file) |
| `LLAMA_JINJA` / `LLAMA_CHAT_TEMPLATE_KWARGS` | chat-template handling |
| `LLAMA_CONTEXT_SHIFT` | `0` emits `--no-context-shift` (required for DeltaNet/MTP) |

**Production default (`default.sh`):** 35B MoE Q8_0 · Vulkan (mainline 10358)
· f16 KV · 256k context · MTP `n_max=2` · `--no-context-shift` · `--reasoning on`
· `--jinja` · ubatch 512 · temp 0.6 / top-p 0.95 / top-k 20.

### Dual-GPU profiles

The `preset-36-*` and `preset-38-*` files use **both GPUs** via ROCm (`HIP_VISIBLE_DEVICES=0,1`)
or Vulkan (`HIP_VISIBLE_DEVICES=0,1`). They load on:

- ROCm0 = RX 7900 XTX (24 GB), also drives the desktop
- ROCm1 = AI PRO R9700 (32 GB), compute partner

**Key constraints:**

- The XTX must stay under ~24200 MiB. At 24388 MiB the compositor was evicted to GTT
  and the screen dropped to ~3 fps.
- `--tensor-split 18,22` (ROCm) moves 2 layers to the R9700, freeing ~1.8 GiB on the XTX.
- `MTMD_BACKEND_DEVICE=ROCm1` places the vision tower on the R9700, permanently
  keeping ~0.9 GiB + the vision transient off the constrained XTX.
- `LLAMA_CACHE_RAM=16384` doubles the default to hold more prompt-cache checkpoints.
  The hybrid DeltaNet models can only resume from stored checkpoints, not arbitrary positions.
- `LLAMA_TIMEOUT=3600` is mandatory, llama-server reads `--timeout 0` as a zero-second
  socket timeout, which kills large requests with an empty 400 and no log.

### MTP speculation depth (measured, not guessed)

Draft depth is **not portable** across models. Each model and backend has its own optimum
(measured on the 2026-08-16 harness; f16 KV, dual-GPU):

| Model | Optimal n_max | Dec at opt | Why |
| --- | ---: | ---: | --- |
| 35B MoE (ROCm) | **2** | **94.4** | MoE decode step is fast (7.5 ms @ batch-1); deeper drafts cost more in the verify pass than they save. |
| 35B MoE (Vulkan) | **2** | **93.7** | Same MoE reason; Vulkan verify is cheaper but n_max=3 still collapses. |
| 27B Dense (Vulkan) | **3** | 35.0 | Dense decode step is expensive; deeper drafts pay off until n=4 which collapses. |
| 27B Dense (ROCm) | **2** | 36.9 | ROCm verify pass scaling differs, n_max=3 gains marginal, n_max=4 collapses. |
| Muse Glimmer (ROCm) | **2** | 33.2 | DFlash on this box: n-max sweep 1/2/4/8/15 → acceptance 60.5/43.6/21.7/11.9/6.5%. Upstream recommends 15; it measures **worst** here (was the optimal for Q5_K_M on a **single** GPU). |
| Muse Glimmer (Vulkan) | **2** | 31.7 | Same drafter; dual-GPU verify cost favors shallow depth on both backends. |

Mainline's adaptive-p controller can adapt depth per step within the ceiling.
On the MoE, even with the controller on, n_max=3 is slower than n_max=2 because the
verify pass at 3-wide is already expensive relative to the fast MoE decode.

### ubatch tuning

The single largest prefill win found: raising ubatch from 192 to 512 lifted MoE
prefill from 1319 to 2332 t/s (+77%) with zero decode cost (June 2026).

| Model | ubatch | Prefill t/s | Dec t/s | XTX free |
| --- | ---: | ---: | ---: | ---: |
| 35B MoE @256k | 512 | 2332 | ~167 | 1.40 GiB |
| 35B MoE @256k | 1024 | 2789 | ~172 | 0.83 GiB |
| 27B Dense @128k | 512 | 787 | ~82 | 1.00 GiB |
| 27B Dense @128k | 1024 | 808 | ~82 | 0.56 GiB |

The current shape: ROCm dual-GPU profiles use **ubatch 512** and **batch-size 6144–12288**
(+13% prefill, ROCm-only, a measured no-op on Vulkan, confirmed on three models).
Vulkan stays at batch 2048 (no-op above that). The 35B dual uses ubatch 1024 because
the second GPU provides the headroom. The table above is historical; the 512/1024
comparison is still correct but the final operating point was re-tuned on the
08-16 harness with the larger batch sizes.

---

## Backends & binaries

| Binary | Path | Used for |
| --- | --- | --- |
| mainline ROCm | `../llama.cpp/build/bin/llama-server` | All ROCm dual-GPU profiles (35B, 27B, Qwen3.8, Muse, Nemotron) |
| mainline Vulkan | `../llama.cpp/build-vulkan/bin/llama-server` | All Vulkan profiles (35B, 27B, Qwen3.8, Muse, single-GPU default) |

**Current state:** Mainline 10358 is the only installed engine.
BeeLlama v0.3.2 was removed 2026-08-16.

---

## Speculative decoding

### MTP (self-speculation), the production path

The MTP-head models draft their own next tokens; accepted drafts skip a forward pass.
**`--spec-draft-n-max` is the pinned draft depth on mainline**, the adaptive-p
controller (`--adaptive-target`, off by default) is the only thing that can vary
the effective depth, and production runs with it off.

Key points:

- On the MoE, `n_max=2` is optimal. The MoE's fast decode (7.5 ms per token at
  batch-1) means deeper drafts cost more in the verify pass than they save.
  n_max=3 → 86.08 t/s vs n_max=2 → 95.33 t/s (Vulkan, fixed-depth sweep).
- On the dense 27B Vulkan, `n_max=3` is optimal. The slower dense decode (25.87 ms
  at batch-1) justifies the deeper draft. n_max=4 → 73.5 t/s, n_max=3 → 75.93 t/s.
- Speculation is disabled by tool grammars and reasoning budgets, by design.
  A drafted token could violate the grammar → malformed tool call/JSON.
  No override flag exists.

### DFlash, Muse Glimmer only

Muse Glimmer uses the `dflash-kquant.gguf` sidecar with mainline ROCm or Vulkan.
It drafts blocks of up to 15 tokens; on this box the optimal n_max is **2 on both
backends** (f16 KV, dual-GPU, 2026-08-16 re-measurement). The old 15/4 numbers
were single-GPU Q5_K_M (single GPU makes the verify pass cheap enough for deep
drafts; dual-GPU penalizes them). DFlash and MTP are mutually exclusive.

> **Qwen DFlash is NOT recommended.** MTP beats DFlash on Qwen models by 22% (MoE)
> and 12% (dense), while using less VRAM. The old Qwen DFlash sidecars were
> rejected by both engines (`dflash-draft` vs `dflash` arch mismatch); new sidecars
> exist but MTP still wins.

### Known limitation, post-image prefill regression

Sending **a single image** to a dual-GPU profile permanently drops text prefill ~31–40%
until the server is restarted. Decode is unaffected. Confirmed on both 35B and 27B
profiles. Ruled out: batching, thermals, mmproj device placement, prompt cache, GTT spill.
Workaround: restart the server after heavy image use, or run a text-only profile (no
`--mmproj`) for sessions that will not send images.

---

## KV cache types

| Type | Notes |
| `f16` | full-float KV, the **standing default (2026-08-16)**. Raises speculative
  acceptance (quantized KV makes drafter and target disagree): Muse 43.6→58.5%,
  +19% decode. Use q8_0 only where f16 genuinely does not fit, and record why. |
| `q8_0` | symmetric 8-bit KV; used only where f16 does not fit (single-GPU 24 GB cards,
  Qwen3.8 `ppopt` prefill variant). |
| `kvarn5` / `kvarn4` | Retired with the BeeLlama tree (2026-08-16). Was q6-class quality at q4-class size;
  superseded by f16 KV on dual-GPU. No active profile uses it. |
| `q5_1` / `q5_0` / `q4_0` | smaller/older; Vulkan base only. Archived. |

---

## GPU memory & spill detection

The RX 7900 XTX drives the desktop. When VRAM fills, the amdgpu/RADV driver silently
backs allocations with GTT (system RAM), llama.cpp still reports them as on-GPU,
so decode quietly collapses.

`llamactl` surfaces spill two ways:

- **whole-card** sysfs (`mem_info_vram_*`, `mem_info_gtt_*`), context only; GTT
  always has a baseline from the desktop + pinned buffers.
- **per-process** amdgpu **`fdinfo`** (`drm-memory-vram` / `drm-memory-gtt` of the
  real `llama-server` pid), the direct, desktop-noise-free measure of how much of
  *this server* is in VRAM vs system RAM.

> **Vulkan RADV GTT is misleading.** RADV exposes a 29.71 GiB host-visible heap by design.
> The GTT numbers from `/proc/<pid>/fdinfo` include virtual mappings that are NOT capacity
> spill. On the Vulkan dual-GPU profile, GTT stayed at 4.52 GiB across tensor splits
> 1,1 → 18,22 → 14,26 while free XTX VRAM went from 2.23 to 5.72 GiB. The RADV
> backend always reports elevated GTT. Trust VRAM headroom, not GTT, for Vulkan.

> `common_fit_params: failed to fit params to free device memory … abort` is **NOT** a
> spill signal, it's the auto-layer-fit routine bailing out because `-ngl` is
> user-forced. Ignore it; trust the fdinfo/VRAM numbers.

---

## Benchmarks

All stream a timestamped CSV + per-run logs to `output/benchmarks/<stamp>/`, and
all require the GPU free (`llamakill` first). Each binds port `1235`.

| Command | Script | Finds |
| `llama-benchmark` | `src/benchmarks/matrix.py` | full backend × model × spec × KV matrix (encode/decode t/s, VRAM) |
| `llama-benchmark-mtp` | `src/benchmarks/spec_depth.py` | MTP `n_max` 2→6 sweep on the mainline Vulkan path; depth pinned by default (production); `--adaptive` tests adaptive-p; `--repeats N`; parses `draft acceptance` |
| `llama-benchmark-ctx` | `src/benchmarks/ctx_headroom.py` | **max context before VRAM spill**, per model, via binary search on fdinfo GTT |

`--list` and `--dry-run` work on all three.

### Context-headroom spill rule (`llama-benchmark-ctx`)

A probe counts as **spilled** when the server process has meaningful GPU data in
system RAM (amdgpu `drm-memory-gtt > 0.3 GiB). A fully-resident server sits near
baseline (~0.02–0.2 GiB GTT on ROCm).

### Single-GPU context ceilings (Vulkan, MTP on, f16/q8_0 KV, ubatch 512)

> Historical (May–June 2026, single XTX). The KVarN rows are retired with BeeLlama;
dual-GPU headroom makes these ceilings moot for the tuned profiles.

| Model | KV | Max context | Spills by | procVRAM |
| --- | --- | ---: | ---: | ---: |
| 35B MoE Q8_0 | q8_0 | ≥256k | N/A | 20.10 GiB |
| 35B MoE Q8_0 | f16 | ≥256k | N/A | 20.66 GiB |
| 35B MoE Q4_K_M | q8_0 | ~136k | 144k | 22.49 GiB |
| 35B MoE Q4_K_M | kvarn5/4 | **190k** | 224k | 23.02 GiB |
| 27B MTP Q4_K_M | q8_0 | **195k** | 216k | 23.28 GiB |
| 27B MTP Q4_K_M | kvarn5/4 | **220k** | N/A | 22.59 GiB |

---

## Models (`models/`)

| GGUF | Size | Type |
| --- | ---: | --- |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | 36 GB | MoE + MTP, Q8_0, **dual-GPU primary** |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf` | 33 GB | Mamba-2/attention MoE, **dual-GPU shipped (spec off)** |
| `Qwen3.8-27B-Q8_0.gguf` | 28 GB | Dense + MTP + vision, **dual-GPU (ROCm + Vulkan)** |
| `Muse-Glimmer-30B-Q8_0.gguf` | 28 GB | Dense + DFlash, **dual-GPU (ROCm + Vulkan, YaRN 256k variant)** |
| `Qwen3.6-27B-MTP-Q4_K_M.gguf` | 16 GB | Dense + MTP, Q4_K_M, **single-GPU** |
| `dflash-kquant.gguf` | 1.6 GB | Muse Glimmer DFlash sidecar |
| `mmproj-Qwen3.6-35B-A3B-BF16.gguf` | 862 MB | Qwen3.6 vision projector |
| `mmproj-BF16.gguf` | 889 MB | Qwen3.8 vision projector |

---

## Diagnostics & clients

| Tool | Path | What it does |
| --- | --- | --- |
| `llama-watch` | `src/diagnostics/watch_server.py` | Real-time log watcher; highlights 4xx/5xx responses with ~25 lines of preceding context. Keywords: `error`, `fail`, `template`, `jinja`, `slot`, `draft`, `oom`. |
| `llama-proxy` | `src/diagnostics/llama_proxy.py` | Transparent TCP passthrough (listen :1235 → upstream :1234). Tees every connection's traffic to disk for postmortem of 400/500 responses. |
| `diagnose_hermes_400` | `src/diagnostics/diagnose_hermes_400.py` | Hermes-specific 400 debugging |
| `tool_call_proxy` | `src/server/tool_call_proxy.py` | Forces `parallel_tool_calls: false` on Cline requests (Cline's system prompt batches tool calls; Cline's pipeline expects one per turn). |
| `sync_clients` | `src/server/sync_clients.py` | Pushes alias/context into OpenCode, OpenClaw, Cline, and Kilo config files after a healthy start. |
| `harness` | `bin/harness` | Menu of coding/agent front-ends: Pi, Qwen Code, Hermes, oh-my-pi, OpenCode, OpenClaw, Cline, OpenHands, Kilo Code |

### Local LLM MCP Server

`integrations/agent_tools/local-llm-mcp.js` is a read-only MCP bridge that lets coding
harnesses delegate bounded analysis tasks to the local Qwen model:

- **`local_llm_chat`**, One prompt + optional system instructions. For summaries,
  alternatives, log analysis, second opinions.
- **`local_llm_context`**, Continue a supplied multi-turn conversation.
- **`local_llm_status`**, Verify the local llama.cpp endpoint is reachable.

Default: `LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1`, `LOCAL_LLM_MODEL=local`,
timeout 300s, max_tokens 2048.

---

## Codebase Memory MCP Server

`integrations/agent_tools/codebase-memory-mcp` indexes a codebase into a persistent
knowledge graph (SQLite per-project in `~/.cache/codebase-memory-mcp/`) for structural
queries: call-path tracing, symbol search, architecture visualization.

- 8 MCP tools, ~3k tokens of schemas, safe for local Qwen models
- `auto_watch` is on by default: re-syncs on file changes
- CLI mode: `codebase-memory-mcp cli index_repository --repo_path /path/to/repo`
- Visualization UI at `--ui=true` (port 9749)

---

## Other integrations

| Integration | Size | Description |
| --- | ---: | --- |
| `agent_tools/` | 346 MB | MCP servers: SearXNG, Firecrawl, Filesystem, GitHub, Context7, local-llm, codebase-memory |
| `codegraphcontext/` | 436 MB | Python venv + Jupyter stack |
| `firecrawl/` | 82 MB | Firecrawl MCP server dependencies |
| `pi-emote/` | 16 MB | Pi emotion UI integration |

---

## Paths

```text
config/profiles/default.sh                          canonical production (35B MoE, Vulkan, single-GPU)
config/profiles/preset-36-*                         35B MoE dual-GVP (ROCm + Vulkan)
config/profiles/preset-38-*                         27B Q8_0 dual-GPU
config/profiles/archive-<date>/*                    retired profiles, timestamped
~/.local/state/local-llm/interactive-profile.sh     generated by llamaserver
~/.local/state/local-llm/llama-server.log           rotating server log (50 MB × 5)
$XDG_RUNTIME_DIR/local-llm/ (or /tmp/local-llm-$UID/)   supervisor.pid, server.pid, current-profile
output/benchmarks/<stamp>/                          CSV + per-run logs
output/diagnostics/llama-proxy/<stamp>/             TCP passthrough dumps (from llama-proxy)
llama.cpp/                                          mainline llama.cpp 10358 (ROCm + Vulkan builds)
llama.cpp-next/                                     next-line llama.cpp
```

---

## Audit & experiment docs

| Document | Date | Summary |
| --- | --- | --- |
| [QWEN36_SPECULATION_BACKEND_AUDIT.md](docs/QWEN36_SPECULATION_BACKEND_AUDIT.md) | 2026-08-11 | MTP vs DFlash, Vulkan vs ROCm, mainline vs BeeLlama, ubatch tuning, n_max optima for every model |
| [MUSE_GLIMMER_DFLASH_BACKEND_AUDIT.md](docs/MUSE_GLIMMER_DFLASH_BACKEND_AUDIT.md) | 2026-08-11 | Muse Glimmer DFlash profiling, backend comparison |
| [KVARN_EXPERIMENT.md](docs/KVARN_EXPERIMENT.md) | 2026-06-10 | KVarN KV-cache quantization experiments (superseded by f16 KV on dual-GPU) |
| [KV_QUANTIZATION_COMPARISON.md](docs/KV_QUANTIZATION_COMPARISON.md) | 2026-05-27 | KVarN vs turbo vs q8_0 quality-vs-size analysis |
| [SETUP.md](docs/SETUP.md) | 2026-05-27 | Installation and configuration guide |

---
