#!/usr/bin/env python3
"""Find the largest context each model holds before it spills into system RAM.

llama.cpp preallocates the full KV cache for `-c` at load, so spill is decided at
load time, not during generation — we just load, read the server process's amdgpu
fdinfo, and stop. The DIRECT spill signal is `drm-memory-gtt` (how much of THIS
process's GPU data is backed by system RAM): a resident model sits at ~0.1-0.2 GiB
of GTT (pinned buffers); once VRAM fills, KV/compute spill and process GTT jumps to
GiB-scale. We binary-search `-c` per model for the crossover.

Default config matches production: Vulkan (BeeLlama v0.3.2), q8_0 KV, MTP on (q8_0
draft KV). `--kv kvarn` and `--no-mtp` explore how much further those push context.

Reuses matrix.py for launch/health/terminate; needs the GPU free (stop the server
first: `llamactl stop`).
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix  # noqa: E402

MODELS = [
    ("Qwen3.6-35B-A3B-MTP-UD-IQ4_NL", matrix.MODEL_35B_MTP_IQ4),
    ("Qwen3.6-35B-A3B-UD-Q4_K_M", matrix.MODEL_35B_MTP_Q4KM),
    ("Qwen3.6-27B-MTP-Q4_K_M", matrix.MODEL_27B_MTP),
]

DEFAULT_MIN_CTX_K = 16     # lower bound, assumed to fit
DEFAULT_MAX_CTX_K = 256    # Qwen3.6 trains at 262144 (256k); no point above
DEFAULT_GRANULARITY_K = 8  # stop when the fit/spill bracket is this narrow
# A fully-resident server sits near baseline (~0.02-0.2 GiB GTT); empirically
# 0.78 GiB (27B/KVarN @240k) was already slow, so treat any meaningful process
# GTT in system RAM as spill.
DEFAULT_SPILL_GTT_GIB = 0.3       # process GTT above this = spilled into system RAM
DEFAULT_VRAM_FLOOR_GIB = 0.3      # fallback only (no fdinfo): whole-card free floor
DEFAULT_PORT = 1235
DEFAULT_UBATCH = matrix.DEFAULT_UBATCH
DEFAULT_HEALTH_WAIT = 240
DEFAULT_SETTLE = 6.0

KIB = 1024
GIB = 1024 ** 3

DRM_VRAM_RE = re.compile(r"^drm-memory-vram:\s*(\d+)")
DRM_GTT_RE = re.compile(r"^drm-memory-gtt:\s*(\d+)")


# ── GPU memory readers ────────────────────────────────────────────────────────
def dgpu_vram_bytes() -> tuple[int, int] | None:
    """(used, total) bytes for the discrete GPU (VRAM total > 8 GiB)."""
    for total_path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        try:
            total = int(Path(total_path).read_text())
            if total < 8 * GIB:
                continue
            used = int(Path(total_path.replace("vram_total", "vram_used")).read_text())
            return used, total
        except (OSError, ValueError):
            continue
    return None


def proc_gpu_mem_kib(pid: int) -> tuple[int, int] | None:
    """Per-process (vram_kib, gtt_kib) from amdgpu fdinfo — max across the
    process's render-node fds picks the dGPU figures. None if not exposed."""
    vram: list[int] = []
    gtt: list[int] = []
    for f in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = DRM_VRAM_RE.match(line)
                    if m:
                        vram.append(int(m.group(1)))
                        continue
                    m = DRM_GTT_RE.match(line)
                    if m:
                        gtt.append(int(m.group(1)))
        except OSError:
            continue
    if not vram and not gtt:
        return None
    return (max(vram) if vram else 0, max(gtt) if gtt else 0)


# ── Server command (production Vulkan path, variable ctx/KV/spec) ──────────────
def build_server_cmd(model_path, ctx_size, port, ubatch, kv, mtp) -> list[str]:
    # KVarN dropped 2026-08-09 (47% slower encode on gfx1100, see beellama #112).
    # Only types in Vulkan's fa_kv_ok() whitelist belong here — q6_0/q3_*/q2_*
    # silently fall off FlashAttention and run ~10x slower.
    ck = cv = {"q8_0": "q8_0", "q5_1": "q5_1", "q5_0": "q5_0", "q4_0": "q4_0"}[kv]
    cores = matrix.physical_core_count()
    threads = os.cpu_count() or cores * 2
    cmd = [
        # Switched off BeeLlama 2026-08-11: mainline 10358 serves qwen35moe MTP on
        # Vulkan and measured within 0.4%, so headroom must be scanned against the
        # binary the profiles actually use. See docs/QWEN36_SPECULATION_BACKEND_AUDIT.md.
        matrix.BIN_VULKAN,
        "-m", model_path, "--alias", "local",
        "--device", "Vulkan0", "-ngl", "99",
        # A failed auto-fit is fatal and auto-fit always aborts when -ngl is
        # explicit, so keep it off at every context. Matches common.sh.
        "-fit", "off",
        "-c", str(ctx_size), "-t", str(cores), "-tb", str(threads),
        "--parallel", "1",
        "--cache-type-k", ck, "--cache-type-v", cv,
        "--flash-attn", "on", "--ubatch-size", str(ubatch), "--no-context-shift",
        "--port", str(port), "--host", "127.0.0.1", "--timeout", "0", "--log-colors", "off",
    ]
    if mtp:
        cmd += [
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-p-min", "0.0",
            "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0",
        ]
    return cmd


# ── One probe: load at ctx_k, classify fit / spill / fail ─────────────────────
def test_ctx(model_path, ctx_k, log_path, kv, args) -> dict:
    cmd = build_server_cmd(model_path, ctx_k * 1024, args.port, args.ubatch_size, kv, args.mtp)
    env = os.environ.copy()
    env.pop("HIP_VISIBLE_DEVICES", None)  # Vulkan path
    out = {"ctx_k": ctx_k, "result": "fail", "proc_vram_gib": None,
           "proc_gtt_gib": None, "vram_free_gib": None}

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            if not matrix.wait_for_health(args.port, args.startup_timeout, proc):
                out["result"] = "fail"  # never became healthy: too big / load error
                return out
            time.sleep(3)  # let KV + compute buffers settle
            pm = proc_gpu_mem_kib(proc.pid)
            vram = dgpu_vram_bytes()
            if vram:
                out["vram_free_gib"] = (vram[1] - vram[0]) / GIB
            if pm is not None:
                out["proc_vram_gib"] = pm[0] * KIB / GIB
                out["proc_gtt_gib"] = pm[1] * KIB / GIB
            # Spill = the server has meaningful GPU data in system RAM. Process
            # GTT (drm-memory-gtt) is the direct measure: a resident server stays
            # near baseline and real overflow pushes it up. ~0.78 GiB was already
            # slow in practice, so the bar is low. Fall back to whole-card free
            # VRAM only when fdinfo isn't available.
            free, proc_gtt = out["vram_free_gib"], out["proc_gtt_gib"]
            if proc_gtt is not None:
                spilled = proc_gtt > args.spill_gtt_gib
            else:
                spilled = free is not None and free < args.vram_floor_gib
            out["result"] = "spill" if spilled else "fit"
        finally:
            matrix.terminate_process(proc)
    return out


def fmt(x, suffix=" GiB"):
    return f"{x:.2f}{suffix}" if isinstance(x, float) else "N/A"


def find_max_ctx(name, model_path, kv, logs_dir, args) -> dict:
    gran = args.granularity_k
    probes: list[dict] = []

    def probe(ctx_k):
        log_path = logs_dir / f"{name}-{kv}-{ctx_k}k.log"
        r = test_ctx(model_path, ctx_k, log_path, kv, args)
        probes.append(r)
        print(f"    {ctx_k:>4}k -> {r['result']:<5}  procVRAM {fmt(r['proc_vram_gib'])}  "
              f"procGTT {fmt(r['proc_gtt_gib'])}  VRAMfree {fmt(r['vram_free_gib'])}")
        time.sleep(args.settle_seconds)
        return r

    lo, hi = args.min_ctx_k, args.max_ctx_k
    print(f"\n[{name}]  kv={kv} mtp={args.mtp}  searching {lo}k..{hi}k")

    r_lo = probe(lo)
    if r_lo["result"] != "fit":
        return {"name": name, "max_fit_k": None, "spill_k": lo,
                "note": f"does not fit even at {lo}k ({r_lo['result']})", "at": r_lo}

    r_hi = probe(hi)
    if r_hi["result"] == "fit":
        return {"name": name, "max_fit_k": hi, "spill_k": None,
                "note": f">= {hi}k (no spill within tested range)", "at": r_hi}

    best, best_r = lo, r_lo
    spill_k = hi
    while hi - lo > gran:
        mid = (lo + hi) // 2
        mid -= mid % gran
        if mid <= lo:
            mid = lo + gran
        r = probe(mid)
        if r["result"] == "fit":
            best, best_r, lo = mid, r, mid
        else:
            spill_k, hi = mid, mid
    return {"name": name, "max_fit_k": best, "spill_k": spill_k,
            "note": f"fits to ~{best}k, spills by {spill_k}k", "at": best_r}


# ── CSV / args / main ─────────────────────────────────────────────────────────
CSV_FIELDS = ["model", "kv", "mtp", "max_fit_ctx_k", "first_spill_ctx_k",
              "proc_vram_gib_at_max", "proc_gtt_gib_at_max", "vram_free_gib_at_max", "note"]


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    d = Path(matrix.LLAMA_SERVER_DIR) / "output" / "benchmarks" / f"ctx-headroom-{stamp}"
    return d / f"ctx-headroom-{stamp}.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kv", default="q8_0,q5_1",
                   help="comma list of KV types to test per model: q8_0,q5_1,q5_0,q4_0")
    p.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                   help="include the MTP draft context (matches production); --no-mtp tests base")
    p.add_argument("--min-ctx-k", type=int, default=DEFAULT_MIN_CTX_K)
    p.add_argument("--max-ctx-k", type=int, default=DEFAULT_MAX_CTX_K)
    p.add_argument("--granularity-k", type=int, default=DEFAULT_GRANULARITY_K)
    p.add_argument("--spill-gtt-gib", type=float, default=DEFAULT_SPILL_GTT_GIB,
                   help="server-process GTT (system RAM) above this = spilled (default 0.3)")
    p.add_argument("--vram-floor-gib", type=float, default=DEFAULT_VRAM_FLOOR_GIB,
                   help="fallback only (no fdinfo): whole-card free VRAM below this = spill (default 0.3)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH)
    p.add_argument("--startup-timeout", type=int, default=DEFAULT_HEALTH_WAIT)
    p.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE)
    p.add_argument("--min-free-vram-gib", type=float, default=18.0)
    p.add_argument("--output", type=Path)
    p.add_argument("--only", default="",
                   help="substring filter on model name (e.g. Q4_K_M)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.min_ctx_k >= a.max_ctx_k:
        p.error("--min-ctx-k must be < --max-ctx-k")
    if a.granularity_k <= 0 or a.port <= 0:
        p.error("--granularity-k and --port must be positive")
    a.kv_list = [k.strip() for k in a.kv.split(",") if k.strip()]
    bad = [k for k in a.kv_list if k not in ("q8_0", "q5_1", "q5_0", "q4_0")]
    if not a.kv_list or bad:
        p.error(f"--kv must be a comma list of q8_0/q5_1/q5_0/q4_0 (bad: {bad})")
    return a


def main():
    args = parse_args()
    output_csv = (args.output or default_output_path()).expanduser().resolve()

    if args.list:
        for i, (n, pth) in enumerate(MODELS, 1):
            print(f"{i}. {n}  ({os.path.basename(pth)})")
        return
    if args.dry_run:
        for n, pth in MODELS:
            ok = "OK" if os.path.isfile(pth) else "MODEL MISSING"
            for kv in args.kv_list:
                cmd = build_server_cmd(pth, args.max_ctx_k * 1024, args.port, args.ubatch_size, kv, args.mtp)
                print(f"\n{n}  kv={kv}  [{ok}]  search {args.min_ctx_k}k..{args.max_ctx_k}k (±{args.granularity_k}k)")
                print("  " + " ".join(subprocess.list2cmdline([c]) for c in cmd))
        return
    if output_csv.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_csv}")
    if not matrix.port_is_available(args.port):
        raise SystemExit(f"benchmark port {args.port} is in use; refusing to disturb it")

    print("=" * 74)
    print(f"  Context-headroom finder — kv={','.join(args.kv_list)} mtp={args.mtp}")
    print(f"  spill = server-process GTT > {args.spill_gtt_gib} GiB in system RAM")
    print("=" * 74)
    print("\n  Pre-flight VRAM check (GPU must be free — `llamactl stop` if not)...")
    matrix.check_vram_free_or_die(min_free_gib=args.min_free_vram_gib)

    logs_dir = output_csv.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    results = []
    for name, model_path in MODELS:
        if args.only and args.only.lower() not in name.lower():
            continue
        if not os.path.isfile(model_path):
            print(f"\n[{name}] SKIP — model missing")
            continue
        for kv in args.kv_list:
            res = find_max_ctx(name, model_path, kv, logs_dir, args)
            at = res.get("at") or {}
            row = {
                "model": name, "kv": kv, "mtp": args.mtp,
                "max_fit_ctx_k": res["max_fit_k"] if res["max_fit_k"] is not None else "N/A",
                "first_spill_ctx_k": res["spill_k"] if res["spill_k"] is not None else "N/A",
                "proc_vram_gib_at_max": fmt(at.get("proc_vram_gib"), ""),
                "proc_gtt_gib_at_max": fmt(at.get("proc_gtt_gib"), ""),
                "vram_free_gib_at_max": fmt(at.get("vram_free_gib"), ""),
                "note": res["note"],
            }
            results.append(row)
            with output_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
            print(f"  => {name} [{kv}]: {res['note']}")

    print("\n" + "=" * 94)
    print(f"  {'Model':<30} {'KV':>6} {'max fit':>8} {'spills by':>10} {'procVRAM':>9} {'procGTT':>8}")
    print("-" * 94)
    for r in results:
        mk = f"{r['max_fit_ctx_k']}k" if r["max_fit_ctx_k"] != "N/A" else "none"
        sk = f"{r['first_spill_ctx_k']}k" if r["first_spill_ctx_k"] != "N/A" else "-"
        print(f"  {r['model'][:30]:<30} {r['kv']:>6} {mk:>8} {sk:>10} "
              f"{r['proc_vram_gib_at_max']:>9} {r['proc_gtt_gib_at_max']:>8}")
    print("=" * 94)
    print(f"  mtp={args.mtp}. 'max fit' = largest -c with server-process GTT <= {args.spill_gtt_gib} GiB.")
    print(f"\n  Results saved to {output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
