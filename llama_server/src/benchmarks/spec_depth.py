#!/usr/bin/env python3
"""Sweep MTP self-speculation depth (--spec-draft-n-max) on the Vulkan path.

Question this answers: your production profile drafts 2 tokens per step
(LLAMA_SPEC_DRAFT_N_MAX=2). Is decoding faster at 3/4/5/6?

IMPORTANT — adaptive controller: BeeLlama defaults `dm_adaptive = true` with the
PROFIT controller (common/common.h:392), so `--spec-draft-n-max` is only a
*ceiling* — the server adapts the real draft depth per step within it (probing,
EWMA raise/lower, recalibrated every 1024 tokens). Two regimes:
  * --adaptive    (default): measures "does raising the ceiling help production?"
                  Matches the production server. Effective depth != n_max.
  * --no-adaptive : appends --no-spec-dm-adaptive so depth is pinned at n_max
                  (with --spec-draft-p-min 0). Use this for a TRUE fixed-depth
                  sweep — the clean form of the original question.

Matrix: 3 MTP models x n_max in {2,3,4,5,6}, all on the production path
(BeeLlama v0.3.2 Vulkan, q8_0 KV, q8_0 draft KV, flash-attn on, --no-context-shift).

Measurement: one SHORT-context prompt, a long sustained generation
(--decode-tokens, default 2048), repeated --repeats times (default 1) and
averaged. Greedy decode on Vulkan+MoE is not bit-reproducible, so a single run is
noisy; use --repeats 5 to separate signal from run-to-run variance. Decode t/s is
the headline; draft acceptance (parsed from the server's `draft acceptance = ...`
log line) explains the curve.

Reuses the launch/health/VRAM/completion machinery from matrix.py.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix  # noqa: E402  (sibling module; reuse its proven helpers)

# ── The sweep: production MTP GGUFs x speculation depth ────────────────────────
# Only these three MTP models run on the Vulkan path in practice.
MODELS = [
    ("Qwen3.6-35B-A3B-MTP-UD-IQ4_NL", matrix.MODEL_35B_MTP_IQ4),  # MoE+MTP, IQ4_NL
    ("Qwen3.6-35B-A3B-UD-Q4_K_M", matrix.MODEL_35B_MTP_Q4KM),  # MoE+MTP, Q4_K_M
    ("Qwen3.6-27B-MTP-Q4_K_M", matrix.MODEL_27B_MTP),  # dense+MTP, Q4_K_M
]
DEFAULT_N_MAX = [2, 3, 4, 5, 6]

DEFAULT_CTX_K = 128
DEFAULT_PORT = 1235  # production server holds 1234; benchmark uses a separate port
DEFAULT_UBATCH = matrix.DEFAULT_UBATCH
DEFAULT_DECODE_TOKENS = 2048  # long enough that steady-state decode dominates
DEFAULT_REPEATS = 1
DEFAULT_HEALTH_WAIT = matrix.DEFAULT_HEALTH_WAIT
DEFAULT_MIN_FREE_VRAM_GIB = matrix.DEFAULT_MIN_FREE_VRAM_GIB

# `slot print_timing: ... draft acceptance = 0.65909 (  145 accepted /   220 generated)`
ACCEPT_RE = re.compile(
    r"draft acceptance = ([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)"
)


# ── Server command: the production Vulkan MTP path, only n_max (+adaptive) vary ─
def build_server_cmd(
    model_path: str, ctx_size: int, port: int, ubatch_size: int, n_max: int, adaptive: bool
) -> list[str]:
    physical_cores = matrix.physical_core_count()
    logical_threads = os.cpu_count() or physical_cores * 2
    cmd = [
        matrix.BIN_BEELLAMA_VULKAN,
        "-m", model_path,
        "--alias", "local",
        "--device", "Vulkan0",
        "-ngl", "99",
        "-c", str(ctx_size),
        "-t", str(physical_cores),
        "-tb", str(logical_threads),
        "--parallel", "1",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--flash-attn", "on",
        "--ubatch-size", str(ubatch_size),
        "--no-context-shift",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(n_max),
        "--spec-draft-p-min", "0.0",
        "--spec-draft-type-k", "q8_0",
        "--spec-draft-type-v", "q8_0",
        "--timeout", "0",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--log-colors", "off",
    ]
    if not adaptive:
        # Pin draft depth at exactly n_max (otherwise the PROFIT controller adapts
        # within the ceiling and n_max stops meaning "depth").
        cmd.append("--no-spec-dm-adaptive")
    return cmd


# ── Acceptance is read back from the server log the request just produced ──────
def read_log_from(log_path: Path, offset: int) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()
    except OSError:
        return ""


def parse_acceptance(text: str) -> dict | None:
    matches = ACCEPT_RE.findall(text)
    if not matches:
        return None
    rate, accepted, generated = matches[-1]  # the timed run is the last task logged
    return {"rate": float(rate), "accepted": int(accepted), "generated": int(generated)}


def warmup(port: int, prompt: str, max_tokens: int) -> bool:
    print("      warmup...")
    return matrix.run_completion(port, prompt, min(max_tokens, 32)) is not None


def measure_once(
    port: int, prompt: str, max_tokens: int, log_path: Path, label: str
) -> dict | None:
    """One timed completion; pair its server timings with the draft acceptance
    logged for that same task (isolated by log byte offset)."""
    offset = log_path.stat().st_size if log_path.exists() else 0
    result = matrix.run_completion(port, prompt, max_tokens)
    if result is None:
        return None

    t = result.get("timings", {})
    m = {
        "encode": float(t.get("prompt_per_second", 0.0)),
        "decode": float(t.get("predicted_per_second", 0.0)),
        "prompt_n": int(t.get("prompt_n", 0)),
        "predicted_n": int(t.get("predicted_n", 0)),
    }
    if m["prompt_n"] <= 0 or m["predicted_n"] < max_tokens:
        print(
            f"      [{label}] incomplete token counts: "
            f"prompt={m['prompt_n']} predicted={m['predicted_n']}/{max_tokens}"
        )
        return None

    time.sleep(0.5)  # let the server flush its `print_timing` block to the log
    accept = parse_acceptance(read_log_from(log_path, offset))
    m["accept_rate"] = accept["rate"] if accept else None
    acc_str = f"{accept['rate']:.3f}" if accept else "N/A"
    print(
        f"      {label}: encode {m['encode']:.1f}  decode {m['decode']:.1f} t/s  accept {acc_str}"
    )
    return m


def aggregate(samples: list[dict]) -> dict | None:
    """Mean/stdev over the repeated timed runs of one config."""
    if not samples:
        return None
    decodes = [s["decode"] for s in samples]
    encodes = [s["encode"] for s in samples]
    accepts = [s["accept_rate"] for s in samples if s["accept_rate"] is not None]
    return {
        "runs": len(samples),
        "prompt_n": samples[-1]["prompt_n"],
        "predicted_n": samples[-1]["predicted_n"],
        "encode": statistics.fmean(encodes),
        "decode": statistics.fmean(decodes),
        "decode_std": statistics.stdev(decodes) if len(decodes) > 1 else 0.0,
        "accept": statistics.fmean(accepts) if accepts else None,
        "accept_std": statistics.stdev(accepts) if len(accepts) > 1 else 0.0,
    }


# ── Incremental CSV ───────────────────────────────────────────────────────────
CSV_FIELDS = [
    "model",
    "n_max",
    "adaptive",
    "ctx_k",
    "backend",
    "quant",
    "vram",
    "runs",
    "prompt_tokens",
    "output_tokens",
    "encode",
    "decode",
    "decode_std",
    "accept",
    "accept_std",
    "status",
]


def result_row(
    model: str, n_max: int, adaptive: bool, ctx_k: int, vram: str, agg: dict | None, status: str
) -> dict:
    return {
        "model": model,
        "n_max": n_max,
        "adaptive": "on" if adaptive else "off",
        "ctx_k": ctx_k,
        "backend": "vulkan",
        "quant": "q8_0",
        "vram": vram,
        "runs": agg["runs"] if agg else 0,
        "prompt_tokens": agg["prompt_n"] if agg else "N/A",
        "output_tokens": agg["predicted_n"] if agg else "N/A",
        "encode": f"{agg['encode']:.1f}" if agg else "N/A",
        "decode": f"{agg['decode']:.1f}" if agg else "N/A",
        "decode_std": f"{agg['decode_std']:.1f}" if agg else "N/A",
        "accept": f"{agg['accept']:.4f}" if agg and agg["accept"] is not None else "N/A",
        "accept_std": f"{agg['accept_std']:.4f}" if agg and agg["accept"] is not None else "N/A",
        "status": status,
    }


def na_row(model: str, n_max: int, adaptive: bool, ctx_k: int, status: str) -> dict:
    return result_row(model, n_max, adaptive, ctx_k, "N/A", None, status)


def init_csv(output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv(output_csv: Path, row: dict):
    with output_csv.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(matrix.LLAMA_SERVER_DIR) / "output" / "benchmarks" / f"spec-depth-{stamp}"
    return run_dir / f"spec-depth-{stamp}.csv"


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_n_max(spec: str) -> list[int]:
    try:
        values = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"--n-max must be a comma list of ints: {exc}") from exc
    if not values or any(v < 1 for v in values):
        raise SystemExit("--n-max values must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ctx-k", type=int, default=DEFAULT_CTX_K)
    parser.add_argument(
        "--n-max", default=",".join(map(str, DEFAULT_N_MAX)),
        help="comma list of speculation depths to sweep (default 2,3,4,5,6)",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="timed runs per config, averaged with stdev (default 1; use 5 to beat noise)",
    )
    parser.add_argument(
        "--adaptive", action=argparse.BooleanOptionalAction, default=True,
        help="adaptive draft-max controller. --adaptive (default) measures the "
        "production ceiling; --no-adaptive pins depth = n_max for a true fixed-depth sweep",
    )
    parser.add_argument(
        "--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS,
        help=f"tokens generated per timed run (default {DEFAULT_DECODE_TOKENS})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_HEALTH_WAIT)
    parser.add_argument("--min-free-vram-gib", type=float, default=DEFAULT_MIN_FREE_VRAM_GIB)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list", action="store_true", help="list selected runs")
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands only")
    args = parser.parse_args()

    for name in ("ctx_k", "port", "ubatch_size", "decode_tokens", "startup_timeout", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.settle_seconds < 0 or args.min_free_vram_gib < 0:
        parser.error("settle time and minimum free VRAM cannot be negative")
    args.n_max_values = parse_n_max(args.n_max)
    return args


def selected_runs(n_max_values: list[int]) -> list[tuple[str, str, int]]:
    return [(name, path, n) for name, path in MODELS for n in n_max_values]


def main():
    args = parse_args()
    runs = selected_runs(args.n_max_values)
    ctx_size = args.ctx_k * 1024
    output_csv = (args.output or default_output_path()).expanduser().resolve()

    if args.list:
        for i, (name, path, n) in enumerate(runs, 1):
            print(f"{i:2d}. {name}  n_max={n}  model={os.path.basename(path)}")
        return

    if args.dry_run:
        for name, path, n in runs:
            problems = []
            if not os.path.isfile(path):
                problems.append("model missing")
            if not os.path.isfile(matrix.BIN_BEELLAMA_VULKAN) or not os.access(
                matrix.BIN_BEELLAMA_VULKAN, os.X_OK
            ):
                problems.append("binary missing/not executable")
            cmd = build_server_cmd(path, ctx_size, args.port, args.ubatch_size, n, args.adaptive)
            print(f"\n{name}  n_max={n}  {'; '.join(problems) if problems else 'OK'}")
            print("  " + " ".join(subprocess.list2cmdline([part]) for part in cmd))
        return

    if output_csv.exists():
        raise SystemExit(
            f"refusing to overwrite existing output: {output_csv}\n"
            "Choose a different --output path; timestamped output is the default."
        )
    if not matrix.port_is_available(args.port):
        raise SystemExit(
            f"benchmark port {args.port} is already in use; refusing to kill its owner"
        )

    prompt = matrix.SHORT_PROMPT  # short context on purpose; see module docstring

    results = []

    def record(row: dict):
        results.append(row)
        append_csv(output_csv, row)

    print("=" * 70)
    print(f"  MTP spec-depth sweep — {len(runs)} runs @ {args.ctx_k}k ctx [vulkan, q8_0 KV]")
    print(
        f"  n_max: {args.n_max_values}   adaptive: {'on' if args.adaptive else 'off (depth pinned)'}"
        f"   {args.decode_tokens}-tok gen x {args.repeats}"
    )
    print("=" * 70)
    print("\n  Pre-flight VRAM check...")
    matrix.check_vram_free_or_die(min_free_gib=args.min_free_vram_gib)
    print()

    init_csv(output_csv)
    logs_dir = (
        output_csv.parent / "logs"
        if args.output is None
        else output_csv.parent / f"{output_csv.stem}-logs"
    )
    logs_dir.mkdir(parents=True, exist_ok=False)
    print(f"  Live results streaming to {output_csv}")
    print(f"  Per-run logs: {logs_dir}\n")

    active_proc: subprocess.Popen | None = None
    active_log = None
    try:
        for i, (name, model_path, n_max) in enumerate(runs):
            tag = f"{name}  n_max={n_max}"
            if not os.path.isfile(model_path):
                print(f"\n[{i + 1}/{len(runs)}] SKIP {tag} — model missing: {model_path}")
                record(na_row(name, n_max, args.adaptive, args.ctx_k, "model_missing"))
                continue
            if not os.path.isfile(matrix.BIN_BEELLAMA_VULKAN) or not os.access(
                matrix.BIN_BEELLAMA_VULKAN, os.X_OK
            ):
                print(f"\n[{i + 1}/{len(runs)}] SKIP {tag} — binary missing/not executable")
                record(na_row(name, n_max, args.adaptive, args.ctx_k, "binary_missing"))
                continue
            if not matrix.port_is_available(args.port):
                raise RuntimeError(f"benchmark port {args.port} became occupied")

            print(f"\n[{i + 1}/{len(runs)}] {tag}  [vulkan]")
            cmd = build_server_cmd(model_path, ctx_size, args.port, args.ubatch_size, n_max, args.adaptive)

            env = os.environ.copy()
            env.pop("HIP_VISIBLE_DEVICES", None)  # Vulkan path

            log_path = logs_dir / f"{i + 1:02d}-{name}-nmax{n_max}.log"
            active_log = log_path.open("w", encoding="utf-8")
            active_proc = subprocess.Popen(
                cmd, stdout=active_log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )

            print(f"  server log: {log_path} — waiting for /health (up to {args.startup_timeout}s)...")
            ready = matrix.wait_for_health(args.port, args.startup_timeout, active_proc)
            if not ready:
                active_log.flush()
                exit_code = active_proc.poll()
                status = f"failed(exit={exit_code})" if exit_code is not None else "startup_timeout"
                print(f"  [FAILED] {status} — last log lines:")
                matrix.tail_log(log_path)
                matrix.terminate_process(active_proc)
                active_log.close()
                active_proc = active_log = None
                record(na_row(name, n_max, args.adaptive, args.ctx_k, status))
                time.sleep(args.settle_seconds)
                continue

            vram = matrix.get_vram_used_gib()
            print(f"  server ready — VRAM in use: {vram}")

            samples = []
            if warmup(args.port, prompt, args.decode_tokens):
                for r in range(args.repeats):
                    if active_proc.poll() is not None:
                        break
                    m = measure_once(
                        args.port, prompt, args.decode_tokens, log_path, f"run {r + 1}/{args.repeats}"
                    )
                    if m is not None:
                        samples.append(m)

            agg = aggregate(samples)
            if active_proc.poll() is not None:
                status = f"crashed_during_gen(exit={active_proc.poll()})"
                print(f"  [WARN] server died during generation: {status}")
            elif agg is None:
                status = "request_or_timing_failed"
            elif agg["runs"] < args.repeats:
                status = f"partial({agg['runs']}/{args.repeats})"
            else:
                status = "ok"

            if agg:
                acc = f"  accept {agg['accept']:.3f}" if agg["accept"] is not None else ""
                print(f"  => decode {agg['decode']:.1f} ±{agg['decode_std']:.1f} t/s{acc}")
            record(result_row(name, n_max, args.adaptive, args.ctx_k, vram, agg, status))

            matrix.terminate_process(active_proc)
            active_log.close()
            active_proc = active_log = None
            print("  server stopped.")
            time.sleep(args.settle_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted; cleaning up the benchmark-owned server...", file=sys.stderr)
        raise
    finally:
        if active_proc is not None:
            matrix.terminate_process(active_proc)
        if active_log is not None and not active_log.closed:
            active_log.close()

    print("\n")
    print("=" * 96)
    print(
        f"  {'Model':<30} {'n_max':>5} {'adpt':>4}  {'VRAM':>9}  "
        f"{'Encode':>7} {'Decode':>7} {'±std':>6} {'Accept':>7}  Status"
    )
    print("-" * 96)
    for row in results:
        print(
            f"  {str(row['model'])[:30]:<30} {str(row['n_max']):>5} {row['adaptive']:>4}  "
            f"{row['vram']:>9}  {row['encode']:>7} {row['decode']:>7} {row['decode_std']:>6} "
            f"{row['accept']:>7}  {row['status']}"
        )
    print("=" * 96)
    print("  Decode t/s is the headline; ±std is over --repeats runs; accept = drafts accepted/generated.")
    print(f"\n  Results saved to {output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
