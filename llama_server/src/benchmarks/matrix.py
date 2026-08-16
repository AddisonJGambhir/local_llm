#!/usr/bin/env python3
"""Benchmark llama-server backend/model/spec/KV-cache combinations safely.

Matrix policy (2026-08): one row per **downloaded GGUF** × every spec mode that
GGUF supports × backend (dflash is ROCm-only; base/mtp run on both) × KV.
The KVarN rows were dropped with the BeeLlama v0.3.2 tree (removed 2026-08-16;
f16 KV superseded KVarN on dual-GPU). Rows whose GGUF is no longer downloaded
are skipped at run time. Each GGUF is tested independently so weight-quant
differences show up.
"""

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
LLAMA_SERVER_DIR = str(Path(__file__).resolve().parents[2])
LLM_DIR = str(Path(__file__).resolve().parents[3])
MODELS_DIR = f"{LLAMA_SERVER_DIR}/models"

# Only the mainline llama.cpp tree is installed (ROCm + Vulkan builds). The
# legacy BeeLlama v0.3.2/v0.3.1 and TurboQuant trees were deleted; mainline
# 10358 serves MTP, DFlash and adaptive-p speculation on both backends.
BIN_MAINLINE = f"{LLM_DIR}/llama.cpp/build/bin/llama-server"  # ROCm: q8_0, MTP
BIN_VULKAN = (
    f"{LLM_DIR}/llama.cpp/build-vulkan/bin/llama-server"  # Vulkan mainline: base, MTP
)

# Target models (each a distinct downloaded GGUF — all are benchmarked).
MODEL_35B_MTP_IQ4 = (
    f"{MODELS_DIR}/Qwen3.6-35B-A3B-MTP-UD-IQ4_NL.gguf"  # MoE+MTP, IQ4_NL
)
MODEL_35B_MTP_Q4KM = f"{MODELS_DIR}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"  # MoE+MTP, Q4_K_M
MODEL_35B_MTP_Q8 = (
    f"{MODELS_DIR}/Qwen3.6-35B-A3B-Q8_0.gguf"  # MoE+MTP, Q8_0 — current primary
)
MODEL_35B_MOE = f"{MODELS_DIR}/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"  # MoE plain, IQ4_NL
MODEL_27B_MTP = f"{MODELS_DIR}/Qwen3.6-27B-MTP-Q4_K_M.gguf"  # dense+MTP, Q4_K_M
MODEL_27B = f"{MODELS_DIR}/Qwen3.6-27B-IQ4_NL.gguf"  # dense plain, IQ4_NL

# DFlash drafters (companion models, auto-picked by target architecture).
DRAFT_27B = f"{MODELS_DIR}/Qwen3.6-27B-DFlash-IQ4_XS.gguf"
DRAFT_MOE = f"{MODELS_DIR}/qwen36-35b-a3b-dflash-IQ4_XS.gguf"

DEFAULT_CTX_K = 128

# ── The combo matrix ──────────────────────────────────────────────────────────
# Tuple: (label, model, quant, spec, binary, backend)
#   quant : "q8_0" for symmetric K/V, OR ("kvarn5","kvarn4") for asymmetric KVarN
#   spec  : "none" | "mtp" | "dflash"   (dflash drafter auto-picked by model arch)
#   backend: "rocm" | "vulkan"
CONFIGS = [
    # ══ GGUF: Qwen3.6-35B-A3B-MTP-UD-IQ4_NL  (MoE + MTP heads, IQ4_NL) ══════════
    ("MoE/IQ4(mtp) base q8_0", MODEL_35B_MTP_IQ4, "q8_0", "none", BIN_MAINLINE, "rocm"),
    ("MoE/IQ4(mtp) +MTP q8_0", MODEL_35B_MTP_IQ4, "q8_0", "mtp", BIN_MAINLINE, "rocm"),
    (
        "MoE/IQ4(mtp) base q8_0 (vk)",
        MODEL_35B_MTP_IQ4,
        "q8_0",
        "none",
        BIN_VULKAN,
        "vulkan",
    ),
    (
        "MoE/IQ4(mtp) +MTP q8_0 (vk)",
        MODEL_35B_MTP_IQ4,
        "q8_0",
        "mtp",
        BIN_VULKAN,
        "vulkan",
    ),
    # ══ GGUF: Qwen3.6-35B-A3B-UD-Q4_K_M  (MoE + MTP heads, Q4_K_M) ══════════════
    (
        "MoE/Q4KM(mtp) base q8_0",
        MODEL_35B_MTP_Q4KM,
        "q8_0",
        "none",
        BIN_MAINLINE,
        "rocm",
    ),
    (
        "MoE/Q4KM(mtp) +MTP q8_0",
        MODEL_35B_MTP_Q4KM,
        "q8_0",
        "mtp",
        BIN_MAINLINE,
        "rocm",
    ),
    (
        "MoE/Q4KM(mtp) base q8_0 (vk)",
        MODEL_35B_MTP_Q4KM,
        "q8_0",
        "none",
        BIN_VULKAN,
        "vulkan",
    ),
    (
        "MoE/Q4KM(mtp) +MTP q8_0 (vk)",
        MODEL_35B_MTP_Q4KM,
        "q8_0",
        "mtp",
        BIN_VULKAN,
        "vulkan",
    ),
    # ══ GGUF: Qwen3.6-35B-A3B-Q8_0  (MoE + MTP heads, Q8_0 — current primary) ═══
    ("MoE/Q8(mtp) base q8_0", MODEL_35B_MTP_Q8, "q8_0", "none", BIN_MAINLINE, "rocm"),
    ("MoE/Q8(mtp) +MTP q8_0", MODEL_35B_MTP_Q8, "q8_0", "mtp", BIN_MAINLINE, "rocm"),
    (
        "MoE/Q8(mtp) base q8_0 (vk)",
        MODEL_35B_MTP_Q8,
        "q8_0",
        "none",
        BIN_VULKAN,
        "vulkan",
    ),
    (
        "MoE/Q8(mtp) +MTP q8_0 (vk)",
        MODEL_35B_MTP_Q8,
        "q8_0",
        "mtp",
        BIN_VULKAN,
        "vulkan",
    ),
    # ══ GGUF: Qwen3.6-35B-A3B-UD-IQ4_NL  (MoE plain — base only) ════════════════
    ("MoE/IQ4 base q8_0", MODEL_35B_MOE, "q8_0", "none", BIN_MAINLINE, "rocm"),
    ("MoE/IQ4 base q8_0 (vk)", MODEL_35B_MOE, "q8_0", "none", BIN_VULKAN, "vulkan"),
    # ══ GGUF: Qwen3.6-27B-MTP-Q4_K_M  (dense + MTP heads, Q4_K_M) ═══════════════
    ("27B/Q4KM(mtp) base q8_0", MODEL_27B_MTP, "q8_0", "none", BIN_MAINLINE, "rocm"),
    ("27B/Q4KM(mtp) +MTP q8_0", MODEL_27B_MTP, "q8_0", "mtp", BIN_MAINLINE, "rocm"),
    (
        "27B/Q4KM(mtp) base q8_0 (vk)",
        MODEL_27B_MTP,
        "q8_0",
        "none",
        BIN_VULKAN,
        "vulkan",
    ),
    (
        "27B/Q4KM(mtp) +MTP q8_0 (vk)",
        MODEL_27B_MTP,
        "q8_0",
        "mtp",
        BIN_VULKAN,
        "vulkan",
    ),
    # ══ GGUF: Qwen3.6-27B-IQ4_NL  (dense plain — base only) ═════════════════════
    ("27B/IQ4 base q8_0", MODEL_27B, "q8_0", "none", BIN_MAINLINE, "rocm"),
    ("27B/IQ4 base q8_0 (vk)", MODEL_27B, "q8_0", "none", BIN_VULKAN, "vulkan"),
]

PROMPT_CORPUS = (
    "A production inference service receives concurrent requests, tokenizes input, "
    "allocates cache pages, executes attention and feed-forward layers, samples the "
    "next token, and records latency and throughput. Engineers inspect queueing, "
    "memory pressure, batching, error handling, and reproducibility before changing "
    "the deployment. The benchmark should represent sustained work rather than a "
    "single tiny request whose fixed overhead dominates the measurement. "
)

DEFAULT_MAX_TOKENS = 256
DEFAULT_HEALTH_WAIT = 240
POLL_INTERVAL = 2
DEFAULT_PORT = 1235
DEFAULT_UBATCH = 192
DEFAULT_MIN_FREE_VRAM_GIB = 20.0

# ── Helpers ───────────────────────────────────────────────────────────────────


def quant_label(q) -> str:
    return f"{q[0]}/{q[1]}" if isinstance(q, (tuple, list)) else q


def quant_kv(q) -> tuple[str, str]:
    return (q[0], q[1]) if isinstance(q, (tuple, list)) else (q, q)


def default_drafter_for_model(model_path: str) -> str:
    return DRAFT_MOE if "35B-A3B" in model_path else DRAFT_27B


def model_label(model_path: str, spec: str) -> str:
    """Far-left CSV column: the exact GGUF(s) used. DFlash also loads a drafter."""
    name = os.path.basename(model_path)
    if spec == "dflash":
        name += f" + {os.path.basename(default_drafter_for_model(model_path))}"
    return name


def make_prompt(target_chars: int) -> str:
    repetitions = (target_chars // len(PROMPT_CORPUS)) + 1
    body = (PROMPT_CORPUS * repetitions)[:target_chars]
    return (
        "<|im_start|>user\n"
        "Read the following technical notes and summarize the operational risks.\n\n"
        f"{body}"
        "\n<|im_end|>\n<|im_start|>assistant\n"
    )


SHORT_PROMPT = make_prompt(2048)
LONG_PROMPT = make_prompt(16384)


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(LLAMA_SERVER_DIR) / "output" / "benchmarks" / stamp
    return run_dir / f"benchmark-{stamp}.csv"


def physical_core_count() -> int:
    result = subprocess.run(
        ["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True, check=False
    )
    cores = {
        tuple(line.split(",")[:2])
        for line in result.stdout.splitlines()
        if line and not line.startswith("#") and "," in line
    }
    return len(cores) or (os.cpu_count() or 1)


def port_is_available(port: int) -> bool:
    # SO_REUSEADDR so a just-stopped server's TIME_WAIT sockets don't read as
    # "occupied" — this matches how llama-server itself binds. A live LISTEN
    # socket still fails the bind, so a genuine conflict is still detected.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def check_vram_free_or_die(min_free_gib: float = 20.0):
    import glob

    for total_path in sorted(
        glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
    ):
        used_path = total_path.replace("vram_total", "vram_used")
        try:
            with open(total_path) as f:
                total = int(f.read().strip())
            if total < 8 * 1024**3:
                continue
            with open(used_path) as f:
                used = int(f.read().strip())
            free_gib = (total - used) / 1024**3
            print(
                f"  VRAM: {used / 1024**3:.2f} GiB used / {total / 1024**3:.2f} GiB total, {free_gib:.2f} GiB free"
            )
            if free_gib < min_free_gib:
                procs = subprocess.run(
                    ["bash", "-c", "pgrep -af llama-server | grep -v grep"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                print(
                    f"\n  *** ABORT: only {free_gib:.1f} GiB free, need >= {min_free_gib} GiB ***"
                )
                if procs:
                    print("  Something is already on the GPU:")
                    for line in procs.splitlines():
                        print(f"    {line}")
                print("  Stop the production server with: bin/llamactl stop")
                sys.exit(2)
            return
        except Exception:
            pass
    print("  VRAM: unknown (no /sys/class/drm/card*/device/mem_info_*)")


def wait_for_health(port: int, timeout: int, proc: subprocess.Popen) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # exited (OOM, bad args, KVarN abort, etc.)
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    return False


def get_vram_used_gib() -> str:
    import glob

    for used_path in sorted(
        glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
    ):
        total_path = used_path.replace("vram_used", "vram_total")
        try:
            with open(total_path) as f:
                total = int(f.read().strip())
            if total < 8 * 1024**3:
                continue
            with open(used_path) as f:
                used = int(f.read().strip())
            return f"{used / 1024**3:.1f} GiB"
        except Exception:
            pass
    return "N/A"


def run_completion(port: int, prompt: str, max_tokens: int) -> dict | None:
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "cache_prompt": False,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"      [completion error] {e}")
        return None


def measure(port: int, prompt: str, label: str, max_tokens: int) -> dict | None:
    """Return server-reported timings. The first run is discarded as warmup."""
    print(f"      warmup  [{label}]...")
    if run_completion(port, prompt, min(max_tokens, 32)) is None:
        return None
    print(f"      timed   [{label}]...")
    result = run_completion(port, prompt, max_tokens)
    if result is None:
        return None
    t = result.get("timings", {})
    measured = {
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        "encode": float(t.get("prompt_per_second", 0.0)),
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        "decode": float(t.get("predicted_per_second", 0.0)),
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        "prompt_n": int(t.get("prompt_n", 0)),
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        "predicted_n": int(t.get("predicted_n", 0)),
    }
    if measured["prompt_n"] <= 0 or measured["predicted_n"] < max_tokens:
        print(
            "      [measurement error] incomplete token counts: "
            f"prompt={measured['prompt_n']} predicted={measured['predicted_n']}/{max_tokens}"
        )
        return None
    print(
        f"        encode {measured['encode']:.1f} t/s "
        f"({measured['prompt_n']} tokens)  "
        f"decode {measured['decode']:.1f} t/s "
        f"({measured['predicted_n']} tokens)"
    )
    return measured


def build_server_cmd(
    cfg: tuple, ctx_size: int, port: int, ubatch_size: int
) -> list[str]:
    model_path, quant, spec, binary, backend = cfg[1:]
    qk, qv = quant_kv(quant)

    physical_cores = physical_core_count()
    logical_threads = os.cpu_count() or physical_cores * 2

    cmd = [
        binary,
        "-m",
        model_path,
        "--alias",
        "local",
        "-ngl",
        "99",
        # BeeLlama v0.4.3 makes a failed auto-fit FATAL, and auto-fit always
        # aborts when -ngl is explicit. Without this, larger models fail to
        # load outright. Matches common.sh and ctx_headroom.py.
        "-fit",
        "off",
        "-c",
        str(ctx_size),
        "-t",
        str(physical_cores),
        "-tb",
        str(logical_threads),
        "--parallel",
        "1",
        "--cache-type-k",
        qk,
        "--cache-type-v",
        qv,
        "--flash-attn",
        "on",
        "--ubatch-size",
        str(ubatch_size),
        "--no-context-shift",
        "--timeout",
        "0",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--log-colors",
        "off",
    ]

    # Vulkan must be pinned to the dGPU (Vulkan0); otherwise it can pick the CPU RADV device.
    if backend == "vulkan":
        cmd += ["--device", "Vulkan0"]

    if spec == "mtp":
        cmd += [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "2",
            "--spec-draft-p-min",
            "0.0",
        ]
        # On Vulkan the draft KV defaults to f16 which overflows VRAM at high context,
        # causing the draft to run on CPU and burn ~55 GB of system RAM. Quantize to q8_0.
        if backend == "vulkan":
            cmd += ["--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0"]
    elif spec == "dflash":
        cmd += [
            "--spec-type",
            "dflash",
            "--spec-draft-model",
            default_drafter_for_model(model_path),
            "--spec-draft-ngl",
            "99",
            "--spec-dflash-cross-ctx",
            "1024",
        ]

    return cmd


def na_row(model, label, quant, spec, backend, ctx_k, status):
    return {
        "model": model,
        "config": label,
        "ctx_k": ctx_k,
        "backend": backend,
        "quant": quant_label(quant),
        "spec": spec,
        "vram": "N/A",
        "short_prompt_tokens": "N/A",
        "short_output_tokens": "N/A",
        "short_encode": "N/A",
        "short_decode": "N/A",
        "long_prompt_tokens": "N/A",
        "long_output_tokens": "N/A",
        "long_encode": "N/A",
        "long_decode": "N/A",
        "status": status,
    }


# ── Incremental CSV: header once, one row appended per finished combo ──────────
CSV_FIELDS = [
    "model",
    "config",
    "ctx_k",
    "backend",
    "quant",
    "spec",
    "vram",
    "short_prompt_tokens",
    "short_output_tokens",
    "short_encode",
    "short_decode",
    "long_prompt_tokens",
    "long_output_tokens",
    "long_encode",
    "long_decode",
    "status",
]


def init_csv(output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv(output_csv: Path, row: dict):
    # open/close per row so the file is flushed immediately and tail -f friendly
    with output_csv.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctx-k", type=int, default=DEFAULT_CTX_K)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH)
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_HEALTH_WAIT)
    parser.add_argument(
        "--min-free-vram-gib", type=float, default=DEFAULT_MIN_FREE_VRAM_GIB
    )
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--backend", choices=("all", "rocm", "vulkan"), default="all")
    parser.add_argument(
        "--filter", help="case-insensitive regex matched against config labels"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--list", action="store_true", help="list selected configurations"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print commands only"
    )
    args = parser.parse_args()

    for name in ("ctx_k", "port", "ubatch_size", "decode_tokens", "startup_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.settle_seconds < 0 or args.min_free_vram_gib < 0:
        parser.error("settle time and minimum free VRAM cannot be negative")
    return args


def selected_configs(args: argparse.Namespace) -> list[tuple]:
    configs = CONFIGS
    if args.backend != "all":
        configs = [cfg for cfg in configs if cfg[5] == args.backend]
    if args.filter:
        try:
            pattern = re.compile(args.filter, re.IGNORECASE)
        except re.error as exc:
            raise SystemExit(f"invalid --filter regex: {exc}") from exc
        configs = [cfg for cfg in configs if pattern.search(cfg[0])]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def terminate_process(proc: subprocess.Popen, timeout: int = 20) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def tail_log(log_path: Path, lines: int = 20) -> None:
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
            -lines:
        ]:
            print(f"    {line}")
    except OSError:
        pass


def main():
    args = parse_args()
    configs = selected_configs(args)
    if not configs:
        raise SystemExit("no benchmark configurations matched")

    ctx_size = args.ctx_k * 1024
    output_csv = (args.output or default_output_path()).expanduser().resolve()

    if args.list:
        for index, cfg in enumerate(configs, 1):
            print(
                f"{index:2d}. {cfg[0]} [{cfg[5]}] "
                f"model={os.path.basename(cfg[1])} "
                f"KV={quant_label(cfg[2])} spec={cfg[3]}"
            )
        return

    if args.dry_run:
        for cfg in configs:
            label, model_path, quant, spec, binary, backend = cfg
            problems = []
            if not os.path.isfile(model_path):
                problems.append("model missing")
            if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
                problems.append("binary missing/not executable")
            if spec == "dflash" and not os.path.isfile(
                default_drafter_for_model(model_path)
            ):
                problems.append("drafter missing")
            cmd = build_server_cmd(cfg, ctx_size, args.port, args.ubatch_size)
            print(f"\n{label} [{backend}] {'; '.join(problems) if problems else 'OK'}")
            print("  " + " ".join(subprocess.list2cmdline([part]) for part in cmd))
        return

    if output_csv.exists():
        raise SystemExit(
            f"refusing to overwrite existing output: {output_csv}\n"
            "Choose a different --output path; timestamped output is the default."
        )
    if not port_is_available(args.port):
        raise SystemExit(
            f"benchmark port {args.port} is already in use; refusing to kill its owner"
        )

    results = []

    def record(row: dict):
        results.append(row)
        append_csv(output_csv, row)

    print("=" * 70)
    print(f"  llama.cpp Benchmark — {len(configs)} combos @ {args.ctx_k}k context")
    print("=" * 70)
    print("\n  Pre-flight VRAM check...")
    check_vram_free_or_die(min_free_gib=args.min_free_vram_gib)
    print()

    init_csv(output_csv)
    logs_dir = (
        output_csv.parent / "logs"
        if args.output is None
        else output_csv.parent / f"{output_csv.stem}-logs"
    )
    logs_dir.mkdir(parents=True, exist_ok=False)
    print(f"  Live results streaming to {output_csv}")
    print(f"  Per-config logs: {logs_dir}\n")

    active_proc: subprocess.Popen | None = None
    active_log = None
    try:
        for i, cfg in enumerate(configs):
            label, model_path, quant, spec, binary, backend = cfg
            model = model_label(model_path, spec)

            if not os.path.isfile(model_path):
                print(
                    f"\n[{i + 1}/{len(configs)}] SKIP {label} — model missing: {model_path}"
                )
                record(
                    na_row(
                        model, label, quant, spec, backend, args.ctx_k, "model_missing"
                    )
                )
                continue
            if spec == "dflash":
                drafter = default_drafter_for_model(model_path)
                if not os.path.isfile(drafter):
                    print(
                        f"\n[{i + 1}/{len(configs)}] SKIP {label} — drafter missing: {drafter}"
                    )
                    record(
                        na_row(
                            model,
                            label,
                            quant,
                            spec,
                            backend,
                            args.ctx_k,
                            "draft_missing",
                        )
                    )
                    continue
            if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
                print(
                    f"\n[{i + 1}/{len(configs)}] SKIP {label} — binary missing/not executable: {binary}"
                )
                record(
                    na_row(
                        model, label, quant, spec, backend, args.ctx_k, "binary_missing"
                    )
                )
                continue
            if not port_is_available(args.port):
                raise RuntimeError(f"benchmark port {args.port} became occupied")

            print(f"\n[{i + 1}/{len(configs)}] {label}  [{backend}]  {model}")
            cmd = build_server_cmd(cfg, ctx_size, args.port, args.ubatch_size)
            print(
                f"  cmd: {Path(binary).parent.parent.parent.name} "
                f"-c {ctx_size} --ubatch-size {args.ubatch_size} "
                f"--cache-type-k {quant_kv(quant)[0]} "
                f"--cache-type-v {quant_kv(quant)[1]} spec={spec}"
            )

            env = os.environ.copy()
            if backend == "vulkan":
                env.pop("HIP_VISIBLE_DEVICES", None)
            else:
                env["HIP_VISIBLE_DEVICES"] = "0"

            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
            log_path = logs_dir / f"{i + 1:02d}-{safe_label}.log"
            active_log = log_path.open("w", encoding="utf-8")
            active_proc = subprocess.Popen(
                cmd,
                stdout=active_log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

            print(
                f"  server log: {log_path} — waiting for /health "
                f"(up to {args.startup_timeout}s)..."
            )
            ready = wait_for_health(args.port, args.startup_timeout, active_proc)

            if not ready:
                active_log.flush()
                exit_code = active_proc.poll()
                status = (
                    f"failed(exit={exit_code})"
                    if exit_code is not None
                    else "startup_timeout"
                )
                print(f"  [FAILED] {status} — last log lines:")
                tail_log(log_path)
                terminate_process(active_proc)
                active_log.close()
                active_proc = None
                active_log = None
                record(na_row(model, label, quant, spec, backend, args.ctx_k, status))
                time.sleep(args.settle_seconds)
                continue

            vram = get_vram_used_gib()
            print(f"  server ready — VRAM in use: {vram}")

            short = measure(args.port, SHORT_PROMPT, "short", args.decode_tokens)
            long = measure(args.port, LONG_PROMPT, "long", args.decode_tokens)

            crashed = active_proc.poll() is not None
            if crashed:
                status = f"crashed_during_gen(exit={active_proc.poll()})"
                print(f"  [WARN] server died during generation: {status}")
            elif short is None or long is None:
                status = "request_or_timing_failed"
            else:
                status = "ok"

            record(
                {
                    "model": model,
                    "config": label,
                    "ctx_k": args.ctx_k,
                    "backend": backend,
                    "quant": quant_label(quant),
                    "spec": spec,
                    "vram": vram,
                    "short_prompt_tokens": short["prompt_n"] if short else "N/A",
                    "short_output_tokens": short["predicted_n"] if short else "N/A",
                    "short_encode": f"{short['encode']:.1f}" if short else "N/A",
                    "short_decode": f"{short['decode']:.1f}" if short else "N/A",
                    "long_prompt_tokens": long["prompt_n"] if long else "N/A",
                    "long_output_tokens": long["predicted_n"] if long else "N/A",
                    "long_encode": f"{long['encode']:.1f}" if long else "N/A",
                    "long_decode": f"{long['decode']:.1f}" if long else "N/A",
                    "status": status,
                }
            )

            terminate_process(active_proc)
            active_log.close()
            active_proc = None
            active_log = None
            print("  server stopped.")
            time.sleep(args.settle_seconds)
    except KeyboardInterrupt:
        print(
            "\nInterrupted; cleaning up the benchmark-owned server...", file=sys.stderr
        )
        raise
    finally:
        if active_proc is not None:
            terminate_process(active_proc)
        if active_log is not None and not active_log.closed:
            active_log.close()

    print("\n")
    print("=" * 140)
    print(
        f"  {'Model':<34} {'Config':<28} {'Bk':<6} {'Quant':<12} {'Spec':>6}  {'VRAM':>9}  "
        f"{'ShEnc':>7} {'ShDec':>7}  {'LgEnc':>7} {'LgDec':>7}  Status"
    )
    print("-" * 140)
    for row in results:
        print(
            f"  {str(row['model'])[:34]:<34} {row['config']:<28} {row['backend']:<6} "
            f"{row['quant']:<12} {row['spec']:>6}  {row['vram']:>9}  "
            f"{row['short_encode']:>7} {row['short_decode']:>7}  "
            f"{row['long_encode']:>7} {row['long_decode']:>7}  {row['status']}"
        )
    print("=" * 140)
    print(
        f"  All combos @ {args.ctx_k}k allocation. "
        "Rates and token counts are server-reported."
    )
    print(f"\n  Results saved to {output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
