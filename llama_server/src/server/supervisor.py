#!/usr/bin/env python3
"""Run llama-server, rotate its log, and forward termination signals."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def remove_if_owned(path: Path, pid: int) -> None:
    try:
        if path.read_text(encoding="utf-8").strip() == str(pid):
            path.unlink()
    except FileNotFoundError:
        pass


class RotatingWriter:
    def __init__(self, path: Path, max_bytes: int, backups: int) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", errors="replace")

    def rotate_if_needed(self, incoming_bytes: int) -> None:
        self.stream.flush()
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        if current_size + incoming_bytes <= self.max_bytes:
            return

        self.stream.close()
        if self.backups > 0:
            oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
            oldest.unlink(missing_ok=True)
            for index in range(self.backups - 1, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{index}")
                target = self.path.with_name(f"{self.path.name}.{index + 1}")
                if source.exists():
                    source.replace(target)
            if self.path.exists():
                self.path.replace(self.path.with_name(f"{self.path.name}.1"))
        else:
            self.path.unlink(missing_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", errors="replace")

    def write(self, text: str) -> None:
        encoded_size = len(text.encode("utf-8", errors="replace"))
        self.rotate_if_needed(encoded_size)
        self.stream.write(text)
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-pid-file", required=True, type=Path)
    parser.add_argument("--server-pid-file", required=True, type=Path)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--backups", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing server command after --")
    return args


def main() -> int:
    args = parse_args()
    writer = RotatingWriter(args.log_file, args.max_bytes, args.backups)
    supervisor_pid = os.getpid()
    process: subprocess.Popen[str] | None = None

    atomic_write(args.supervisor_pid_file, f"{supervisor_pid}\n")
    writer.write(
        f"\n[{datetime.now().astimezone().isoformat()}] supervisor {supervisor_pid} starting\n"
    )

    def stop_child(signum: int, _frame: object) -> None:
        writer.write(f"[supervisor] received signal {signum}\n")
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)

    try:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        atomic_write(args.server_pid_file, f"{process.pid}\n")
        writer.write(f"[supervisor] server pid {process.pid}\n")

        assert process.stdout is not None
        for line in process.stdout:
            writer.write(line)
        return_code = process.wait()
        writer.write(f"[supervisor] server exited with code {return_code}\n")
        return return_code
    except Exception as exc:
        writer.write(f"[supervisor] fatal error: {exc!r}\n")
        return 1
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if process is not None:
            remove_if_owned(args.server_pid_file, process.pid)
        remove_if_owned(args.supervisor_pid_file, supervisor_pid)
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
