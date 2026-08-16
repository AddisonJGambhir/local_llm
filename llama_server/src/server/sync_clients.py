#!/usr/bin/env python3
"""Best-effort synchronization of local client metadata with the active profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable


def update_json(path: Path, mutate: Callable[[dict[str, Any]], bool]) -> str:
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not mutate(data):
            return "unchanged"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return "updated"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f"warning: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", required=True)
    parser.add_argument("--context", required=True, type=int)
    args = parser.parse_args()

    opencode = Path("~/.config/opencode/opencode.json").expanduser()

    def mutate_opencode(data: dict[str, Any]) -> bool:
        model = data["provider"]["llama-local"]["models"]["local"]
        if model.get("name") == args.alias:
            return False
        model["name"] = args.alias
        return True

    openclaw = Path("~/.openclaw/openclaw.json").expanduser()

    def mutate_openclaw(data: dict[str, Any]) -> bool:
        changed = False
        models = (
            data.get("models", {})
            .get("providers", {})
            .get("custom-localhost-1234", {})
            .get("models", [])
        )
        for model in models:
            if (
                model.get("id") == args.alias
                and model.get("contextWindow") != args.context
            ):
                model["contextWindow"] = args.context
                changed = True
        return changed

    print(f"client sync: opencode={update_json(opencode, mutate_opencode)}")

    cline = Path("~/.cline/data/settings/providers.json").expanduser()

    def mutate_cline(data: dict[str, Any]) -> bool:
        settings = data["providers"]["openai-compatible"]["settings"]
        models = settings.setdefault("models", [])
        model = next((item for item in models if item.get("id") == args.alias), None)
        if model is None:
            models.append({"id": args.alias, "contextWindow": args.context})
            return True
        if model.get("contextWindow") == args.context:
            return False
        model["contextWindow"] = args.context
        return True

    print(f"client sync: cline={update_json(cline, mutate_cline)}")

    kilo = Path("~/.config/kilo/kilo.jsonc").expanduser()

    def mutate_kilo(data: dict[str, Any]) -> bool:
        model = data["provider"]["llama-local"]["models"]["local"]
        if model.get("name") == args.alias:
            return False
        model["name"] = args.alias
        return True

    print(f"client sync: kilo={update_json(kilo, mutate_kilo)}")
    print(f"client sync: openclaw={update_json(openclaw, mutate_openclaw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
