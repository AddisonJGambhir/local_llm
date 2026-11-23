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
        model = data["provider"]["local"]["models"]["local"]
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
        model = data["provider"]["local"]["models"]["local"]
        if model.get("name") == args.alias:
            return False
        model["name"] = args.alias
        return True

    print(f"client sync: kilo={update_json(kilo, mutate_kilo)}")
    print(f"client sync: openclaw={update_json(openclaw, mutate_openclaw)}")

    # Pi and oh-my-pi list "local" statically (models.json / models.yml); keep its context + image support current.
    vision = True
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:1234/props?model=local", timeout=5) as r:
            vision = bool((json.load(r).get("modalities") or {}).get("vision", True))
    except Exception:
        pass
    inputs = ["text", "image"] if vision else ["text"]

    def fix_local(models: list) -> bool:
        changed = False
        for model in models:
            if model.get("id") == args.alias:
                if model.get("contextWindow") != args.context or model.get("input") != inputs:
                    model["contextWindow"] = args.context
                    model["input"] = inputs
                    changed = True
        return changed

    pi = Path("~/.pi/agent/models.json").expanduser()
    print(f"client sync: pi={update_json(pi, lambda d: fix_local(d.get('providers', {}).get('local', {}).get('models', [])))}")

    omp = Path("~/.omp/agent/models.yml").expanduser()
    if omp.exists():
        try:
            import yaml
            text = omp.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if fix_local(data.get("providers", {}).get("local", {}).get("models", [])):
                header = "".join(line for line in text.splitlines(True) if line.startswith("#"))
                temporary = omp.with_name(f".{omp.name}.{os.getpid()}.tmp")
                temporary.write_text(header + "\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
                os.replace(temporary, omp)
                print("client sync: omp=updated")
            else:
                print("client sync: omp=unchanged")
        except Exception as exc:  # best effort, like the others
            print(f"client sync: omp=warning: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
