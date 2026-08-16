from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "benchmarks" / "matrix.py"
SPEC = importlib.util.spec_from_file_location("local_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_prompt_sizes_are_materially_different() -> None:
    assert len(benchmark.SHORT_PROMPT) >= 2_000
    assert len(benchmark.LONG_PROMPT) >= 16_000
    assert len(benchmark.LONG_PROMPT) > len(benchmark.SHORT_PROMPT) * 7


def test_vulkan_mtp_command_matches_production_constraints() -> None:
    config = next(
        cfg for cfg in benchmark.CONFIGS if cfg[0] == "MoE+MTP  q8_0   (vulkan)"
    )
    command = benchmark.build_server_cmd(
        config, ctx_size=131_072, port=12_345, ubatch_size=192
    )

    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "12345"
    assert command[command.index("--ubatch-size") + 1] == "192"
    assert command[command.index("--device") + 1] == "Vulkan0"
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-type-k") + 1] == "q8_0"
    assert command[command.index("--spec-draft-type-v") + 1] == "q8_0"
    assert "--no-context-shift" in command


def test_filters_do_not_mutate_global_matrix() -> None:
    original_count = len(benchmark.CONFIGS)
    args = argparse.Namespace(backend="vulkan", filter=r"MoE\+MTP", limit=1)
    selected = benchmark.selected_configs(args)

    assert len(selected) == 1
    assert selected[0][5] == "vulkan"
    assert len(benchmark.CONFIGS) == original_count


def test_failure_rows_preserve_context_and_token_columns() -> None:
    row = benchmark.na_row("test", "q8_0", "none", "vulkan", 128, "binary_missing")

    assert row["ctx_k"] == 128
    assert row["short_prompt_tokens"] == "N/A"
    assert row["long_output_tokens"] == "N/A"
    assert row["status"] == "binary_missing"
