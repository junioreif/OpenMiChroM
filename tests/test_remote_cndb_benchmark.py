"""Unit tests for the remote CNDB benchmark's offline helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_remote_cndb.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_remote_cndb", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_rows_uses_median_and_preserves_byte_check():
    module = _load_benchmark_module()
    rows = []
    for repeat, elapsed in enumerate((0.4, 0.2, 0.3), start=1):
        rows.append(
            {
                "phase": "read",
                "operation": "single_frame",
                "method": "exact_range",
                "repeat": repeat,
                "frames": "1",
                "requested_beads": 10,
                "elapsed_seconds": elapsed,
                "data_bytes": 120,
                "total_bytes": 120,
                "expected_coordinate_bytes": 120,
                "byte_match": True,
            }
        )

    summary = module.aggregate_rows(rows)

    assert len(summary) == 1
    assert summary[0]["median_seconds"] == 0.3
    assert summary[0]["data_bytes"] == 120
    assert summary[0]["all_byte_checks_passed"] is True


def test_parse_csv_values_rejects_empty_input():
    module = _load_benchmark_module()

    try:
        module.parse_csv_values(" , ")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("Expected empty comma-separated input to fail.")
