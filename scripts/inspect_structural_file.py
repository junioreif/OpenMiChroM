#!/usr/bin/env python
"""Inspect CNDB/NDB/SW structural files without full remote downloads."""

from __future__ import annotations

import argparse
import json
from typing import Any

from OpenMiChroM._structural_io import StructuralFileInfo, detect_structural_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely inspect local or remote CNDB/NDB/SW structural files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="Local structural file path.")
    source.add_argument("--url", help="Remote structural file URL.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4096,
        help="Number of bytes to sample for file signature detection.",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for inspection probes.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    target = args.url or args.path
    info = detect_structural_file(
        target,
        timeout=args.timeout,
        sample_size=args.sample_size,
        verify_ssl=not args.no_verify_ssl,
    )
    if args.json:
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
    else:
        print_human(info)
    return 0


def print_human(info: StructuralFileInfo) -> None:
    fields: list[tuple[str, Any]] = [
        ("source", info.source),
        ("remote", info.is_remote),
        ("file type", info.file_type),
        ("layout", info.layout),
        ("file size", info.file_size),
        ("range supported", info.range_supported),
        ("hdf5", info.detected_hdf5),
        ("text ndb", info.detected_text_ndb),
        ("spacewalk", info.detected_spacewalk),
        ("embedded index", info.has_embedded_index),
        ("direct streaming", info.direct_streaming_supported),
        ("trajectories", _format_list(info.trajectories)),
        ("frame count", info.frame_count),
        ("bead count", info.bead_count),
        ("coordinate paths", _format_list(info.coordinate_paths)),
        ("dtype", info.dtype),
        ("compression", info.compression),
        ("chunks", info.chunks),
    ]
    for label, value in fields:
        print(f"{label:18s}: {value}")
    if info.notes:
        print("notes:")
        for note in info.notes:
            print(f"  - {note}")


def _format_list(values: list[str], *, limit: int = 20) -> str | None:
    if not values:
        return None
    shown = ", ".join(values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown


if __name__ == "__main__":
    raise SystemExit(main())
