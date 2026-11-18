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
    source.add_argument(
        "--urls",
        help="Text file containing one local path or remote URL per line.",
    )
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
    parser.add_argument(
        "--json-output",
        help="Write machine-readable JSON to this path. Useful with --urls.",
    )
    args = parser.parse_args(argv)

    targets = _targets_from_args(args)
    infos = [
        detect_structural_file(
            target,
            timeout=args.timeout,
            sample_size=args.sample_size,
            verify_ssl=not args.no_verify_ssl,
        )
        for target in targets
    ]
    payload = [info.to_dict() for info in infos]

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(payload if len(payload) > 1 else payload[0], handle, indent=2, sort_keys=True)
            handle.write("\n")

    if args.json:
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2, sort_keys=True))
    else:
        for index, info in enumerate(infos):
            if index:
                print()
                print("-" * 72)
            print_human(info)
    return 0


def _targets_from_args(args: argparse.Namespace) -> list[str]:
    if args.urls:
        targets: list[str] = []
        with open(args.urls, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                targets.append(stripped)
        if not targets:
            raise SystemExit(f"No URLs or paths found in {args.urls}.")
        return targets
    return [args.url or args.path]


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
        ("filters", info.filters),
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
