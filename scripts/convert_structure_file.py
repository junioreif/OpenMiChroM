#!/usr/bin/env python
"""Convert small local CNDB/NDB/PDB/SW structural files."""

from __future__ import annotations

import argparse
from pathlib import Path

from OpenMiChroM.CndbTools import convert_structure_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert small local structural trajectory files. Remote URLs are "
            "not downloaded by this tool."
        )
    )
    parser.add_argument("input", help="Input structural file path.")
    parser.add_argument("output", help="Output structural file path.")
    parser.add_argument("--input-format", default="auto", help="Input format, or 'auto'.")
    parser.add_argument("--output-format", default=None, help="Output format, or infer from output suffix.")
    parser.add_argument("--trajectory", default=None, help="Nested HDF5 trajectory name when needed.")
    parser.add_argument(
        "--indexed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write embedded index metadata for CNDB outputs.",
    )
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CNDB /Header metadata for CNDB outputs.",
    )
    parser.add_argument(
        "--coordinate-dtype",
        default=None,
        help="Reserved for future dtype conversion; currently unsupported.",
    )
    parser.add_argument("--frames", default=None, help="Reserved for future frame filtering.")
    parser.add_argument("--start", type=int, default=None, help="Reserved for future bead start filtering.")
    parser.add_argument("--stop", type=int, default=None, help="Reserved for future bead stop filtering.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args(argv)

    unsupported_filters = {
        "--coordinate-dtype": args.coordinate_dtype,
        "--frames": args.frames,
        "--start": args.start,
        "--stop": args.stop,
    }
    requested = [name for name, value in unsupported_filters.items() if value is not None]
    if requested:
        parser.error(
            "Frame, bead-window, and dtype-filtered conversions are not implemented yet: "
            + ", ".join(requested)
        )

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    result = convert_structure_file(
        args.input,
        output_path,
        input_format=args.input_format,
        output_format=args.output_format,
        trajectory=args.trajectory,
        indexed=args.indexed,
        metadata=args.metadata,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
