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
    parser.add_argument("--frames", default=None, help="Comma-separated frame IDs to convert.")
    parser.add_argument("--start", type=int, default=None, help="First bead row to convert.")
    parser.add_argument("--stop", type=int, default=None, help="Exclusive bead row stop.")
    parser.add_argument("--max-frames", type=int, default=None, help="Convert at most this many frames.")
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=512,
        help="Refuse conversions whose selected in-memory payload exceeds this size.",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Bypass the in-memory payload safety limit.",
    )
    parser.add_argument("--pdb-atom-name", default="CA", help="PDB atom name for PDB output.")
    parser.add_argument("--pdb-residue-name", default=None, help="Override PDB residue name.")
    parser.add_argument("--pdb-chain-id", default="A", help="PDB chain ID for PDB output.")
    parser.add_argument("--pdb-element", default="C", help="PDB element symbol for PDB output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args(argv)

    unsupported_filters = {"--coordinate-dtype": args.coordinate_dtype}
    requested = [name for name, value in unsupported_filters.items() if value is not None]
    if requested:
        parser.error(
            "Dtype-filtered conversion is not implemented yet: "
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
        frames=args.frames,
        start=args.start,
        stop=args.stop,
        max_frames=args.max_frames,
        max_memory_mb=args.max_memory_mb,
        allow_large=args.allow_large,
        pdb_atom_name=args.pdb_atom_name,
        pdb_residue_name=args.pdb_residue_name,
        pdb_chain_id=args.pdb_chain_id,
        pdb_element=args.pdb_element,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
