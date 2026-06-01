"""Small structural file converters for OpenMiChroM.

The converters here are intentionally conservative and local-file only. They
support small interoperability tasks among simple NDB, CNDB/HDF5, PDB, and
supported SpaceWalk/SW-style layouts without downloading remote files.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import h5py
import numpy as np

from OpenMiChroM._cndb_stream.embedded_writer import (
    finalize_cndb_header,
    initialize_cndb_header,
    write_embedded_index,
)
from OpenMiChroM._cndb_stream.utils import sort_frame_ids

from .detect import detect_structural_file
from .readers import NDBTextReader


TYPE_TO_RESIDUE = {
    "A1": "ASP",
    "A2": "GLU",
    "B1": "HIS",
    "B2": "LYS",
    "B3": "ARG",
    "B4": "ARG",
    "NA": "ASN",
    "UN": "GLY",
}
RESIDUE_TO_TYPE = {
    "ASP": "A1",
    "GLU": "A2",
    "HIS": "B1",
    "LYS": "B2",
    "ARG": "B3",
    "ASN": "NA",
    "GLY": "UN",
}
NUMERIC_TYPE_TO_LABEL = {
    0: "A1",
    1: "A2",
    2: "B1",
    3: "B2",
    4: "B3",
    5: "B4",
    6: "NA",
}


@dataclass
class StructureTrajectory:
    """Small in-memory representation used by local converters."""

    frames: "OrderedDict[str, np.ndarray]"
    types: list[str]
    genomic_positions: np.ndarray | None = None
    title: str = "Converted by OpenMiChroM"

    @property
    def n_beads(self) -> int:
        if self.types:
            return len(self.types)
        if self.frames:
            first = next(iter(self.frames.values()))
            return int(first.shape[0])
        return 0


def convert_structure_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    input_format: str = "auto",
    output_format: str | None = None,
    trajectory: str | None = None,
    indexed: bool = True,
    metadata: bool = True,
) -> Path:
    """Convert a small local structural trajectory file.

    Supported conversions are simple local ``ndb -> cndb``, ``cndb -> ndb``,
    ``ndb -> pdb``, simple ``pdb -> ndb``, supported HDF5 ``sw/swb -> ndb``,
    and simple text SpaceWalk ``sw/spw`` conversions. Remote URLs are rejected
    because conversion is intentionally a local-file operation.
    """

    input_value = str(input_path)
    if _is_url(input_value):
        raise ValueError("Structural conversion is local-file only; remote URLs are not downloaded.")
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    resolved_input = _resolve_input_format(input_path, input_format)
    resolved_output = _resolve_output_format(output_path, output_format)
    if resolved_output is None:
        raise ValueError("Specify output_path or output_format for conversion.")
    if output_path is None:
        output_path = input_path.with_suffix(f".{resolved_output}")
    output_path = Path(output_path)

    trajectory_data = _read_structure(input_path, resolved_input, trajectory=trajectory)
    if resolved_output == "cndb":
        _write_cndb(
            trajectory_data,
            output_path,
            indexed=indexed,
            metadata=metadata,
        )
    elif resolved_output == "ndb":
        _write_ndb(trajectory_data, output_path)
    elif resolved_output == "pdb":
        _write_pdb(trajectory_data, output_path)
    elif resolved_output == "sw":
        _write_text_spacewalk(trajectory_data, output_path)
    else:
        raise ValueError(
            f"Unsupported output format {resolved_output!r}. "
            "Supported outputs are 'cndb', 'ndb', 'pdb', and 'sw'."
        )
    return output_path


def _read_structure(path: Path, input_format: str, *, trajectory: str | None) -> StructureTrajectory:
    if input_format == "ndb":
        return _read_ndb(path)
    if input_format == "pdb":
        return _read_pdb(path)
    if input_format == "sw":
        info = detect_structural_file(path)
        if info.detected_hdf5:
            return _read_hdf5(path, trajectory=trajectory)
        if info.detected_spacewalk:
            return _read_text_spacewalk(path)
        raise ValueError(f"Unsupported SpaceWalk layout in {path}.")
    if input_format in {"cndb", "hdf5"}:
        return _read_hdf5(path, trajectory=trajectory)
    raise ValueError(
        f"Unsupported input format {input_format!r}. "
        "Supported inputs are 'ndb', 'pdb', 'cndb', 'hdf5', 'sw', 'spw', and 'swb'."
    )


def _read_ndb(path: Path) -> StructureTrajectory:
    reader = NDBTextReader(path)
    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for frame_id in reader.frame_ids:
        frames[frame_id] = reader.get_coordinates(frame_id).astype(np.float32, copy=False)
    genomic_positions = reader.genomic_positions
    return StructureTrajectory(
        frames=frames,
        types=[_normalize_type_label(value) for value in reader.types],
        genomic_positions=genomic_positions,
        title=path.stem,
    )


def _read_pdb(path: Path) -> StructureTrajectory:
    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    types: list[str] = []
    current_frame: str | None = None
    current_coords: list[list[float]] | None = None
    current_types: list[str] = []

    def commit() -> None:
        nonlocal types, current_frame, current_coords, current_types
        if current_frame is None or current_coords is None:
            return
        frames[str(int(current_frame))] = np.array(current_coords, dtype=np.float32)
        if not types:
            types = list(current_types)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                commit()
                parts = line.split()
                current_frame = parts[1] if len(parts) > 1 else str(len(frames) + 1)
                current_coords = []
                current_types = []
            elif record in {"ATOM", "HETATM"}:
                if current_coords is None:
                    current_frame = str(len(frames) + 1)
                    current_coords = []
                    current_types = []
                atom = _parse_pdb_atom_line(line)
                current_coords.append([atom["x"], atom["y"], atom["z"]])
                current_types.append(RESIDUE_TO_TYPE.get(atom["residue"], "UN"))
            elif record == "ENDMDL":
                commit()
                current_frame = None
                current_coords = None
                current_types = []
    commit()

    if not frames:
        raise ValueError(f"No ATOM/HETATM coordinates were found in PDB file {path}.")
    return StructureTrajectory(
        frames=frames,
        types=types or ["UN"] * next(iter(frames.values())).shape[0],
        genomic_positions=_default_genomic_positions(next(iter(frames.values())).shape[0]),
        title=path.stem,
    )


def _read_text_spacewalk(path: Path) -> StructureTrajectory:
    """Read a simple trace-style SpaceWalk text file.

    The supported clean-room dialect is the common tab/space-delimited form:
    ``##format=sw1 ...``, an optional ``chromosome start end x y z`` header,
    ``trace N`` frame markers, and rows containing chromosome/start/end plus
    Cartesian coordinates. SpaceWalk text rows do not carry OpenMiChroM type
    labels, so converted beads are assigned ``UN``.
    """

    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    first_genomic: list[tuple[int, int]] = []
    current_frame: str | None = None
    current_coords: list[list[float]] = []
    current_genomic: list[tuple[int, int]] = []
    title = path.stem

    def commit() -> None:
        nonlocal current_frame, current_coords, current_genomic, first_genomic
        if current_frame is None or not current_coords:
            return
        frame_key = str(int(current_frame))
        coords = np.array(current_coords, dtype=np.float32)
        if frames and coords.shape[0] != next(iter(frames.values())).shape[0]:
            raise ValueError(
                f"Inconsistent bead count in SpaceWalk file {path}: "
                f"frame {frame_key} has {coords.shape[0]} beads."
            )
        frames[frame_key] = coords
        if not first_genomic:
            first_genomic = list(current_genomic)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if stripped.startswith("##"):
                title = _spacewalk_title_from_header(stripped, default=title)
                continue
            if lower.startswith("#"):
                continue
            if lower.startswith("chromosome"):
                continue
            if lower.startswith("trace"):
                commit()
                parts = stripped.split()
                trace_index = len(frames)
                if len(parts) > 1:
                    try:
                        trace_index = int(float(parts[1]))
                    except ValueError:
                        trace_index = len(frames)
                current_frame = str(trace_index + 1)
                current_coords = []
                current_genomic = []
                continue

            if current_frame is None:
                current_frame = str(len(frames) + 1)
                current_coords = []
                current_genomic = []
            try:
                parsed = _parse_spacewalk_row(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse SpaceWalk row {line_number} in {path}: {stripped}"
                ) from exc
            current_genomic.append((parsed["start"], parsed["end"]))
            current_coords.append([parsed["x"], parsed["y"], parsed["z"]])

    commit()
    if not frames:
        raise ValueError(f"No trace coordinate rows were found in SpaceWalk file {path}.")

    n_beads = next(iter(frames.values())).shape[0]
    genomic = np.array(first_genomic, dtype=np.int64) if first_genomic else _default_genomic_positions(n_beads)
    return StructureTrajectory(
        frames=frames,
        types=["UN"] * n_beads,
        genomic_positions=genomic,
        title=title,
    )


def _read_hdf5(path: Path, *, trajectory: str | None) -> StructureTrajectory:
    with h5py.File(path, "r") as h5:
        simple_frames = _root_frame_ids(h5)
        if simple_frames:
            frames = OrderedDict((frame_id, h5[frame_id][()].astype(np.float32, copy=False)) for frame_id in simple_frames)
            types = _read_types_dataset(h5["types"]) if "types" in h5 else ["UN"] * next(iter(frames.values())).shape[0]
            return StructureTrajectory(
                frames=frames,
                types=types,
                genomic_positions=_default_genomic_positions(len(types)),
                title=path.stem,
            )

        trajectory_names = _nested_trajectory_names(h5)
        if trajectory_names:
            selected = _select_trajectory(trajectory_names, trajectory)
            group = h5[selected]
            spatial = group["spatial_position"]
            frame_ids = _group_frame_ids(spatial)
            frames = OrderedDict((frame_id, spatial[frame_id][()].astype(np.float32, copy=False)) for frame_id in frame_ids)
            n_beads = next(iter(frames.values())).shape[0] if frames else 0
            types = _read_types_dataset(group["types"]) if "types" in group else ["UN"] * n_beads
            genomic = group["genomic_position"][()] if "genomic_position" in group else _default_genomic_positions(n_beads)
            return StructureTrajectory(
                frames=frames,
                types=types,
                genomic_positions=np.asarray(genomic, dtype=np.int64),
                title=selected,
            )
    raise ValueError(f"Unsupported HDF5 structural layout in {path}.")


def _write_cndb(
    trajectory: StructureTrajectory,
    output_path: Path,
    *,
    indexed: bool,
    metadata: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = _first_frame_dtype(trajectory)
    with h5py.File(output_path, "w") as h5:
        if metadata:
            initialize_cndb_header(
                h5,
                n_beads=trajectory.n_beads,
                coordinate_dtype=dtype,
                indexed=indexed,
            )
        h5.create_dataset("types", data=np.array([label.encode("utf-8") for label in trajectory.types]))
        for frame_id, coords in trajectory.frames.items():
            h5.create_dataset(_numeric_frame_name(frame_id), data=np.asarray(coords))
        if metadata:
            finalize_cndb_header(
                h5,
                n_frames=len(trajectory.frames),
                n_beads=trajectory.n_beads,
                coordinate_dtype=dtype,
                indexed=indexed,
            )
        h5.flush()
    if indexed:
        write_embedded_index(output_path)


def _write_ndb(trajectory: StructureTrajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genomic = _genomic_positions_or_default(trajectory)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("HEADER    NDB File generated by OpenMiChroM\n")
        handle.write(f"TITLE     {trajectory.title}\n")
        handle.write(_seqchr_line(trajectory.types))
        for frame_id, coords in trajectory.frames.items():
            handle.write(f"MODEL { _model_number(frame_id) }\n")
            for bead_index, row in enumerate(np.asarray(coords), start=1):
                type_label = trajectory.types[bead_index - 1] if bead_index - 1 < len(trajectory.types) else "UN"
                start, end = genomic[bead_index - 1]
                handle.write(
                    "CHROM "
                    f"{bead_index:9d} {type_label:>2} C1 {bead_index:9d} "
                    f"{float(row[0]):10.3f} {float(row[1]):10.3f} {float(row[2]):10.3f} "
                    f"{int(start):10d} {int(end):10d}    0.000\n"
                )
            handle.write("ENDMDL\n")
        handle.write("END\n")


def _write_pdb(trajectory: StructureTrajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"TITLE     {trajectory.title}\n")
        for frame_id, coords in trajectory.frames.items():
            handle.write(f"MODEL     {_model_number(frame_id)}\n")
            serial = 1
            for bead_index, row in enumerate(np.asarray(coords), start=1):
                type_label = trajectory.types[bead_index - 1] if bead_index - 1 < len(trajectory.types) else "UN"
                residue = TYPE_TO_RESIDUE.get(type_label, "GLY")
                handle.write(
                    f"ATOM  {serial:5d}  CA  {residue:>3} A{bead_index:4d}    "
                    f"{float(row[0]):8.3f}{float(row[1]):8.3f}{float(row[2]):8.3f}"
                    "  1.00  0.00           C\n"
                )
                serial += 1
            handle.write("ENDMDL\n")
        handle.write("END\n")


def _write_text_spacewalk(trajectory: StructureTrajectory, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    genomic = _genomic_positions_or_default(trajectory)
    title = _spacewalk_safe_field(trajectory.title)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"##format=sw1 name={title}\n")
        handle.write("chromosome\tstart\tend\tx\ty\tz\n")
        for frame_id, coords in trajectory.frames.items():
            handle.write(f"trace {_spacewalk_trace_number(frame_id)}\n")
            for bead_index, row in enumerate(np.asarray(coords), start=1):
                start, end = genomic[bead_index - 1]
                handle.write(
                    "chr1\t"
                    f"{int(start)}\t{int(end)}\t"
                    f"{float(row[0]):.6f}\t{float(row[1]):.6f}\t{float(row[2]):.6f}\n"
                )


def _resolve_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return _normalize_format(input_format)
    suffix_format = _format_from_suffix(path)
    if suffix_format == "pdb":
        return "pdb"
    info = detect_structural_file(path)
    if info.detected_text_ndb:
        return "ndb"
    if info.detected_spacewalk:
        return "sw"
    if info.detected_hdf5:
        return info.file_type if info.file_type != "unknown" else "hdf5"
    return suffix_format


def _resolve_output_format(output_path: str | Path | None, output_format: str | None) -> str | None:
    if output_format is not None:
        return _normalize_format(output_format)
    if output_path is None:
        return None
    return _format_from_suffix(Path(output_path))


def _normalize_format(value: str) -> str:
    text = value.lower().lstrip(".")
    if text in {"swb", "spw"}:
        return "sw"
    if text in {"h5", "hdf5"}:
        return "hdf5"
    return text


def _format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return _normalize_format(suffix)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "ftp"}


def _root_frame_ids(h5: h5py.File) -> list[str]:
    return sort_frame_ids(
        [
            name
            for name, obj in h5.items()
            if isinstance(obj, h5py.Dataset)
            and _is_frame_id(name)
            and len(obj.shape) == 2
            and int(obj.shape[1]) == 3
        ]
    )


def _group_frame_ids(group: h5py.Group) -> list[str]:
    return sort_frame_ids(
        [
            name
            for name, obj in group.items()
            if isinstance(obj, h5py.Dataset)
            and _is_frame_id(name)
            and len(obj.shape) == 2
            and int(obj.shape[1]) == 3
        ]
    )


def _nested_trajectory_names(h5: h5py.File) -> list[str]:
    return sorted(
        name
        for name, obj in h5.items()
        if isinstance(obj, h5py.Group)
        and "spatial_position" in obj
        and isinstance(obj["spatial_position"], h5py.Group)
    )


def _select_trajectory(names: list[str], trajectory: str | None) -> str:
    if trajectory is not None:
        if trajectory not in names:
            raise ValueError(f"Trajectory {trajectory!r} not found. Available trajectories: {names}")
        return trajectory
    if len(names) == 1:
        return names[0]
    raise ValueError(f"Multiple trajectories are available; specify trajectory=. Available: {names}")


def _is_frame_id(value: str) -> bool:
    text = str(value)
    return text.isdigit() or (text.startswith("t_") and text[2:].isdigit())


def _numeric_frame_name(frame_id: str) -> str:
    text = str(frame_id)
    if text.startswith("t_") and text[2:].isdigit():
        return text[2:]
    if text.isdigit():
        return text
    return str(_model_number(text))


def _model_number(frame_id: str) -> int:
    text = str(frame_id)
    if text.startswith("t_") and text[2:].isdigit():
        return int(text[2:])
    if text.isdigit():
        return int(text)
    return 1


def _read_types_dataset(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[()]).reshape(-1)
    return [_normalize_type_label(value) for value in values]


def _normalize_type_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.integer):
        return NUMERIC_TYPE_TO_LABEL.get(int(value), "UN")
    if isinstance(value, (int, np.int64, np.int32)):
        return NUMERIC_TYPE_TO_LABEL.get(int(value), "UN")
    return str(value)


def _default_genomic_positions(n_beads: int) -> np.ndarray:
    return np.array(
        [[index * 50000 + 1, (index + 1) * 50000] for index in range(int(n_beads))],
        dtype=np.int64,
    )


def _genomic_positions_or_default(trajectory: StructureTrajectory) -> np.ndarray:
    if trajectory.genomic_positions is None:
        return _default_genomic_positions(trajectory.n_beads)
    genomic = np.asarray(trajectory.genomic_positions, dtype=np.int64)
    if genomic.shape != (trajectory.n_beads, 2):
        return _default_genomic_positions(trajectory.n_beads)
    return genomic


def _seqchr_line(types: list[str]) -> str:
    return f"SEQCHR   1 C1 {len(types):5d}  {' '.join(types)}\n"


def _first_frame_dtype(trajectory: StructureTrajectory) -> str:
    if not trajectory.frames:
        return "float32"
    return str(next(iter(trajectory.frames.values())).dtype)


def _parse_pdb_atom_line(line: str) -> dict[str, Any]:
    try:
        return {
            "residue": line[17:20].strip() or "GLY",
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
        }
    except ValueError:
        parts = line.split()
        residue = parts[3] if len(parts) > 3 else "GLY"
        for start in (6, 5, max(0, len(parts) - 6)):
            try:
                return {
                    "residue": residue,
                    "x": float(parts[start]),
                    "y": float(parts[start + 1]),
                    "z": float(parts[start + 2]),
                }
            except (IndexError, ValueError):
                continue
    raise ValueError(f"Could not parse PDB ATOM line: {line.rstrip()}")


def _parse_spacewalk_row(line: str) -> dict[str, Any]:
    parts = line.split()
    if len(parts) < 6:
        raise ValueError("SpaceWalk rows require chromosome, start, end, x, y, z fields.")
    return {
        "chromosome": parts[0],
        "start": int(float(parts[1])),
        "end": int(float(parts[2])),
        "x": float(parts[3]),
        "y": float(parts[4]),
        "z": float(parts[5]),
    }


def _spacewalk_title_from_header(line: str, *, default: str) -> str:
    for token in line[2:].split():
        if token.startswith("name="):
            value = token.split("=", 1)[1].strip()
            return value or default
    return default


def _spacewalk_trace_number(frame_id: str) -> int:
    return max(0, _model_number(frame_id) - 1)


def _spacewalk_safe_field(value: str) -> str:
    text = str(value).strip().replace("\t", "_").replace("\n", "_")
    return text.replace(" ", "_") or "converted"
