"""Validated trajectory converters for OpenMiChroM and NDB formats.

This module provides a reusable, side-effect-free replacement for the historical
command-line scripts in ``mellofariam/NDB-Converters``.  The format behavior is
reimplemented here rather than copied: that repository does not currently
declare a software license.  We acknowledge the original converter work by
Vinicius G. Contessoto, Matheus F. Mello, and Antonio B. Oliveira Junior.

The supported routes are CNDB <-> NDB, NDB <-> PDB, NDB <-> SpaceWalk text
(``.spw``), GRO -> NDB, and Bintu-layout CSV -> NDB.  Binary OpenMiChroM SWB
files are a different HDF5 format and are deliberately not treated as SPW.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

from OpenMiChroM._cndb_stream.exceptions import CNDBFormatError
from OpenMiChroM._cndb_stream.version import (
    CNDB_FORMAT_NAME,
    CNDB_FORMAT_VERSION,
    metadata_from_attrs,
    validate_format_metadata,
)


__all__ = [
    "ConverterError",
    "ConverterWarning",
    "SUPPORTED_CONVERSIONS",
    "convert",
    "ndb_to_cndb",
    "cndb_to_ndb",
    "ndb_to_pdb",
    "pdb_to_ndb",
    "ndb_to_spw",
    "spw_to_ndb",
    "gro_to_ndb",
    "csv_to_ndb",
]


class ConverterError(ValueError):
    """Raised when a trajectory cannot be converted without ambiguity."""


class ConverterWarning(UserWarning):
    """Warn about information that a target format cannot represent."""


TYPE_LABELS = ("A1", "A2", "B1", "B2", "B3", "B4", "NA")
NUMERIC_TYPES = {index: value for index, value in enumerate(TYPE_LABELS)}
LEGACY_ATOM_TYPES = {
    "ZA": "A1",
    "OA": "A2",
    "FB": "B1",
    "SB": "B2",
    "TB": "B3",
    "LB": "B4",
    "UN": "NA",
}
RESIDUE_TYPES = {
    "ASP": "A1",
    "GLU": "A2",
    "HIS": "B1",
    "LYS": "B2",
    "ARG": "B3",
    "LEU": "B4",
    "ASN": "NA",
    "GLY": "NA",
}
PDB_RESIDUES = {
    "A1": "ASP",
    "A2": "GLU",
    "B1": "HIS",
    "B2": "LYS",
    "B3": "ARG",
    "B4": "LEU",
    "NA": "ASN",
}
FORMAT_SUFFIXES = {
    "ndb": ".ndb",
    "cndb": ".cndb",
    "pdb": ".pdb",
    "gro": ".gro",
    "spw": ".spw",
    "csv": ".csv",
}
FORMAT_ALIASES = {
    "spacewalk": "spw",
    "spacewalk-text": "spw",
    "bintu": "csv",
    "bintu-csv": "csv",
    "bintu_2018": "csv",
}
SUPPORTED_CONVERSIONS = frozenset(
    {
        ("ndb", "cndb"),
        ("cndb", "ndb"),
        ("ndb", "pdb"),
        ("pdb", "ndb"),
        ("ndb", "spw"),
        ("spw", "ndb"),
        ("gro", "ndb"),
        ("csv", "ndb"),
    }
)


@dataclass
class TrajectoryData:
    """Common representation used to validate every conversion route."""

    coordinates: list[np.ndarray]
    frame_ids: list[int]
    types: list[str]
    chain_ids: list[str]
    chain_indices: list[int]
    genomic_intervals: np.ndarray
    sigma: np.ndarray
    loops: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinates = [np.asarray(frame, dtype=np.float64) for frame in self.coordinates]
        self.frame_ids = [int(value) for value in self.frame_ids]
        self.types = [_normalize_type(value) for value in self.types]
        self.chain_ids = [str(value).strip() for value in self.chain_ids]
        self.chain_indices = [int(value) for value in self.chain_indices]
        self.genomic_intervals = np.asarray(self.genomic_intervals, dtype=np.int64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        loops = np.asarray(self.loops, dtype=np.int64)
        if loops.size == 0:
            loops = np.empty((0, 2), dtype=np.int64)
        self.loops = loops
        self._validate()

    @property
    def n_beads(self) -> int:
        return len(self.types)

    def _validate(self) -> None:
        if not self.coordinates:
            raise ConverterError("Trajectory contains no coordinate models.")
        if not self.types:
            raise ConverterError("Trajectory contains no beads.")
        if len(self.frame_ids) != len(self.coordinates):
            raise ConverterError("Frame identifiers do not match coordinate models.")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ConverterError("Frame identifiers must be unique.")
        n_beads = self.n_beads
        for frame_id, frame in zip(self.frame_ids, self.coordinates):
            if frame.shape != (n_beads, 3):
                raise ConverterError(
                    f"Frame {frame_id} has shape {frame.shape}; expected ({n_beads}, 3)."
                )
            if not np.isfinite(frame).all():
                raise ConverterError(f"Frame {frame_id} contains NaN or infinite coordinates.")
        if len(self.chain_ids) != n_beads or any(not value for value in self.chain_ids):
            raise ConverterError("Each bead must have a non-empty chain identifier.")
        if len(self.chain_indices) != n_beads or any(value < 1 for value in self.chain_indices):
            raise ConverterError("Each bead must have a positive chain-local index.")
        if self.genomic_intervals.shape != (n_beads, 2):
            raise ConverterError(
                "Genomic intervals must have shape (n_beads, 2), including explicit start/end."
            )
        if np.any(self.genomic_intervals[:, 1] < self.genomic_intervals[:, 0]):
            raise ConverterError("A genomic interval ends before it starts.")
        if self.sigma.shape != (n_beads,) or not np.isfinite(self.sigma).all():
            raise ConverterError("Sigma values must be one finite value per bead.")
        if self.loops.ndim != 2 or self.loops.shape[1] != 2:
            raise ConverterError("Loops must have shape (n_loops, 2).")


def _normalize_type(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConverterError("A chromatin type is not valid UTF-8.") from exc
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        try:
            return NUMERIC_TYPES[int(value)]
        except KeyError as exc:
            raise ConverterError(f"Unsupported numeric CNDB type code {value!r}.") from exc
    if isinstance(value, float) and value.is_integer():
        return _normalize_type(int(value))
    text = str(value).strip().upper()
    text = LEGACY_ATOM_TYPES.get(text, text)
    if text == "UN":
        text = "NA"
    if not text or len(text) > 2 or any(character.isspace() for character in text):
        raise ConverterError(f"Invalid NDB chromatin type {value!r}.")
    return text


def _normalize_format(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().lstrip(".")
    normalized = FORMAT_ALIASES.get(normalized, normalized)
    if normalized not in FORMAT_SUFFIXES:
        raise ConverterError(
            f"Unsupported format {value!r}; choose from {sorted(FORMAT_SUFFIXES)}."
        )
    return normalized


def _source_path(source, expected_format: str | None = None) -> Path:
    try:
        path = Path(source).expanduser()
    except TypeError as exc:
        raise TypeError("Converter source must be a local path-like object.") from exc
    if not path.exists():
        raise FileNotFoundError(f"Converter source does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Converter source is not a regular file: {path}")
    if expected_format is not None:
        expected_suffix = FORMAT_SUFFIXES[expected_format]
        if path.suffix.lower() != expected_suffix:
            raise ConverterError(
                f"Expected a {expected_suffix} source for {expected_format.upper()}, got {path.name!r}."
            )
    return path


def _destination(source: Path, output, output_format: str) -> Path:
    suffix = FORMAT_SUFFIXES[output_format]
    if output is None:
        destination = source.with_suffix(suffix)
    else:
        destination = Path(output).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(suffix)
        elif destination.suffix.lower() != suffix:
            raise ConverterError(
                f"Output for {output_format.upper()} must use the {suffix} suffix: {destination}"
            )
    if destination.resolve(strict=False) == source.resolve(strict=False):
        raise ConverterError("Source and output paths must be different.")
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")
    return destination


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _check_outputs(destinations: Iterable[Path], *, overwrite: bool) -> None:
    for destination in destinations:
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Pass overwrite=True to replace it."
            )
        if destination.exists() and not destination.is_file():
            raise FileExistsError(f"Output path is not a regular file: {destination}")


def _write_text_files(contents: Mapping[Path, str], *, overwrite: bool) -> None:
    destinations = list(contents)
    _check_outputs(destinations, overwrite=overwrite)
    temporary: dict[Path, Path] = {}
    try:
        for destination, text in contents.items():
            temporary[destination] = _temporary_path(destination)
            temporary[destination].write_text(text, encoding="utf-8", newline="\n")
        for destination in destinations:
            os.replace(temporary[destination], destination)
            temporary.pop(destination, None)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _read_loops(loops) -> np.ndarray:
    if loops is None:
        return np.empty((0, 2), dtype=np.int64)
    if isinstance(loops, (str, os.PathLike)):
        path = _source_path(loops)
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if fields[0].upper() == "LOOPS":
                fields = fields[1:]
            if len(fields) < 2:
                raise ConverterError(f"Invalid loop row at {path}:{line_number}.")
            try:
                rows.append((int(fields[0]), int(fields[1])))
            except ValueError as exc:
                raise ConverterError(f"Invalid loop row at {path}:{line_number}.") from exc
        return np.asarray(rows, dtype=np.int64).reshape((-1, 2))
    array = np.asarray(list(loops), dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ConverterError("Loops must contain pairs of bead identifiers.")
    return array


def _loop_text(loops: np.ndarray) -> str:
    return "".join(f"{int(first)} {int(second)}\n" for first, second in loops)


def _generated_intervals(n_beads: int, *, genomic_start: int, resolution: int) -> np.ndarray:
    if resolution <= 0:
        raise ConverterError("resolution must be a positive integer.")
    start = int(genomic_start)
    starts = start + np.arange(n_beads, dtype=np.int64) * int(resolution)
    return np.column_stack((starts, starts + int(resolution) - 1))


def _sequence_override(types, n_beads: int) -> list[str] | None:
    if types is None:
        return None
    if isinstance(types, (str, os.PathLike)):
        path = _source_path(types)
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                values.append(fields[-1])
    else:
        values = list(types)
    normalized = [_normalize_type(value) for value in values]
    if len(normalized) != n_beads:
        raise ConverterError(
            f"Type sequence has {len(normalized)} entries; coordinates contain {n_beads} beads."
        )
    return normalized


def _parse_ndb(path: Path) -> TrajectoryData:
    models: list[list[tuple]] = []
    frame_ids: list[int] = []
    current: list[tuple] | None = None
    current_id: int | None = None
    loops: list[tuple[int, int]] = []
    metadata: dict[str, str] = {}

    def finalize(line_number: int) -> None:
        nonlocal current, current_id
        if current is None:
            return
        if not current:
            raise ConverterError(f"NDB model {current_id} ending near line {line_number} is empty.")
        models.append(current)
        frame_ids.append(int(current_id if current_id is not None else len(models)))
        current = None
        current_id = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        record = raw_line[:6].strip().upper()
        fields = stripped.split()
        if record == "MODEL":
            if current is not None:
                raise ConverterError(
                    f"NDB model beginning before line {line_number} has no ENDMDL/END record."
                )
            if len(fields) < 2:
                raise ConverterError(f"MODEL at {path}:{line_number} has no frame identifier.")
            try:
                current_id = int(fields[1])
            except ValueError as exc:
                raise ConverterError(
                    f"Invalid MODEL identifier at {path}:{line_number}: {fields[1]!r}."
                ) from exc
            current = []
        elif record == "CHROM":
            if current is None:
                current = []
                current_id = 1 if not models else frame_ids[-1] + 1
            try:
                if len(fields) >= 12 and len(fields[3]) == 1:
                    # Current CustomReporter historically formatted the chain as
                    # ``A{number:4d}``, which whitespace-splits into two fields.
                    chain_id = f"{fields[3]}{int(fields[4])}"
                    chain_index = int(fields[5])
                    offset = 6
                elif len(fields) >= 11:
                    chain_id = fields[3]
                    chain_index = int(fields[4])
                    offset = 5
                else:
                    raise ValueError("too few fields")
                serial = int(fields[1])
                kind = _normalize_type(fields[2])
                x, y, z = (float(fields[offset + index]) for index in range(3))
                genomic_start = int(fields[offset + 3])
                genomic_end = int(fields[offset + 4])
                sigma = float(fields[offset + 5])
            except (ValueError, IndexError) as exc:
                raise ConverterError(
                    f"Malformed CHROM record at {path}:{line_number}: {stripped!r}."
                ) from exc
            current.append(
                (
                    serial,
                    kind,
                    chain_id,
                    chain_index,
                    np.array([x, y, z], dtype=np.float64),
                    genomic_start,
                    genomic_end,
                    sigma,
                )
            )
        elif record == "ENDMDL":
            finalize(line_number)
        elif record == "END":
            finalize(line_number)
        elif record == "LOOPS":
            if len(fields) < 3:
                raise ConverterError(f"Malformed LOOPS record at {path}:{line_number}.")
            try:
                loops.append((int(fields[1]), int(fields[2])))
            except ValueError as exc:
                raise ConverterError(f"Malformed LOOPS record at {path}:{line_number}.") from exc
        elif record in {"TITLE", "HEADER", "EXPDTA", "AUTHOR", "ASMBLY"}:
            metadata_key = "genome" if record == "ASMBLY" else record.lower()
            metadata.setdefault(metadata_key, stripped[6:].strip())
    finalize(line_number if "line_number" in locals() else 0)

    if not models:
        raise ConverterError(f"NDB file contains no CHROM records: {path}")
    reference = models[0]
    serials = [row[0] for row in reference]
    if len(set(serials)) != len(serials):
        raise ConverterError("The first NDB model contains duplicate CHROM serial identifiers.")
    reference_metadata = [(row[1], row[2], row[3], row[5], row[6], row[7]) for row in reference]
    coordinates = []
    for frame_id, model in zip(frame_ids, models):
        model_metadata = [(row[1], row[2], row[3], row[5], row[6], row[7]) for row in model]
        if model_metadata != reference_metadata:
            raise ConverterError(
                f"NDB model {frame_id} changes bead count, types, chains, intervals, or sigma."
            )
        coordinates.append(np.vstack([row[4] for row in model]))
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=frame_ids,
        types=[row[1] for row in reference],
        chain_ids=[row[2] for row in reference],
        chain_indices=[row[3] for row in reference],
        genomic_intervals=np.asarray([(row[5], row[6]) for row in reference]),
        sigma=np.asarray([row[7] for row in reference]),
        loops=np.asarray(loops, dtype=np.int64).reshape((-1, 2)),
        metadata=metadata,
    )


def _ndb_text(data: TrajectoryData, *, title: str | None = None) -> str:
    lines = ["HEADER    NDB file generated by OpenMiChroM Converters"]
    title_value = title or data.metadata.get("title") or "Converted OpenMiChroM trajectory"
    lines.append(f"TITLE     {title_value}")
    if data.metadata.get("genome"):
        lines.append(f"ASMBLY    {data.metadata['genome']}")
    for chain_id in dict.fromkeys(data.chain_ids):
        chain_types = [kind for kind, chain in zip(data.types, data.chain_ids) if chain == chain_id]
        for chunk_number, start in enumerate(range(0, len(chain_types), 23), 1):
            chunk = " ".join(chain_types[start : start + 23])
            lines.append(
                f"SEQCHR {chunk_number:3d} {chain_id:4s} {len(chain_types):5d}  {chunk}"
            )
    for frame_id, coordinates in zip(data.frame_ids, data.coordinates):
        lines.append(f"MODEL     {frame_id:4d}")
        previous_chain = None
        for index, (kind, chain, chain_index, interval, bead_sigma, coordinate) in enumerate(
            zip(
                data.types,
                data.chain_ids,
                data.chain_indices,
                data.genomic_intervals,
                data.sigma,
                coordinates,
            ),
            1,
        ):
            if previous_chain is not None and chain != previous_chain:
                lines.append(f"TER    {index:8d}           {previous_chain:4s}")
            lines.append(
                f"CHROM  {index:8d} {kind:2s} {chain:4s} {chain_index:8d} "
                f"{coordinate[0]:12.6f} {coordinate[1]:12.6f} {coordinate[2]:12.6f} "
                f"{int(interval[0]):10d} {int(interval[1]):10d} {bead_sigma:10.6f}"
            )
            previous_chain = chain
        lines.append(f"TER    {data.n_beads + 1:8d}           {previous_chain:4s}")
        lines.append("ENDMDL")
    lines.extend(f"LOOPS {int(first):8d} {int(second):8d}" for first, second in data.loops)
    resolution = 0
    widths = data.genomic_intervals[:, 1] - data.genomic_intervals[:, 0] + 1
    if len(widths) and np.all(widths == widths[0]):
        resolution = int(widths[0])
    lines.append(
        f"MASTER {data.n_beads:8d} {len(set(data.chain_ids)):6d} "
        f"{len(data.loops):6d} {resolution:10d}"
    )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse_cndb(
    path: Path,
    *,
    resolution: int,
    genomic_start: int,
    chromosome: str,
    sigma: float,
) -> TrajectoryData:
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise CNDBFormatError(f"Could not open {path} as CNDB/HDF5: {exc}") from exc
    with handle:
        metadata = metadata_from_attrs(handle.attrs)
        header_group = handle.get("Header")
        if (
            not metadata["cndb_format"]
            and isinstance(header_group, h5py.Group)
        ):
            metadata = metadata_from_attrs(header_group.attrs)
        validate_format_metadata(metadata, source=str(path))
        if "types" not in handle or not isinstance(handle["types"], h5py.Dataset):
            raise CNDBFormatError(f"CNDB file {path} is missing the required 'types' dataset.")
        type_values = np.asarray(handle["types"])
        if type_values.ndim != 1:
            raise CNDBFormatError("CNDB 'types' must be a one-dimensional dataset.")
        types = [_normalize_type(value) for value in type_values]
        frame_names = sorted(
            (
                name
                for name in handle.keys()
                if str(name).isdigit() and isinstance(handle[name], h5py.Dataset)
            ),
            key=int,
        )
        if not frame_names:
            raise CNDBFormatError(f"CNDB file {path} contains no numeric coordinate frames.")
        coordinates = []
        for name in frame_names:
            frame = np.asarray(handle[name], dtype=np.float64)
            if frame.shape != (len(types), 3):
                raise CNDBFormatError(
                    f"CNDB frame {name!r} has shape {frame.shape}; expected ({len(types)}, 3)."
                )
            coordinates.append(frame)

        def optional_array(name: str):
            return np.asarray(handle[name]) if name in handle else None

        chain_values = optional_array("_ndb_chain_ids")
        chain_ids = (
            [_decode_text(value) for value in chain_values]
            if chain_values is not None
            else [str(chromosome)] * len(types)
        )
        chain_indices_array = optional_array("_ndb_chain_indices")
        chain_indices = (
            [int(value) for value in chain_indices_array]
            if chain_indices_array is not None
            else _chain_local_indices(chain_ids)
        )
        intervals = optional_array("_ndb_genomic_intervals")
        if intervals is None:
            intervals = _generated_intervals(
                len(types), genomic_start=int(genomic_start), resolution=int(resolution)
            )
        sigma_values = optional_array("_ndb_sigma")
        if sigma_values is None:
            sigma_values = np.full(len(types), float(sigma))
        loops = optional_array("loops")
        if loops is None:
            loops = np.empty((0, 2), dtype=np.int64)
        text_metadata = {}
        for name in ("title", "name", "genome"):
            if name in handle.attrs:
                text_metadata[name] = _decode_text(handle.attrs[name])
            elif isinstance(header_group, h5py.Group) and name in header_group.attrs:
                text_metadata[name] = _decode_text(header_group.attrs[name])
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=[int(name) for name in frame_names],
        types=types,
        chain_ids=chain_ids,
        chain_indices=chain_indices,
        genomic_intervals=intervals,
        sigma=sigma_values,
        loops=loops,
        metadata=text_metadata,
    )


def _decode_text(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConverterError("Converter metadata is not valid UTF-8.") from exc
    return str(value)


def _chain_local_indices(chain_ids: Sequence[str]) -> list[int]:
    counts: dict[str, int] = {}
    result = []
    for chain in chain_ids:
        counts[chain] = counts.get(chain, 0) + 1
        result.append(counts[chain])
    return result


def _write_cndb(data: TrajectoryData, destination: Path, *, overwrite: bool) -> None:
    _check_outputs([destination], overwrite=overwrite)
    temporary = _temporary_path(destination)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = CNDB_FORMAT_NAME
            handle.attrs["format_version"] = CNDB_FORMAT_VERSION
            handle.attrs["converter"] = "OpenMiChroM.Converters"
            for name, value in data.metadata.items():
                if name in {"title", "name", "genome"} and value:
                    handle.attrs[name] = str(value)
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("types", data=np.asarray(data.types, dtype=object), dtype=string_dtype)
            for frame_id, coordinates in zip(data.frame_ids, data.coordinates):
                handle.create_dataset(str(frame_id), data=coordinates)
            if len(data.loops):
                handle.create_dataset("loops", data=data.loops.astype(np.int64))
            handle.create_dataset(
                "_ndb_chain_ids",
                data=np.asarray(data.chain_ids, dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "_ndb_chain_indices", data=np.asarray(data.chain_indices, dtype=np.int64)
            )
            handle.create_dataset("_ndb_genomic_intervals", data=data.genomic_intervals)
            handle.create_dataset("_ndb_sigma", data=data.sigma)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ndb_to_cndb(source, output=None, *, overwrite: bool = False) -> Path:
    """Convert NDB text to versioned CNDB/HDF5 without losing NDB metadata."""

    source_path = _source_path(source, "ndb")
    destination = _destination(source_path, output, "cndb")
    data = _parse_ndb(source_path)
    _write_cndb(data, destination, overwrite=overwrite)
    return destination


def cndb_to_ndb(
    source,
    output=None,
    *,
    resolution: int = 50_000,
    genomic_start: int = 1,
    chromosome: str = "C1",
    sigma: float = 0.0,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert CNDB/HDF5 frames (including frame 0/noncontiguous IDs) to NDB."""

    source_path = _source_path(source, "cndb")
    destination = _destination(source_path, output, "ndb")
    data = _parse_cndb(
        source_path,
        resolution=resolution,
        genomic_start=genomic_start,
        chromosome=chromosome,
        sigma=sigma,
    )
    _write_text_files({destination: _ndb_text(data, title=title)}, overwrite=overwrite)
    return destination


def _type_from_atom(atom_name: str, residue_name: str) -> tuple[str, bool]:
    atom = atom_name.strip().upper()
    residue = residue_name.strip().upper()
    if atom in TYPE_LABELS or atom in LEGACY_ATOM_TYPES or atom == "UN":
        return _normalize_type(atom), False
    if residue in RESIDUE_TYPES:
        return RESIDUE_TYPES[residue], atom in {"", "CA"} and residue in {"ARG", "HIS"}
    if residue in {"CHRA", "CHRB", "CHRU"}:
        return {"CHRA": "A1", "CHRB": "B1", "CHRU": "NA"}[residue], False
    return "NA", False


def _intervals_by_chain(
    chain_ids: Sequence[str], *, genomic_start: int, resolution: int
) -> np.ndarray:
    counts: dict[str, int] = {}
    intervals = []
    for chain in chain_ids:
        index = counts.get(chain, 0)
        start = int(genomic_start) + index * int(resolution)
        intervals.append((start, start + int(resolution) - 1))
        counts[chain] = index + 1
    return np.asarray(intervals, dtype=np.int64)


def _parse_pdb(
    path: Path,
    *,
    resolution: int,
    genomic_start: int,
    sigma: float,
    scale: float,
    types,
    loops,
) -> TrajectoryData:
    models: list[list[tuple[str, str, int, np.ndarray]]] = []
    frame_ids: list[int] = []
    current: list[tuple[str, str, int, np.ndarray]] | None = None
    current_id: int | None = None
    chain_map: dict[str, str] = {}
    ambiguous_residue = False
    fallback_chain_number = 1

    def finalize(line_number: int) -> None:
        nonlocal current, current_id
        if current is None:
            return
        if not current:
            raise ConverterError(f"PDB model {current_id} ending near line {line_number} is empty.")
        models.append(current)
        frame_ids.append(int(current_id if current_id is not None else len(models)))
        current = None
        current_id = None

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = line[:6].strip().upper()
        if record == "REMARK":
            fields = line.split()
            if len(fields) >= 6 and fields[1:4] == ["900", "OPENMICHROM", "CHAIN"]:
                chain_map[fields[4]] = fields[5]
            continue
        if record == "MODEL":
            if current is not None:
                raise ConverterError(
                    f"PDB model beginning before line {line_number} has no ENDMDL record."
                )
            fields = line.split()
            try:
                current_id = int(fields[1])
            except (IndexError, ValueError) as exc:
                raise ConverterError(f"Invalid MODEL at {path}:{line_number}.") from exc
            current = []
            fallback_chain_number = 1
        elif record in {"ATOM", "HETATM"}:
            if current is None:
                current = []
                current_id = 1 if not models else frame_ids[-1] + 1
            try:
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                pdb_chain = line[21:22].strip()
                residue_index_text = line[22:26].strip()
                chain_index = int(residue_index_text) if residue_index_text else len(current) + 1
                coordinate = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
            except (ValueError, IndexError) as fixed_error:
                fields = line.split()
                try:
                    atom_name = fields[2]
                    residue_name = fields[3]
                    if len(fields) >= 9 and not _is_number(fields[4]):
                        pdb_chain = fields[4]
                        chain_index = int(fields[5])
                        coordinate = np.asarray(fields[6:9], dtype=np.float64)
                    else:
                        pdb_chain = ""
                        chain_index = int(fields[4])
                        coordinate = np.asarray(fields[5:8], dtype=np.float64)
                except (ValueError, IndexError) as exc:
                    raise ConverterError(
                        f"Malformed PDB atom record at {path}:{line_number}: {line!r}."
                    ) from fixed_error
            kind, ambiguous = _type_from_atom(atom_name, residue_name)
            ambiguous_residue = ambiguous_residue or ambiguous
            resolved_chain = (
                chain_map.get(pdb_chain, f"C{pdb_chain}")
                if pdb_chain
                else f"C{fallback_chain_number}"
            )
            current.append(
                (kind, resolved_chain, chain_index, coordinate * scale)
            )
        elif record == "TER":
            fallback_chain_number += 1
        elif record == "ENDMDL":
            finalize(line_number)
        elif record == "END":
            finalize(line_number)
    finalize(line_number if "line_number" in locals() else 0)
    if not models:
        raise ConverterError(f"PDB file contains no ATOM/HETATM records: {path}")
    first = models[0]
    reference = [(row[0], row[1], row[2]) for row in first]
    coordinates = []
    for frame_id, model in zip(frame_ids, models):
        if [(row[0], row[1], row[2]) for row in model] != reference:
            raise ConverterError(f"PDB model {frame_id} changes bead count, types, or chains.")
        coordinates.append(np.vstack([row[3] for row in model]))
    override = _sequence_override(types, len(first))
    inferred_types = override or [row[0] for row in first]
    if ambiguous_residue and override is None:
        warnings.warn(
            "A CA/HIS or CA/ARG PDB bead is ambiguous across historical OpenMiChroM "
            "residue mappings; the current mapping was used. Pass types=... to restore "
            "an exact sequence.",
            ConverterWarning,
            stacklevel=2,
        )
    chain_ids = [row[1] for row in first]
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=frame_ids,
        types=inferred_types,
        chain_ids=chain_ids,
        chain_indices=[row[2] for row in first],
        genomic_intervals=_intervals_by_chain(
            chain_ids, genomic_start=genomic_start, resolution=resolution
        ),
        sigma=np.full(len(first), float(sigma)),
        loops=_read_loops(loops),
        metadata={"title": path.stem},
    )


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _pdb_text(data: TrajectoryData, *, title: str) -> str:
    chain_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    ordered_chains = list(dict.fromkeys(data.chain_ids))
    if len(ordered_chains) > len(chain_characters):
        raise ConverterError("PDB output supports at most 62 distinct chains.")
    chain_map = dict(zip(ordered_chains, chain_characters))
    lines = [f"TITLE     {title}"]
    lines.extend(
        f"REMARK 900 OPENMICHROM CHAIN {character} {chain}"
        for chain, character in chain_map.items()
    )
    for frame_id, coordinates in zip(data.frame_ids, data.coordinates):
        lines.append(f"MODEL     {frame_id:4d}")
        previous_chain = None
        for serial, (kind, chain, chain_index, coordinate) in enumerate(
            zip(data.types, data.chain_ids, data.chain_indices, coordinates), 1
        ):
            if previous_chain is not None and chain != previous_chain:
                lines.append(f"TER   {serial:5d}")
            residue = PDB_RESIDUES.get(kind, "GLY")
            atom_name = kind if len(kind) <= 4 else "CA"
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue:>3s} "
                f"{chain_map[chain]}{chain_index:4d}    "
                f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
                "  1.00  0.00           C"
            )
            previous_chain = chain
        lines.append(f"TER   {data.n_beads + 1:5d}")
        lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


def ndb_to_pdb(
    source,
    output=None,
    *,
    title: str | None = None,
    write_loops: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert NDB to a visualization-friendly, reversibly typed PDB file."""

    source_path = _source_path(source, "ndb")
    destination = _destination(source_path, output, "pdb")
    data = _parse_ndb(source_path)
    contents = {destination: _pdb_text(data, title=title or source_path.stem)}
    if write_loops:
        contents[destination.with_suffix(".loops")] = _loop_text(data.loops)
    _write_text_files(contents, overwrite=overwrite)
    return destination


def pdb_to_ndb(
    source,
    output=None,
    *,
    resolution: int = 50_000,
    genomic_start: int = 1,
    sigma: float = 0.0,
    scale: float = 1.0,
    types=None,
    loops=None,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert PDB models to NDB, accepting modern residues and legacy atom codes."""

    source_path = _source_path(source, "pdb")
    destination = _destination(source_path, output, "ndb")
    if not np.isfinite(scale) or scale <= 0:
        raise ConverterError("scale must be a finite positive number.")
    data = _parse_pdb(
        source_path,
        resolution=resolution,
        genomic_start=genomic_start,
        sigma=sigma,
        scale=float(scale),
        types=types,
        loops=loops,
    )
    _write_text_files({destination: _ndb_text(data, title=title)}, overwrite=overwrite)
    return destination


def _chain_to_chromosome(chain: str) -> str:
    value = str(chain).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    elif value.upper().startswith("C") and len(value) > 1:
        value = value[1:]
    return value or str(chain)


def _chromosome_to_chain(chromosome: str) -> str:
    value = str(chromosome).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    if value.upper().startswith("C") and len(value) > 1:
        return "C" + value[1:]
    return f"C{value}"


def _spw_text(data: TrajectoryData, *, name: str, genome: str) -> str:
    lines = [
        f"##format=sw1 name={shlex.quote(name)} genome={shlex.quote(genome)}",
        "#chromosome start end x y z",
    ]
    for frame_id, coordinates in zip(data.frame_ids, data.coordinates):
        lines.append(f"trace {frame_id - 1}")
        for chain, interval, coordinate in zip(
            data.chain_ids, data.genomic_intervals, coordinates
        ):
            lines.append(
                f"chr{_chain_to_chromosome(chain)} {int(interval[0])} {int(interval[1])} "
                f"{coordinate[0]:.6f} {coordinate[1]:.6f} {coordinate[2]:.6f}"
            )
    return "\n".join(lines) + "\n"


def ndb_to_spw(
    source,
    output=None,
    *,
    name: str | None = None,
    genome: str | None = None,
    write_loops: bool = True,
    overwrite: bool = False,
) -> Path:
    """Convert NDB to SpaceWalk ``sw1`` text; types are not representable in SPW."""

    source_path = _source_path(source, "ndb")
    destination = _destination(source_path, output, "spw")
    data = _parse_ndb(source_path)
    warnings.warn(
        "SpaceWalk SPW does not store MiChroM chromatin types or sigma values; "
        "SPW-to-NDB imports therefore use type NA and the requested sigma.",
        ConverterWarning,
        stacklevel=2,
    )
    selected_genome = genome or data.metadata.get("genome") or "unknown"
    contents = {
        destination: _spw_text(
            data,
            name=name or source_path.stem,
            genome=str(selected_genome),
        )
    }
    if write_loops:
        contents[destination.with_suffix(".loops")] = _loop_text(data.loops)
    _write_text_files(contents, overwrite=overwrite)
    return destination


def _parse_spw(path: Path, *, sigma: float, types, loops) -> TrajectoryData:
    metadata: dict[str, str] = {}
    models: list[list[tuple[str, int, int, np.ndarray]]] = []
    frame_ids: list[int] = []
    current: list[tuple[str, int, int, np.ndarray]] | None = None
    current_id: int | None = None

    def finalize(line_number: int) -> None:
        nonlocal current, current_id
        if current is None:
            return
        if not current:
            raise ConverterError(f"SPW trace {current_id} ending near line {line_number} is empty.")
        models.append(current)
        frame_ids.append(int(current_id if current_id is not None else len(models)))
        current = None
        current_id = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("##"):
            for token in shlex.split(line[2:]):
                if "=" in token:
                    key, value = token.split("=", 1)
                    metadata[key.lower()] = value
            continue
        if line.startswith("#"):
            continue
        fields = line.split()
        if [field.lower() for field in fields[:6]] == [
            "chromosome",
            "start",
            "end",
            "x",
            "y",
            "z",
        ]:
            continue
        if fields[0].lower() == "trace":
            finalize(line_number)
            try:
                current_id = int(fields[1]) + 1
            except (IndexError, ValueError) as exc:
                raise ConverterError(f"Invalid trace record at {path}:{line_number}.") from exc
            current = []
            continue
        if current is None:
            current = []
            current_id = 1 if not models else frame_ids[-1] + 1
        if len(fields) < 6:
            raise ConverterError(f"Malformed SPW coordinate row at {path}:{line_number}.")
        try:
            chain = _chromosome_to_chain(fields[0])
            start, end = int(fields[1]), int(fields[2])
            coordinate = np.asarray(fields[3:6], dtype=np.float64)
        except ValueError as exc:
            raise ConverterError(f"Malformed SPW coordinate row at {path}:{line_number}.") from exc
        current.append((chain, start, end, coordinate))
    finalize(line_number if "line_number" in locals() else 0)
    if not models:
        raise ConverterError(f"SPW file contains no coordinate rows: {path}")
    first = models[0]
    reference = [(row[0], row[1], row[2]) for row in first]
    coordinates = []
    for frame_id, model in zip(frame_ids, models):
        if [(row[0], row[1], row[2]) for row in model] != reference:
            raise ConverterError(
                f"SPW trace corresponding to frame {frame_id} changes bead/chromosome intervals."
            )
        coordinates.append(np.vstack([row[3] for row in model]))
    override = _sequence_override(types, len(first))
    chain_ids = [row[0] for row in first]
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=frame_ids,
        types=override or ["NA"] * len(first),
        chain_ids=chain_ids,
        chain_indices=_chain_local_indices(chain_ids),
        genomic_intervals=np.asarray([(row[1], row[2]) for row in first]),
        sigma=np.full(len(first), float(sigma)),
        loops=_read_loops(loops),
        metadata={key: value for key, value in metadata.items() if key in {"name", "genome"}},
    )


def spw_to_ndb(
    source,
    output=None,
    *,
    sigma: float = 0.0,
    types=None,
    loops=None,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert SpaceWalk ``sw1`` text to NDB (not binary OpenMiChroM SWB)."""

    source_path = _source_path(source, "spw")
    destination = _destination(source_path, output, "ndb")
    data = _parse_spw(source_path, sigma=sigma, types=types, loops=loops)
    _write_text_files({destination: _ndb_text(data, title=title)}, overwrite=overwrite)
    return destination


def _parse_gro_atom(line: str, *, path: Path, line_number: int):
    try:
        residue_index = int(line[0:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_index = int(line[15:20])
        coordinate = np.array(
            [float(line[20:28]), float(line[28:36]), float(line[36:44])],
            dtype=np.float64,
        )
        if not residue_name or not atom_name:
            raise ValueError("empty fixed-width field")
        return residue_index, residue_name, atom_name, atom_index, coordinate
    except (ValueError, IndexError) as fixed_error:
        fields = line.split()
        try:
            match = re.fullmatch(r"([+-]?\d+)(\D.*)", fields[0])
            if match:
                residue_index = int(match.group(1))
                residue_name = match.group(2)
                atom_name = fields[1]
                atom_index = int(fields[2])
                coordinate = np.asarray(fields[3:6], dtype=np.float64)
            else:
                residue_index = int(fields[0])
                residue_name = fields[1]
                atom_name = fields[2]
                atom_index = int(fields[3])
                coordinate = np.asarray(fields[4:7], dtype=np.float64)
        except (ValueError, IndexError) as exc:
            raise ConverterError(
                f"Malformed GRO atom record at {path}:{line_number}: {line!r}."
            ) from fixed_error
        return residue_index, residue_name, atom_name, atom_index, coordinate


def _resolved_chain_names(count: int, chromosome) -> list[str]:
    if chromosome is None:
        return [f"C{index}" for index in range(1, count + 1)]
    if isinstance(chromosome, str) or not isinstance(chromosome, Sequence):
        if count != 1:
            raise ConverterError(
                "Multiple chains were detected; pass one chromosome identifier per chain."
            )
        return [_chromosome_to_chain(str(chromosome))]
    values = [_chromosome_to_chain(value) for value in chromosome]
    if len(values) != count:
        raise ConverterError(
            f"Detected {count} GRO chains but received {len(values)} chromosome identifiers."
        )
    return values


def _parse_gro(
    path: Path,
    *,
    resolution: int,
    genomic_start: int,
    chromosome,
    sigma: float,
    scale: float,
    types,
    loops,
) -> TrajectoryData:
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    models: list[list[tuple[int, str, str, int, np.ndarray]]] = []
    titles: list[str] = []
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        title = lines[cursor].strip()
        cursor += 1
        if cursor >= len(lines):
            raise ConverterError(f"GRO title {title!r} has no atom-count line.")
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ConverterError(f"Invalid GRO atom count at {path}:{cursor + 1}.") from exc
        cursor += 1
        if atom_count < 1:
            raise ConverterError("A GRO frame must contain at least one atom.")
        if cursor + atom_count >= len(lines):
            raise ConverterError(f"GRO frame {len(models) + 1} is truncated.")
        rows = []
        for offset in range(atom_count):
            line_number = cursor + offset + 1
            parsed = _parse_gro_atom(lines[cursor + offset], path=path, line_number=line_number)
            if parsed[1].upper() == "PL" or parsed[2].upper() == "PL":
                continue
            rows.append(parsed)
        cursor += atom_count
        box_fields = lines[cursor].split()
        cursor += 1
        if len(box_fields) < 3:
            raise ConverterError(f"GRO frame {len(models) + 1} has a malformed box record.")
        try:
            [float(value) for value in box_fields]
        except ValueError as exc:
            raise ConverterError(f"GRO frame {len(models) + 1} has a malformed box record.") from exc
        if not rows:
            raise ConverterError(f"GRO frame {len(models) + 1} contains no chromosome beads.")
        models.append(rows)
        titles.append(title)
    if not models:
        raise ConverterError(f"GRO file contains no coordinate frames: {path}")

    first = models[0]
    chain_numbers = []
    chain_number = 0
    previous_residue = None
    for residue_index, _residue, _atom, _serial, _coordinate in first:
        if previous_residue is None or residue_index <= previous_residue:
            chain_number += 1
        chain_numbers.append(chain_number - 1)
        previous_residue = residue_index
    chain_names = _resolved_chain_names(max(chain_numbers) + 1, chromosome)
    chain_ids = [chain_names[index] for index in chain_numbers]
    chain_indices = _chain_local_indices(chain_ids)
    inferred = [_type_from_atom(row[2], row[1]) for row in first]
    inferred_types = [value for value, _ambiguous in inferred]
    override = _sequence_override(types, len(first))
    if any(ambiguous for _value, ambiguous in inferred) and override is None:
        warnings.warn(
            "A CA/HIS or CA/ARG GRO bead is ambiguous across historical OpenMiChroM "
            "residue mappings; the current mapping was used. Pass types=... to restore "
            "an exact sequence.",
            ConverterWarning,
            stacklevel=2,
        )
    coordinates = []
    first_identity = [(row[0], row[1].upper(), row[2].upper()) for row in first]
    for frame_number, model in enumerate(models, 1):
        if [(row[0], row[1].upper(), row[2].upper()) for row in model] != first_identity:
            raise ConverterError(
                f"GRO frame {frame_number} changes bead count or atom/residue identities."
            )
        coordinates.append(np.vstack([row[4] for row in model]) * float(scale))
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=list(range(1, len(models) + 1)),
        types=override or inferred_types,
        chain_ids=chain_ids,
        chain_indices=chain_indices,
        genomic_intervals=_intervals_by_chain(
            chain_ids, genomic_start=genomic_start, resolution=resolution
        ),
        sigma=np.full(len(first), float(sigma)),
        loops=_read_loops(loops),
        metadata={"title": titles[0]},
    )


def gro_to_ndb(
    source,
    output=None,
    *,
    resolution: int = 50_000,
    genomic_start: int = 1,
    chromosome=None,
    sigma: float = 0.0,
    scale: float = 1.0,
    types=None,
    loops=None,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert one or more concatenated GRO frames to NDB."""

    source_path = _source_path(source, "gro")
    destination = _destination(source_path, output, "ndb")
    if not np.isfinite(scale) or scale <= 0:
        raise ConverterError("scale must be a finite positive number.")
    data = _parse_gro(
        source_path,
        resolution=resolution,
        genomic_start=genomic_start,
        chromosome=chromosome,
        sigma=sigma,
        scale=float(scale),
        types=types,
        loops=loops,
    )
    _write_text_files({destination: _ndb_text(data, title=title)}, overwrite=overwrite)
    return destination


def _parse_bintu_csv(
    path: Path,
    *,
    chromosome,
    genomic_start: int,
    resolution: int,
    sigma: float,
    types,
    loops,
) -> TrajectoryData:
    models: dict[int, list[tuple[int, np.ndarray]]] = {}
    data_started = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 5:
                numeric_prefix = False
                if len(row) >= 2:
                    try:
                        int(row[0].strip())
                        int(row[1].strip())
                    except ValueError:
                        pass
                    else:
                        numeric_prefix = True
                if data_started or numeric_prefix:
                    raise ConverterError(
                        f"Bintu CSV row at {path}:{line_number} has fewer than five columns."
                    )
                continue
            try:
                model_id = int(row[0].strip())
                locus = int(row[1].strip())
            except ValueError as exc:
                if not data_started:
                    continue
                raise ConverterError(f"Malformed Bintu CSV row at {path}:{line_number}.") from exc
            try:
                z_value = float(row[2].strip())
                x_value = float(row[3].strip())
                y_value = float(row[4].strip())
            except ValueError as exc:
                raise ConverterError(f"Malformed Bintu CSV row at {path}:{line_number}.") from exc
            data_started = True
            if locus < 1:
                raise ConverterError(
                    f"Bintu CSV locus indices must be one-based; got {locus} at line {line_number}."
                )
            models.setdefault(model_id, []).append(
                (locus, np.array([x_value, y_value, z_value], dtype=np.float64))
            )
    if not models:
        raise ConverterError(
            f"CSV file contains no Bintu-layout rows (model,index,z,x,y): {path}"
        )
    frame_ids = list(models)
    first = models[frame_ids[0]]
    loci = [row[0] for row in first]
    if len(set(loci)) != len(loci):
        raise ConverterError("The first Bintu CSV model contains duplicate locus indices.")
    coordinates = []
    for frame_id in frame_ids:
        model = models[frame_id]
        if [row[0] for row in model] != loci:
            raise ConverterError(
                f"Bintu CSV model {frame_id} has inconsistent locus indices or ordering."
            )
        coordinates.append(np.vstack([row[1] for row in model]))
    override = _sequence_override(types, len(first))
    chain = _chromosome_to_chain(str(chromosome))
    starts = int(genomic_start) + (np.asarray(loci, dtype=np.int64) - 1) * int(resolution)
    intervals = np.column_stack((starts, starts + int(resolution) - 1))
    return TrajectoryData(
        coordinates=coordinates,
        frame_ids=frame_ids,
        types=override or ["NA"] * len(first),
        chain_ids=[chain] * len(first),
        chain_indices=list(range(1, len(first) + 1)),
        genomic_intervals=intervals,
        sigma=np.full(len(first), float(sigma)),
        loops=_read_loops(loops),
        metadata={"title": f"Bintu CSV chromosome {chromosome}"},
    )


def csv_to_ndb(
    source,
    output=None,
    *,
    chromosome,
    genomic_start: int = 18_000_000,
    resolution: int = 30_000,
    sigma: float = 0.0,
    types=None,
    loops=None,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert the historical Bintu CSV layout ``model,index,z,x,y`` to NDB."""

    if chromosome is None or not str(chromosome).strip():
        raise ConverterError("csv_to_ndb requires an explicit chromosome identifier.")
    source_path = _source_path(source, "csv")
    destination = _destination(source_path, output, "ndb")
    data = _parse_bintu_csv(
        source_path,
        chromosome=chromosome,
        genomic_start=genomic_start,
        resolution=resolution,
        sigma=sigma,
        types=types,
        loops=loops,
    )
    _write_text_files({destination: _ndb_text(data, title=title)}, overwrite=overwrite)
    return destination


_CONVERTERS = {
    ("ndb", "cndb"): ndb_to_cndb,
    ("cndb", "ndb"): cndb_to_ndb,
    ("ndb", "pdb"): ndb_to_pdb,
    ("pdb", "ndb"): pdb_to_ndb,
    ("ndb", "spw"): ndb_to_spw,
    ("spw", "ndb"): spw_to_ndb,
    ("gro", "ndb"): gro_to_ndb,
    ("csv", "ndb"): csv_to_ndb,
}


def convert(
    source,
    output=None,
    *,
    input_format: str | None = None,
    output_format: str | None = None,
    from_format: str | None = None,
    to_format: str | None = None,
    overwrite: bool = False,
    **options,
) -> Path:
    """Convert a local trajectory through one of the eight supported routes.

    Formats are inferred from suffixes when possible.  ``from_format`` and
    ``to_format`` are readable aliases for ``input_format``/``output_format``.
    Route-specific options are passed to the explicit converter function.
    """

    if input_format is not None and from_format is not None:
        if _normalize_format(input_format) != _normalize_format(from_format):
            raise ConverterError("input_format and from_format disagree.")
    if output_format is not None and to_format is not None:
        if _normalize_format(output_format) != _normalize_format(to_format):
            raise ConverterError("output_format and to_format disagree.")
    source_path = _source_path(source)
    inferred_input = source_path.suffix.lower().lstrip(".")
    selected_input = _normalize_format(input_format or from_format or inferred_input)
    inferred_output = None
    if output is not None:
        inferred_output = Path(output).suffix.lower().lstrip(".") or None
    selected_output = _normalize_format(output_format or to_format or inferred_output)
    if selected_output is None:
        raise ConverterError(
            "Output format is required when output has no recognized suffix; pass output_format."
        )
    route = (selected_input, selected_output)
    if route not in SUPPORTED_CONVERSIONS:
        readable = ", ".join(f"{source_fmt}->{target_fmt}" for source_fmt, target_fmt in sorted(SUPPORTED_CONVERSIONS))
        raise ConverterError(
            f"Unsupported conversion {selected_input}->{selected_output}. Supported routes: {readable}."
        )
    if "coordinate_scale" in options:
        if "scale" in options:
            raise TypeError("Pass only one of scale and coordinate_scale.")
        options["scale"] = options.pop("coordinate_scale")
    if "chromosome_id" in options:
        if "chromosome" in options:
            raise TypeError("Pass only one of chromosome and chromosome_id.")
        options["chromosome"] = options.pop("chromosome_id")
    return _CONVERTERS[route](
        source_path, output, overwrite=overwrite, **options
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``python -m OpenMiChroM.Converters``."""

    parser = argparse.ArgumentParser(description="Convert OpenMiChroM/NDB trajectory files.")
    parser.add_argument("source")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--from-format", dest="input_format")
    parser.add_argument("--to-format", dest="output_format")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--genomic-start", type=int)
    parser.add_argument("--chromosome")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--types")
    parser.add_argument("--loops")
    parser.add_argument("--title")
    parser.add_argument("--name")
    parser.add_argument("--genome")
    arguments = parser.parse_args(argv)
    options = {
        key: value
        for key, value in vars(arguments).items()
        if key
        not in {"source", "output", "input_format", "output_format", "overwrite"}
        and value is not None
    }
    destination = convert(
        arguments.source,
        arguments.output,
        input_format=arguments.input_format,
        output_format=arguments.output_format,
        overwrite=arguments.overwrite,
        **options,
    )
    print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public API
    raise SystemExit(main())
