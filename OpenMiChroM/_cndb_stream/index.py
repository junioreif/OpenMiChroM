"""Build enhanced JSON indexes for OpenMiChroM CNDB files."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

from .filters import hdf5_filter_pipeline_supported
from .utils import json_safe_value, sort_frame_ids

INDEX_VERSION = "0.1"
NESTED_INDEX_VERSION = "0.2"
INDEX_FORMAT = "cndb-stream-index"
OPENMICHROM_SIMPLE_LAYOUT = "openmichrom-simple"
NESTED_NDB_LAYOUT = "nested-ndb"


def build_index(
    h5_path: str | Path,
    output_index_path: str | Path | None = None,
    trajectories: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build an enhanced CNDB index for a local HDF5/CNDB file.

    The MVP supports simple OpenMiChroM files with root-level numeric frame
    datasets and nested NDB/SWB-style files with frame datasets under
    ``/<trajectory>/spatial_position/<frame>``. Chunked or compressed frame
    datasets are included in the index but marked unsupported for direct
    partial reads.
    """

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        simple_index = _build_simple_index(h5)
        if simple_index["n_frames"] > 0 and trajectories is None:
            index = simple_index
        else:
            index = _build_nested_index(h5, trajectories=trajectories)

    if output_index_path is not None:
        output_index_path = Path(output_index_path)
        with output_index_path.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")

    return index


def _build_simple_index(h5: h5py.File) -> dict[str, Any]:
    frames: dict[str, dict[str, Any]] = {}
    frame_ids: list[str] = []
    types_path: str | None = None
    types: list[Any] | None = None

    if "types" in h5 and isinstance(h5["types"], h5py.Dataset):
        types_path = "/types"
        types = _read_types(h5["types"])

    for name, obj in h5.items():
        if not _is_simple_frame_dataset(name, obj):
            continue
        frame_id = str(name)
        frame_ids.append(frame_id)
        frames[frame_id] = _dataset_index_entry(obj, frame_id)

    frame_ids = sort_frame_ids(frame_ids)
    n_frames = len(frame_ids)
    n_beads = int(frames[frame_ids[0]]["shape"][0]) if frame_ids else 0

    index: dict[str, Any] = {
        "version": INDEX_VERSION,
        "format": INDEX_FORMAT,
        "layout": OPENMICHROM_SIMPLE_LAYOUT,
        "n_frames": n_frames,
        "n_beads": n_beads,
        "frame_ids": frame_ids,
        "types_path": types_path,
        "frames": {frame_id: frames[frame_id] for frame_id in frame_ids},
    }
    if types is not None:
        index["types"] = types
    return index


def _build_nested_index(
    h5: h5py.File,
    *,
    trajectories: Sequence[str] | None = None,
) -> dict[str, Any]:
    requested = set(trajectories or [])
    trajectory_indexes: dict[str, dict[str, Any]] = {}

    for name, obj in h5.items():
        if requested and name not in requested:
            continue
        if not _is_nested_trajectory_group(obj):
            continue
        trajectory_indexes[name] = _trajectory_index_entry(obj)

    missing = requested.difference(trajectory_indexes)
    if missing:
        available = sorted(
            name for name, obj in h5.items() if _is_nested_trajectory_group(obj)
        )
        raise ValueError(
            f"Requested trajectories were not found or not supported: {sorted(missing)}. "
            f"Available nested trajectories include: {available[:20]}"
        )

    return {
        "version": NESTED_INDEX_VERSION,
        "format": INDEX_FORMAT,
        "layout": NESTED_NDB_LAYOUT,
        "trajectories": trajectory_indexes,
    }


def _is_simple_frame_dataset(name: str, obj: h5py.Dataset | h5py.Group) -> bool:
    if not isinstance(obj, h5py.Dataset):
        return False
    if not _is_frame_id(name):
        return False
    if len(obj.shape) != 2:
        return False
    return int(obj.shape[1]) == 3


def _is_frame_id(name: str) -> bool:
    text = str(name)
    return text.isdigit() or (text.startswith("t_") and text[2:].isdigit())


def _is_nested_trajectory_group(obj: h5py.Dataset | h5py.Group) -> bool:
    if not isinstance(obj, h5py.Group):
        return False
    if "spatial_position" not in obj:
        return False
    spatial_position = obj["spatial_position"]
    return isinstance(spatial_position, h5py.Group)


def _trajectory_index_entry(group: h5py.Group) -> dict[str, Any]:
    spatial_position = group["spatial_position"]
    frames: dict[str, dict[str, Any]] = {}
    frame_ids: list[str] = []
    for frame_id, obj in spatial_position.items():
        if not _is_simple_frame_dataset(frame_id, obj):
            continue
        frame_ids.append(frame_id)
        frames[frame_id] = _dataset_index_entry(obj, frame_id)

    frame_ids = sort_frame_ids(frame_ids)
    n_frames = len(frame_ids)
    n_beads = int(frames[frame_ids[0]]["shape"][0]) if frame_ids else 0
    types_path = _dataset_path_or_none(group, "types")
    types = _read_types(group["types"]) if types_path is not None else None

    entry: dict[str, Any] = {
        "types_path": types_path,
        "genomic_position_path": _dataset_path_or_none(group, "genomic_position"),
        "time_path": _dataset_path_or_none(group, "time"),
        "spatial_position_path": spatial_position.name,
        "n_frames": n_frames,
        "n_beads": n_beads,
        "frame_ids": frame_ids,
        "frames": {frame_id: frames[frame_id] for frame_id in frame_ids},
    }
    if types is not None:
        entry["types"] = types
    return entry


def _dataset_path_or_none(group: h5py.Group, name: str) -> str | None:
    if name not in group:
        return None
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        return None
    return obj.name


def _dataset_index_entry(dataset: h5py.Dataset, frame_id: str) -> dict[str, Any]:
    dtype = np.dtype(dataset.dtype)
    shape = [int(dim) for dim in dataset.shape]
    compression = dataset.compression
    chunks = [int(dim) for dim in dataset.chunks] if dataset.chunks is not None else None
    filters = _dataset_filters(dataset)
    data_offset = dataset.id.get_offset()
    storage_size = int(dataset.id.get_storage_size())
    nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)

    layout = "contiguous" if chunks is None else "chunked"
    direct_read_supported = (
        layout == "contiguous"
        and compression is None
        and data_offset is not None
        and len(shape) == 2
        and shape[1] == 3
    )

    return {
        "path": dataset.name,
        "frame_id": frame_id,
        "shape": shape,
        "dtype": dtype.name,
        "layout": layout,
        "compression": compression,
        "filters": filters,
        "chunks": chunks,
        "data_offset": int(data_offset) if data_offset is not None else None,
        "storage_size": storage_size,
        "nbytes": nbytes,
        "direct_read_supported": direct_read_supported,
        "chunked_read_supported": bool(
            layout == "chunked"
            and len(shape) == 2
            and shape[1] == 3
            and hdf5_filter_pipeline_supported(filters)
        ),
    }


def _read_types(dataset: h5py.Dataset) -> list[Any]:
    values = dataset[()]
    safe_values = json_safe_value(values)
    if isinstance(safe_values, list):
        return safe_values
    return [safe_values]


def _dataset_filters(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    plist = dataset.id.get_create_plist()
    for index in range(plist.get_nfilters()):
        try:
            filter_id, flags, cd_values, name = plist.get_filter(index)
        except ValueError:
            continue
        filters.append(
            {
                "id": int(filter_id),
                "flags": int(flags),
                "client_data": [int(value) for value in cd_values],
                "name": name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name),
            }
        )
    return filters
