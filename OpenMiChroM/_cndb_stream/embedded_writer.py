"""Embedded HDF5 object-index writer for OpenMiChroM CNDB files.

This module reimplements the small writer-side behavior used by
``hdf5-indexer``: walk an HDF5 file with pyfive, record object-header offsets
for group children, store the resulting JSON as a gzip-compressed opaque
dataset named ``/_index``, and store that dataset's object-header offset in
the root attribute ``_index_offset``.
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .embedded import load_pyfive_file_class

INDEX_DATASET_NAME = "_index"
INDEX_OFFSET_ATTR = "_index_offset"
INDEX_FORMAT_NAME = "hdf5-object-offset-json-gzip"
CNDB_FORMAT_NAME = "OpenMiChroM-CNDB"
CNDB_FORMAT_VERSION = "2.0"


def initialize_cndb_header(
    h5: h5py.File,
    *,
    n_beads: int,
    coordinate_units: str = "nanometer",
    coordinate_dtype: str | None = None,
    frame_layout: str = "root_numeric_frames",
    indexed: bool = True,
    notes: str | None = None,
) -> h5py.Group:
    """Create or update the OpenMiChroM CNDB metadata header."""

    header = h5.require_group("Header")
    header.attrs["format_name"] = CNDB_FORMAT_NAME
    header.attrs["format_version"] = CNDB_FORMAT_VERSION
    header.attrs["creator"] = "OpenMiChroM"
    header.attrs["openmichrom_version"] = _openmichrom_version()
    header.attrs["creation_time"] = datetime.now(timezone.utc).isoformat()
    header.attrs["coordinate_units"] = coordinate_units
    header.attrs["coordinate_dtype"] = coordinate_dtype or ""
    header.attrs["frame_layout"] = frame_layout
    header.attrs["n_beads"] = int(n_beads)
    header.attrs["n_frames"] = 0
    header.attrs["indexed"] = bool(indexed)
    header.attrs["index_format"] = INDEX_FORMAT_NAME if indexed else ""
    header.attrs["index_dataset"] = f"/{INDEX_DATASET_NAME}" if indexed else ""
    header.attrs["notes"] = notes or ""
    return header


def finalize_cndb_header(
    h5: h5py.File,
    *,
    n_frames: int,
    n_beads: int | None = None,
    coordinate_dtype: str | None = None,
    indexed: bool | None = None,
    index_summary: dict[str, Any] | None = None,
) -> None:
    """Update final CNDB metadata after all frame datasets are written."""

    if "Header" not in h5:
        return
    header = h5["Header"]
    header.attrs["n_frames"] = int(n_frames)
    if n_beads is not None:
        header.attrs["n_beads"] = int(n_beads)
    if coordinate_dtype:
        header.attrs["coordinate_dtype"] = str(coordinate_dtype)
    if indexed is not None:
        header.attrs["indexed"] = bool(indexed)
        header.attrs["index_format"] = INDEX_FORMAT_NAME if indexed else ""
        header.attrs["index_dataset"] = f"/{INDEX_DATASET_NAME}" if indexed else ""
    if index_summary:
        header.attrs["index_compressed_nbytes"] = int(index_summary["compressed_nbytes"])
        header.attrs["index_object_count"] = int(index_summary["object_count"])


def write_embedded_index(
    h5_path: str | Path,
    *,
    dataset_name: str = INDEX_DATASET_NAME,
    offset_attr: str = INDEX_OFFSET_ATTR,
) -> dict[str, Any]:
    """Append a gzip JSON object-offset index to an HDF5/CNDB file.

    The generated JSON intentionally excludes the ``/_index`` dataset itself,
    matching the hdf5-indexer convention. The root ``_index_offset`` attribute
    provides the object-header offset needed to find ``/_index`` remotely.
    """

    h5_path = Path(h5_path)
    _remove_existing_index(h5_path, dataset_name=dataset_name, offset_attr=offset_attr)

    object_index = build_object_offset_index(h5_path, exclude_root_names={dataset_name})
    index_json = json.dumps(object_index, separators=(",", ":"), sort_keys=True).encode("utf-8")
    compressed = gzip.compress(index_json)

    with h5py.File(h5_path, "r+") as h5:
        opaque_dtype = h5py.opaque_dtype(np.dtype(f"V{len(compressed)}"))
        dataset = h5.create_dataset(dataset_name, shape=(1,), dtype=opaque_dtype)
        dataset[0] = np.void(compressed)
        h5.flush()

    object_offset = object_header_offset(h5_path, f"/{dataset_name}")
    with h5py.File(h5_path, "r+") as h5:
        if offset_attr in h5.attrs:
            del h5.attrs[offset_attr]
        h5.attrs.create(offset_attr, int(object_offset))
        summary = {
            "dataset_path": f"/{dataset_name}",
            "object_offset": int(object_offset),
            "compressed_nbytes": int(len(compressed)),
            "object_count": int(len(object_index)),
        }
        finalize_cndb_header(h5, n_frames=_count_root_numeric_frames(h5), indexed=True, index_summary=summary)
        h5.flush()
    return summary


def build_object_offset_index(
    h5_path: str | Path,
    *,
    exclude_root_names: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Build a group-to-child-object-offset map for an HDF5 file."""

    file_cls = load_pyfive_file_class()
    h5 = file_cls(str(h5_path), index={"__cndb_stream_no_embedded_index__": {}})
    try:
        object_index: dict[str, dict[str, int]] = {}
        _index_children(h5, object_index, exclude_root_names=exclude_root_names or set())
        return object_index
    finally:
        h5.close()


def object_header_offset(h5_path: str | Path, path: str) -> int:
    """Return the HDF5 object-header offset for ``path``."""

    file_cls = load_pyfive_file_class()
    h5 = file_cls(str(h5_path), index={"__cndb_stream_no_embedded_index__": {}})
    try:
        obj = h5[path]
        return int(obj._dataobjects.offset)
    finally:
        h5.close()


def _remove_existing_index(h5_path: Path, *, dataset_name: str, offset_attr: str) -> None:
    with h5py.File(h5_path, "r+") as h5:
        if dataset_name in h5:
            del h5[dataset_name]
        if offset_attr in h5.attrs:
            del h5.attrs[offset_attr]
        if "Header" in h5:
            finalize_cndb_header(h5, n_frames=_count_root_numeric_frames(h5), indexed=False)
        h5.flush()


def _index_children(
    group: Any,
    object_index: dict[str, dict[str, int]],
    *,
    exclude_root_names: set[str],
) -> None:
    children: dict[str, int] = {}
    for key in group.keys():
        if group.name == "/" and key in exclude_root_names:
            continue
        child = group.get(key)
        if child is None or not hasattr(child, "_dataobjects"):
            continue
        children[str(key)] = int(child._dataobjects.offset)
        if hasattr(child, "keys"):
            _index_children(child, object_index, exclude_root_names=exclude_root_names)
    object_index[group.name] = children


def _count_root_numeric_frames(h5: h5py.File) -> int:
    return sum(
        1
        for name, obj in h5.items()
        if str(name).isdigit()
        and isinstance(obj, h5py.Dataset)
        and len(obj.shape) == 2
        and int(obj.shape[1]) == 3
    )


def _openmichrom_version() -> str:
    package = sys.modules.get("OpenMiChroM")
    version = getattr(package, "__version__", None)
    return str(version) if version is not None else ""
