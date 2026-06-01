"""Embedded HDF5 object-index support for remote CNDB files."""

from __future__ import annotations

import gzip
import json
import posixpath
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import CNDBIndexError
from .index import INDEX_FORMAT, NESTED_NDB_LAYOUT, OPENMICHROM_SIMPLE_LAYOUT
from .remote import RemoteByteReader
from .utils import coerce_frame_id, sort_frame_ids

EMBEDDED_INDEX_CACHE_FORMAT = "cndb-stream-embedded-index-cache"
EMBEDDED_INDEX_CACHE_VERSION = "0.1"


class StrictRangeFile:
    """Small seekable file-like wrapper backed by strict HTTP Range reads."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self.name = url
        self._reader = RemoteByteReader(url, timeout=timeout)
        self._pos = 0
        self.closed = False

    @property
    def bytes_read(self) -> int:
        return self._reader.bytes_read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        else:
            raise NotImplementedError("Seek from end is not supported for remote CNDB files.")
        if self._pos < 0:
            raise ValueError("File position cannot be negative.")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise NotImplementedError("Read-all is intentionally disabled for remote CNDB files.")
        start = self._pos
        stop = start + int(size)
        data = self._reader.read_range(start, stop)
        self._pos = stop
        return data

    def close(self) -> None:
        self.closed = True


class BufferedStrictRangeFile:
    """A tiny read-through cache for pyfive metadata reads."""

    def __init__(
        self,
        file_reader: StrictRangeFile,
        *,
        fetch_size: int = 8192,
        max_size: int = 1024 * 1024,
    ) -> None:
        self.file_reader = file_reader
        self.fetch_size = int(fetch_size)
        self.max_size = int(max_size)
        self._pos = 0
        self._chunks: list[tuple[int, bytes]] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self.file_reader.name

    @property
    def bytes_read(self) -> int:
        return self.file_reader.bytes_read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = int(offset)
        elif whence == 1:
            self._pos += int(offset)
        else:
            raise NotImplementedError("Seek from end is not supported for remote CNDB files.")
        if self._pos < 0:
            raise ValueError("File position cannot be negative.")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise NotImplementedError("Read-all is intentionally disabled for remote CNDB files.")
        start = self._pos
        if int(size) > self.max_size:
            self.file_reader.seek(start)
            data = self.file_reader.read(int(size))
            self._pos += int(size)
            return data

        data = self._read_from_cache(start, int(size))
        if data is None:
            fetch_size = max(int(size), self.fetch_size)
            self.file_reader.seek(start)
            fetched = self.file_reader.read(fetch_size)
            self._add_to_cache(start, fetched)
            data = self._read_from_cache(start, int(size))
        if data is None:
            raise OSError(f"Could not read {size} bytes at offset {start}.")
        self._pos += int(size)
        return data

    def close(self) -> None:
        self.closed = True
        self.file_reader.close()

    def _read_from_cache(self, start: int, size: int) -> bytes | None:
        stop = start + size
        for chunk_start, chunk in self._chunks:
            chunk_stop = chunk_start + len(chunk)
            if chunk_start <= start and chunk_stop >= stop:
                local_start = start - chunk_start
                return chunk[local_start : local_start + size]
        return None

    def _add_to_cache(self, start: int, data: bytes) -> None:
        self._chunks.append((start, data))
        total = sum(len(chunk) for _, chunk in self._chunks)
        while total > self.max_size and self._chunks:
            _, removed = self._chunks.pop(0)
            total -= len(removed)


class EmbeddedIndexProvider:
    """Load and query an embedded hdf5-indexer object index."""

    def __init__(
        self,
        *,
        h5_url: str,
        trajectory: str | None = None,
        timeout: float = 30.0,
        fetch_size: int = 8192,
        cache_size: int = 1024 * 1024,
        index_cache_path: str | Path | None = None,
        use_index_cache: bool = True,
    ) -> None:
        file_cls = load_pyfive_file_class()
        raw_file = StrictRangeFile(h5_url, timeout=timeout)
        self._range_file = BufferedStrictRangeFile(
            raw_file,
            fetch_size=fetch_size,
            max_size=cache_size,
        )
        self.index_cache_path = str(index_cache_path) if index_cache_path is not None else None
        self.index_cache_hit = False
        if index_cache_path is not None and use_index_cache and Path(index_cache_path).exists():
            embedded = load_embedded_index_cache(Path(index_cache_path), h5_url=h5_url)
            self.index_cache_hit = True
            self._index_bytes_read = 0
        else:
            probe_h5 = file_cls(self._range_file, index={"__dummy__": {}})
            embedded = read_embedded_object_index(probe_h5, self._range_file)
            self._index_bytes_read = int(embedded["compressed_nbytes"])
            if index_cache_path is not None and use_index_cache:
                write_embedded_index_cache(Path(index_cache_path), h5_url=h5_url, embedded=embedded)
        self._h5 = file_cls(self._range_file, index=embedded["index"])
        self.h5_url = h5_url
        self.object_index: dict[str, dict[str, int]] = embedded["index"]
        self._dataset_data_bytes_read = 0
        if not self.object_index:
            raise CNDBIndexError(
                "No embedded HDF5 object index was found. Expected a gzip-compressed "
                "'_index' dataset and/or '_index_offset' root attribute."
            )

        self.embedded_index_offset = embedded["object_offset"]
        self.embedded_index_object_offset = self.object_index.get("/", {}).get("_index")
        if self.embedded_index_object_offset is None:
            self.embedded_index_object_offset = self.embedded_index_offset
        self.embedded_index_data_offset = embedded["data_offset"]
        self.embedded_index_compressed_nbytes = embedded["compressed_nbytes"]
        self.current_trajectory = self._resolve_trajectory(trajectory)
        self._frame_info_cache: dict[str, dict[str, Any]] = {}
        self._dataset_value_cache: dict[str, Any] = {}

    @property
    def index_bytes_read(self) -> int:
        """Compressed ``/_index`` payload bytes read from the remote file."""

        return int(self._index_bytes_read)

    @property
    def metadata_bytes_read(self) -> int:
        """Remote HDF5 metadata bytes read, excluding the embedded index payload.

        This includes object headers, superblock/root metadata, and any small
        cache overfetch performed by the embedded backend while inspecting HDF5
        objects.
        """

        return max(
            0,
            int(self._range_file.bytes_read)
            - self.index_bytes_read
            - self.data_bytes_read,
        )

    @property
    def data_bytes_read(self) -> int:
        """Coordinate/chunk payload bytes read through the embedded backend."""

        return int(self._dataset_data_bytes_read)

    @property
    def bytes_read(self) -> int:
        """Total remote bytes read by the embedded-index metadata backend."""

        return int(self._range_file.bytes_read)

    @property
    def metadata_cache_size(self) -> int:
        """Number of frame metadata records resolved for the selected trajectory."""

        return len(self._frame_info_cache)

    @property
    def trajectories(self) -> list[str]:
        """Nested trajectory groups discovered from the embedded object index."""

        root_children = self.object_index.get("/", {})
        names = []
        for name in root_children:
            group_path = f"/{name}"
            spatial_path = f"{group_path}/spatial_position"
            if spatial_path in self.object_index:
                names.append(name)
        return sorted(names)

    @property
    def frame_ids(self) -> list[str]:
        """Frame ids for the selected layout."""

        if self.current_trajectory is None:
            root_children = self.object_index.get("/", {})
            return sort_frame_ids([name for name in root_children if _is_frame_id(name)])
        spatial_path = self._spatial_position_path(self.current_trajectory)
        return sort_frame_ids(
            [name for name in self.object_index.get(spatial_path, {}) if _is_frame_id(name)]
        )

    def to_lazy_cndb_index(self) -> dict[str, Any]:
        """Create an index shell backed by lazy embedded metadata."""

        frame_ids = self.frame_ids
        if not frame_ids:
            raise CNDBIndexError("Embedded object index did not contain numeric frame datasets.")
        first_frame = self.frame_info(frame_ids[0])
        if self.current_trajectory is None:
            return {
                "version": "0.1",
                "format": INDEX_FORMAT,
                "layout": OPENMICHROM_SIMPLE_LAYOUT,
                "n_frames": len(frame_ids),
                "n_beads": int(first_frame["shape"][0]),
                "frame_ids": frame_ids,
                "types_path": "/types" if self.path_is_indexed("/types") else None,
                "frames": {first_frame["frame_id"]: first_frame},
                "embedded_index": self._embedded_index_summary(),
            }

        entry = {
            "types_path": self._dataset_path_if_indexed(f"/{self.current_trajectory}/types"),
            "genomic_position_path": self._dataset_path_if_indexed(
                f"/{self.current_trajectory}/genomic_position"
            ),
            "time_path": self._dataset_path_if_indexed(f"/{self.current_trajectory}/time"),
            "spatial_position_path": self._spatial_position_path(self.current_trajectory),
            "n_frames": len(frame_ids),
            "n_beads": int(first_frame["shape"][0]),
            "frame_ids": frame_ids,
            "frames": {first_frame["frame_id"]: first_frame},
        }
        return {
            "version": "0.2",
            "format": INDEX_FORMAT,
            "layout": NESTED_NDB_LAYOUT,
            "available_trajectories": self.trajectories,
            "trajectories": {self.current_trajectory: entry},
            "embedded_index": self._embedded_index_summary(),
        }

    def frame_info(self, frame: int | str) -> dict[str, Any]:
        """Return frame metadata, parsing the frame object header on demand."""

        frame_id = coerce_frame_id(frame)
        if frame_id in self._frame_info_cache:
            return self._frame_info_cache[frame_id]
        if frame_id not in self.frame_ids:
            raise CNDBIndexError(
                f"Frame {frame_id!r} was not found in embedded index. "
                f"Available frames include: {self.frame_ids[:10]}"
            )

        path = self._frame_path(frame_id)
        dataset = self._h5[path]
        dtype = np.dtype(dataset.dtype)
        shape = [int(dim) for dim in dataset.shape]
        chunks = [int(dim) for dim in dataset.chunks] if dataset.chunks is not None else None
        compression = dataset.compression
        filters = _json_safe_filter_pipeline(getattr(dataset, "filter_pipeline", None))
        layout = _layout_name(dataset.id.layout_class)
        data_offset = getattr(dataset.id, "data_offset", None)
        nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
        direct_read_supported = (
            layout == "contiguous"
            and compression is None
            and data_offset is not None
            and len(shape) == 2
            and shape[1] == 3
        )

        info = {
            "path": path,
            "frame_id": frame_id,
            "shape": shape,
            "dtype": dtype.name,
            "layout": layout,
            "compression": compression,
            "filters": filters,
            "chunks": chunks,
            "data_offset": int(data_offset) if data_offset is not None else None,
            "storage_size": nbytes if direct_read_supported else None,
            "nbytes": nbytes,
            "direct_read_supported": direct_read_supported,
            "chunked_read_supported": bool(layout == "chunked" and len(shape) == 2 and shape[1] == 3),
            "metadata_source": "embedded-index",
        }
        self._frame_info_cache[frame_id] = info
        return info

    def read_dataset_rows(self, path: str, start: int, stop: int) -> np.ndarray:
        """Read rows from a dataset through pyfive and count remote payload bytes."""

        before = int(self._range_file.bytes_read)
        dataset = self._h5[path]
        try:
            values = np.asarray(dataset[start:stop, :])
        except Exception as exc:
            raise CNDBIndexError(
                f"Could not stream chunked dataset rows for {path!r}: {exc}"
            ) from exc
        after = int(self._range_file.bytes_read)
        self._dataset_data_bytes_read += max(0, after - before)
        return values

    def embedded_index_dataset_info(self) -> dict[str, Any] | None:
        """Return metadata for the embedded ``/_index`` dataset when present."""

        if self.embedded_index_compressed_nbytes is not None:
            return {
                "path": "/_index",
                "object_offset": self.embedded_index_object_offset,
                "root_attribute_offset": self.embedded_index_offset,
                "data_offset": self.embedded_index_data_offset,
                "shape": [1],
                "dtype": f"opaque[{self.embedded_index_compressed_nbytes}]",
                "nbytes": self.embedded_index_compressed_nbytes,
            }

        if self.path_is_indexed("/_index"):
            dataset = self._h5["/_index"]
            object_offset = self.embedded_index_object_offset
            dtype = np.dtype(dataset.dtype)
            shape = [int(dim) for dim in dataset.shape]
        elif self.embedded_index_offset is not None:
            DataObjects, DatasetID = load_pyfive_dataset_backend()
            dataobjects = DataObjects(self._range_file, self.embedded_index_offset)
            dataset_id = DatasetID(dataobjects, noindex=True)
            object_offset = self.embedded_index_offset
            dtype = np.dtype(dataset_id.dtype)
            shape = [int(dim) for dim in dataset_id.shape]
        else:
            return None

        return {
            "path": "/_index",
            "object_offset": object_offset,
            "root_attribute_offset": self.embedded_index_offset,
            "shape": shape,
            "dtype": str(dtype),
            "nbytes": self.embedded_index_compressed_nbytes
            or int(np.prod(shape, dtype=np.int64) * dtype.itemsize),
        }

    def close(self) -> None:
        self._h5.close()
        self._range_file.close()

    def path_is_indexed(self, path: str) -> bool:
        """Return whether a path is represented in the embedded object index."""

        normalized = posixpath.normpath(path)
        if normalized in self.object_index:
            return True
        parent = posixpath.dirname(normalized) or "/"
        name = posixpath.basename(normalized)
        return name in self.object_index.get(parent, {})

    def read_dataset(self, path: str) -> Any:
        """Read a small indexed metadata dataset by path.

        This is intended for metadata such as ``types`` or
        ``genomic_position``. Coordinate frame payloads should continue through
        the exact byte-range coordinate reader.
        """

        normalized = posixpath.normpath(path)
        if normalized in self._dataset_value_cache:
            return self._dataset_value_cache[normalized]
        if not self.path_is_indexed(normalized):
            raise CNDBIndexError(f"Path {normalized!r} is not present in the embedded index.")
        dataset = self._h5[normalized]
        value = dataset[()]
        self._dataset_value_cache[normalized] = value
        return value

    def _resolve_trajectory(self, trajectory: str | None) -> str | None:
        trajectories = self.trajectories
        if not trajectories:
            if trajectory is not None:
                raise CNDBIndexError(
                    "A trajectory was provided, but the embedded index appears to use "
                    "a root-level simple layout."
                )
            return None
        if trajectory is not None:
            if trajectory not in trajectories:
                raise CNDBIndexError(
                    f"Trajectory {trajectory!r} was not found in embedded index. "
                    f"Available trajectories include: {trajectories[:25]}"
                )
            return trajectory
        if len(trajectories) == 1:
            return trajectories[0]
        raise CNDBIndexError(
            "Embedded index contains multiple trajectories. Specify one with "
            f"trajectory=... Available trajectories include: {trajectories[:25]}"
        )

    @staticmethod
    def _spatial_position_path(trajectory: str) -> str:
        return f"/{trajectory}/spatial_position"

    def _frame_path(self, frame_id: str) -> str:
        if self.current_trajectory is None:
            return f"/{frame_id}"
        return f"{self._spatial_position_path(self.current_trajectory)}/{frame_id}"

    def _dataset_path_if_indexed(self, path: str) -> str | None:
        return path if self.path_is_indexed(path) else None

    def _embedded_index_summary(self) -> dict[str, Any]:
        dataset_info = self.embedded_index_dataset_info()
        return {
            "format": "hdf5-indexer-object-offsets",
            "dataset_path": "/_index" if dataset_info else None,
            "root_attribute": "_index_offset",
            "root_attribute_offset": self.embedded_index_offset,
            "object_offset": self.embedded_index_object_offset,
            "compressed_nbytes": dataset_info["nbytes"] if dataset_info else None,
            "object_count": len(self.object_index),
        }


def load_pyfive_file_class():
    """Load the vendored hdf5-indexed-reader pyfive backend."""

    from ._vendor.hdf5_indexed_reader.pyfive.high_level import File

    return File


def load_pyfive_dataset_backend():
    """Load low-level pyfive dataset metadata classes."""

    load_pyfive_file_class()
    from ._vendor.hdf5_indexed_reader.pyfive.dataobjects import DataObjects
    from ._vendor.hdf5_indexed_reader.pyfive.h5d import DatasetID

    return DataObjects, DatasetID


def read_embedded_object_index(h5_probe: Any, range_file: BufferedStrictRangeFile) -> dict[str, Any]:
    """Read the embedded gzip JSON object-offset index by byte ranges."""

    object_offset = _json_int(h5_probe.attrs.get("_index_offset"))
    if object_offset is None:
        object_offset = h5_probe._links.get("_index")
    if object_offset is None:
        raise CNDBIndexError(
            "Remote file does not expose an embedded '_index_offset' attribute or '_index' link."
        )

    DataObjects, _ = load_pyfive_dataset_backend()
    dataobjects = DataObjects(range_file, object_offset)
    if not dataobjects.is_dataset:
        raise CNDBIndexError(f"Embedded index object at offset {object_offset} is not a dataset.")

    data_offset, data_size = _contiguous_payload_span(dataobjects)
    range_file.seek(data_offset)
    index_bytes = range_file.read(data_size)
    if index_bytes[:2] == b"\x1f\x8b":
        index_bytes = gzip.decompress(index_bytes)
    object_index = json.loads(index_bytes.decode("utf-8"))
    return {
        "index": object_index,
        "object_offset": int(object_offset),
        "data_offset": int(data_offset),
        "compressed_nbytes": int(data_size),
    }


def load_embedded_index_cache(path: Path, *, h5_url: str) -> dict[str, Any]:
    """Load a parsed embedded object-offset index from a local JSON.gz cache."""

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("format") != EMBEDDED_INDEX_CACHE_FORMAT:
        raise CNDBIndexError(
            f"Unsupported embedded index cache format in {path}: {payload.get('format')!r}."
        )
    if payload.get("version") != EMBEDDED_INDEX_CACHE_VERSION:
        raise CNDBIndexError(
            f"Unsupported embedded index cache version in {path}: {payload.get('version')!r}."
        )
    cached_url = payload.get("h5_url")
    if cached_url and cached_url != h5_url:
        raise CNDBIndexError(
            f"Embedded index cache {path} was created for {cached_url!r}, not {h5_url!r}."
        )

    object_index = payload.get("index")
    if not isinstance(object_index, dict) or not object_index:
        raise CNDBIndexError(f"Embedded index cache {path} does not contain an object index.")

    return {
        "index": object_index,
        "object_offset": int(payload["object_offset"]),
        "data_offset": int(payload["index_data_offset"]),
        "compressed_nbytes": int(payload["compressed_nbytes"]),
    }


def write_embedded_index_cache(
    path: Path,
    *,
    h5_url: str,
    embedded: dict[str, Any],
) -> None:
    """Write the parsed embedded object-offset index to a local JSON.gz cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": EMBEDDED_INDEX_CACHE_FORMAT,
        "version": EMBEDDED_INDEX_CACHE_VERSION,
        "h5_url": h5_url,
        "object_offset": int(embedded["object_offset"]),
        "index_data_offset": int(embedded["data_offset"]),
        "compressed_nbytes": int(embedded["compressed_nbytes"]),
        "index": embedded["index"],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    tmp_path.replace(path)


def _contiguous_payload_span(dataobjects: Any) -> tuple[int, int]:
    """Return ``(data_offset, storage_size)`` from a contiguous dataset layout message."""

    layout_messages = [msg for msg in dataobjects.msgs if msg["type"] == 8]
    if not layout_messages:
        raise CNDBIndexError("Embedded index dataset is missing an HDF5 layout message.")
    msg = layout_messages[0]
    payload = dataobjects.msg_data[msg["offset_to_message"] : msg["offset_to_message"] + msg["size"]]
    version = payload[0]
    layout_class = payload[1]
    if layout_class != 1:
        raise CNDBIndexError(
            "Embedded index dataset is not stored contiguously; only contiguous embedded "
            "indexes are supported."
        )
    if version in {3, 4}:
        data_offset = struct.unpack_from("<Q", payload, 2)[0]
        data_size = struct.unpack_from("<Q", payload, 10)[0]
        return int(data_offset), int(data_size)
    raise CNDBIndexError(f"Unsupported embedded index layout message version: {version}.")


def _json_safe_filter_pipeline(filter_pipeline: Any) -> list[dict[str, Any]]:
    if filter_pipeline is None:
        return []
    safe_filters: list[dict[str, Any]] = []
    for entry in filter_pipeline:
        safe_entry: dict[str, Any] = {}
        for key, value in dict(entry).items():
            if isinstance(value, bytes):
                safe_entry[str(key)] = value.decode("utf-8", errors="replace")
            elif isinstance(value, tuple):
                safe_entry[str(key)] = [int(item) if hasattr(item, "__int__") else item for item in value]
            elif isinstance(value, np.integer):
                safe_entry[str(key)] = int(value)
            else:
                safe_entry[str(key)] = value
        safe_filters.append(safe_entry)
    return safe_filters


def _layout_name(layout_class: int) -> str:
    if layout_class == 0:
        return "compact"
    if layout_class == 1:
        return "contiguous"
    if layout_class == 2:
        return "chunked"
    return f"unknown-{layout_class}"


def _is_frame_id(name: str) -> bool:
    text = str(name)
    return text.isdigit() or (text.startswith("t_") and text[2:].isdigit())


def _json_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except TypeError:
        return int(value.item())
