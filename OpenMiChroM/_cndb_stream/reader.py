"""Indexed CNDB reader for contiguous OpenMiChroM frame datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import numpy as np

from .analysis import distance_matrix, radius_of_gyration
from .exceptions import CNDBIndexError, CNDBStreamError, FrameNotFoundError, UnsupportedLayoutError
from .index import INDEX_FORMAT
from .remote import RemoteByteReader
from .utils import coerce_frame_id, json_safe_value


class LocalByteReader:
    """Local binary reader that counts returned bytes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.bytes_read = 0

    def read_range(self, start: int, stop_exclusive: int) -> bytes:
        """Read ``[start, stop_exclusive)`` bytes from a local file."""

        if start < 0:
            raise ValueError("Range start must be non-negative.")
        if stop_exclusive < start:
            raise ValueError("Range stop must be greater than or equal to start.")
        length = stop_exclusive - start
        if length == 0:
            return b""

        with self.path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(length)

        if len(data) != length:
            raise CNDBStreamError(
                f"Requested {length} bytes from {self.path}, but only read {len(data)} bytes."
            )

        self.bytes_read += len(data)
        return data

    def reset_byte_counter(self) -> None:
        """Reset the byte counter."""

        self.bytes_read = 0


class IndexedCNDB:
    """Read OpenMiChroM CNDB coordinates with indexed partial byte access."""

    def __init__(
        self,
        *,
        h5_path: str | Path | None = None,
        index_path: str | Path | None = None,
        h5_url: str | None = None,
        index_url: str | None = None,
        trajectory: str | None = None,
        timeout: float = 30.0,
        _index: dict[str, Any] | None = None,
        _embedded_provider: Any | None = None,
    ) -> None:
        self.timeout = timeout
        self._embedded_provider = _embedded_provider

        has_local_h5 = h5_path is not None
        has_remote_h5 = h5_url is not None
        has_local_index = index_path is not None
        has_remote_index = index_url is not None
        has_embedded_index = _index is not None
        if has_local_h5 == has_remote_h5:
            raise ValueError("Use exactly one of h5_path or h5_url.")
        if sum([has_local_index, has_remote_index, has_embedded_index]) != 1:
            raise ValueError("Use exactly one of index_path, index_url, or embedded index.")

        if has_embedded_index:
            self.index = _index
        elif has_local_index:
            self.index = self._load_local_index(Path(index_path))  # type: ignore[arg-type]
        else:
            self.index = self._load_remote_index(str(index_url))

        if has_local_h5:
            self._byte_reader = LocalByteReader(Path(h5_path))  # type: ignore[arg-type]
            self.h5_source = str(h5_path)
        else:
            self._byte_reader = RemoteByteReader(str(h5_url), timeout=timeout)
            self.h5_source = str(h5_url)

        self._validate_index()
        self.current_trajectory = self._resolve_trajectory(trajectory)
        self._coordinate_index = self._resolve_coordinate_index()
        self._metadata_cache: dict[str, Any] = {}
        self._metadata_errors: dict[str, str] = {}

    @classmethod
    def from_embedded_index(
        cls,
        *,
        h5_url: str,
        trajectory: str | None = None,
        timeout: float = 30.0,
        fetch_size: int = 8192,
        cache_size: int = 1024 * 1024,
        index_cache_path: str | Path | None = None,
        use_index_cache: bool = True,
    ) -> "IndexedCNDB":
        """Open a remote CNDB using its embedded hdf5-indexer object index.

        This downloads only the embedded ``/_index`` dataset and selected HDF5
        metadata through HTTP Range requests. Frame coordinate reads still use
        exact byte ranges and are counted separately by ``data_bytes_read``.

        If ``index_cache_path`` is provided, the parsed embedded object-offset
        index is cached locally as JSON.gz. Later opens can reuse that cache
        without re-reading the remote ``/_index`` payload. Set
        ``use_index_cache=False`` to force a fresh remote embedded-index read.
        """

        from .embedded import EmbeddedIndexProvider

        provider = EmbeddedIndexProvider(
            h5_url=h5_url,
            trajectory=trajectory,
            timeout=timeout,
            fetch_size=fetch_size,
            cache_size=cache_size,
            index_cache_path=index_cache_path,
            use_index_cache=use_index_cache,
        )
        index = provider.to_lazy_cndb_index()
        return cls(
            h5_url=h5_url,
            _index=index,
            trajectory=provider.current_trajectory,
            timeout=timeout,
            _embedded_provider=provider,
        )

    @property
    def n_frames(self) -> int:
        """Number of indexed frame datasets."""

        return int(self._coordinate_index["n_frames"])

    @property
    def n_beads(self) -> int:
        """Number of beads per frame."""

        return int(self._coordinate_index["n_beads"])

    @property
    def frame_ids(self) -> list[str]:
        """Frame identifiers sorted numerically as strings."""

        return list(self._coordinate_index["frame_ids"])

    @property
    def trajectories(self) -> list[str]:
        """Available trajectory names for nested CNDB/NDB indexes."""

        if "available_trajectories" in self.index:
            return list(self.index["available_trajectories"])
        return list(self.index.get("trajectories", {}).keys())

    @property
    def types(self) -> list[Any] | None:
        """Chromatin type labels when available."""

        if "types" in self._coordinate_index:
            return self._coordinate_index.get("types")
        values = self._read_metadata_values(self._coordinate_index.get("types_path"))
        if values is None:
            return None
        safe_values = json_safe_value(values)
        if isinstance(safe_values, list):
            return safe_values
        return [safe_values]

    @property
    def genomic_positions(self) -> Any | None:
        """Genomic position metadata when available."""

        return self._read_metadata_values(self._coordinate_index.get("genomic_position_path"))

    @property
    def data_bytes_read(self) -> int:
        """Coordinate payload bytes read by direct frame byte-range requests."""

        return int(self._byte_reader.bytes_read)

    @property
    def index_bytes_read(self) -> int:
        """Embedded index payload bytes read, when using ``from_embedded_index``."""

        if self._embedded_provider is None:
            return 0
        return int(getattr(self._embedded_provider, "index_bytes_read", 0))

    @property
    def metadata_bytes_read(self) -> int:
        """HDF5 metadata bytes read by the embedded-index backend."""

        if self._embedded_provider is None:
            return 0
        return int(getattr(self._embedded_provider, "metadata_bytes_read", 0))

    @property
    def bytes_read(self) -> int:
        """Total bytes read by the OpenMiChroM CNDB streaming backend."""

        return self.data_bytes_read + self.index_bytes_read + self.metadata_bytes_read

    @property
    def index_cache_hit(self) -> bool:
        """Whether the embedded object-offset index was loaded from local cache."""

        if self._embedded_provider is None:
            return False
        return bool(getattr(self._embedded_provider, "index_cache_hit", False))

    @property
    def index_cache_path(self) -> str | None:
        """Local embedded-index cache path for this reader, when configured."""

        if self._embedded_provider is None:
            return None
        return getattr(self._embedded_provider, "index_cache_path", None)

    @property
    def metadata_cache_size(self) -> int:
        """Number of resolved frame metadata entries cached by this reader."""

        return len(self._coordinate_index.get("frames", {}))

    def reset_byte_counter(self) -> None:
        """Reset the coordinate payload byte counter.

        Embedded index and metadata counters are session-level diagnostics and
        are not reset because they describe the cost of opening and lazily
        inspecting the remote HDF5 file.
        """

        self._byte_reader.reset_byte_counter()

    def get_coordinates(
        self,
        frame: int | str,
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read coordinates for one frame or bead subset.

        For contiguous uncompressed frame datasets shaped ``(n_beads, 3)``, this
        computes the exact byte range for the requested bead rows and reads only
        those bytes.
        """

        frame_info = self._get_frame_info(frame)
        self._ensure_supported(frame_info)

        shape = tuple(int(dim) for dim in frame_info["shape"])
        start_row, stop_row = self._normalize_rows(start, stop, n_rows=shape[0])
        rows = stop_row - start_row
        dtype = np.dtype(frame_info["dtype"])
        row_size = int(shape[1] * dtype.itemsize)
        byte_start = int(frame_info["data_offset"]) + start_row * row_size
        byte_stop = byte_start + rows * row_size

        raw = self._byte_reader.read_range(byte_start, byte_stop)
        coords = np.frombuffer(raw, dtype=dtype).reshape((rows, shape[1]))
        return coords.copy()

    def get_frame(self, frame: int | str) -> np.ndarray:
        """Read a full frame of coordinates."""

        return self.get_coordinates(frame=frame)

    def get_frames(
        self,
        frames: Iterable[int | str],
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read multiple frames and stack them as ``(n_frames, n_beads, 3)``."""

        arrays = [
            self.get_coordinates(frame=frame, start=start, stop=stop)
            for frame in frames
        ]
        if not arrays:
            return np.empty((0, 0, 3), dtype=np.float32)
        return np.stack(arrays, axis=0)

    def get_coordinates_many(
        self,
        frames: Iterable[int | str],
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read multiple coordinate frames.

        This currently performs one byte-range read per frame. The method exists
        as a stable API point for future batching or adjacent-range coalescing.
        """

        return self.get_frames(frames=frames, start=start, stop=stop)

    def prefetch_frame_metadata(self, frames: Iterable[int | str]) -> dict[str, dict[str, Any]]:
        """Resolve and cache frame dataset metadata without reading coordinates."""

        return {self._resolve_frame_id(frame): self._get_frame_info(frame) for frame in frames}

    def get_distance_matrix(
        self,
        frame: int | str,
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        """Read coordinates and compute a pairwise distance matrix."""

        coords = self.get_coordinates(frame=frame, start=start, stop=stop)
        return distance_matrix(coords)

    def radius_of_gyration(
        self,
        frame: int | str,
        start: int | None = None,
        stop: int | None = None,
    ) -> float:
        """Read coordinates and compute radius of gyration."""

        coords = self.get_coordinates(frame=frame, start=start, stop=stop)
        return radius_of_gyration(coords)

    def get_radius_of_gyration_many(
        self,
        frames: Iterable[int | str],
        start: int | None = None,
        stop: int | None = None,
    ) -> np.ndarray:
        """Compute radius of gyration for several frames."""

        return np.array(
            [
                self.radius_of_gyration(frame=frame, start=start, stop=stop)
                for frame in frames
            ],
            dtype=float,
        )

    @staticmethod
    def _load_local_index(index_path: Path) -> dict[str, Any]:
        with index_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_remote_index(self, index_url: str) -> dict[str, Any]:
        with urlopen(index_url, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _validate_index(self) -> None:
        if self.index.get("format") != INDEX_FORMAT:
            raise CNDBIndexError(
                f"Unsupported index format {self.index.get('format')!r}; expected {INDEX_FORMAT!r}."
            )
        if "frames" not in self.index and "trajectories" not in self.index:
            raise CNDBIndexError("Index is missing required 'frames' mapping.")

    def _resolve_trajectory(self, trajectory: str | None) -> str | None:
        trajectories = self.trajectories
        if not trajectories:
            if trajectory is not None:
                raise CNDBIndexError(
                    "A trajectory was provided, but this index is not a nested trajectory index."
                )
            return None
        if trajectory is not None:
            if trajectory not in trajectories:
                raise CNDBIndexError(
                    f"Trajectory {trajectory!r} was not found. Available trajectories: {trajectories}"
                )
            return trajectory
        if len(trajectories) == 1:
            return trajectories[0]
        raise CNDBIndexError(
            "This index contains multiple trajectories. Specify one with "
            f"trajectory=... Available trajectories: {trajectories}"
        )

    def _resolve_coordinate_index(self) -> dict[str, Any]:
        if self.current_trajectory is None:
            return self.index
        return self.index["trajectories"][self.current_trajectory]

    def _get_frame_info(self, frame: int | str) -> dict[str, Any]:
        frame_id = self._resolve_frame_id(frame)
        frames = self._coordinate_index["frames"]
        if frame_id not in frames and self._embedded_provider is not None:
            frames[frame_id] = self._embedded_provider.frame_info(frame_id)
        if frame_id not in frames:
            raise FrameNotFoundError(
                f"Frame {frame_id!r} was not found. Available frames: {self.frame_ids[:10]}"
            )
        return frames[frame_id]

    def _resolve_frame_id(self, frame: int | str) -> str:
        frame_id = coerce_frame_id(frame)
        if frame_id in self.frame_ids:
            return frame_id
        t_frame_id = f"t_{frame_id}"
        if t_frame_id in self.frame_ids:
            return t_frame_id
        return frame_id

    def _read_metadata_values(self, path: str | None) -> Any | None:
        if path is None:
            return None
        if path in self._metadata_cache:
            return self._metadata_cache[path]
        if self._embedded_provider is None:
            return None
        try:
            values = self._embedded_provider.read_dataset(path)
        except Exception as exc:
            self._metadata_errors[path] = str(exc)
            return None
        self._metadata_cache[path] = values
        return values

    @staticmethod
    def _ensure_supported(frame_info: dict[str, Any]) -> None:
        if frame_info.get("direct_read_supported") is True:
            return
        raise UnsupportedLayoutError(
            "Frame dataset cannot be read with direct byte ranges in the MVP. "
            f"path={frame_info.get('path')!r}, layout={frame_info.get('layout')!r}, "
            f"compression={frame_info.get('compression')!r}, "
            f"data_offset={frame_info.get('data_offset')!r}."
        )

    @staticmethod
    def _normalize_rows(
        start: int | None,
        stop: int | None,
        *,
        n_rows: int,
    ) -> tuple[int, int]:
        start_row = 0 if start is None else int(start)
        stop_row = n_rows if stop is None else int(stop)
        if start_row < 0:
            raise ValueError("start must be non-negative.")
        if stop_row < start_row:
            raise ValueError("stop must be greater than or equal to start.")
        if stop_row > n_rows:
            raise ValueError(f"stop={stop_row} exceeds frame bead count {n_rows}.")
        return start_row, stop_row
