"""Safe structural file detection for CNDBTools."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import h5py

from OpenMiChroM._cndb_stream.embedded import (
    BufferedStrictRangeFile,
    StrictRangeFile,
    load_pyfive_file_class,
    read_embedded_object_index,
)
from OpenMiChroM._cndb_stream.filters import (
    hdf5_filter_pipeline_supported,
    normalize_filter_pipeline,
)

from .formats import StructuralFileInfo

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def detect_structural_file(
    path_or_url: str | Path,
    *,
    timeout: float = 30.0,
    sample_size: int = 4096,
    verify_ssl: bool = True,
) -> StructuralFileInfo:
    """Detect a local or remote CNDB/NDB/SW-like structural file.

    Remote detection is conservative: it uses HEAD and small byte-range probes,
    and it never accepts a ``200 OK`` response as evidence of safe streaming.
    """

    source = str(path_or_url)
    if _is_url(source):
        return _detect_remote(
            source,
            timeout=timeout,
            sample_size=sample_size,
            verify_ssl=verify_ssl,
        )
    return _detect_local(Path(path_or_url), sample_size=sample_size)


def _detect_local(path: Path, *, sample_size: int) -> StructuralFileInfo:
    info = StructuralFileInfo(
        source=str(path),
        is_remote=False,
        range_supported=None,
        file_size=path.stat().st_size if path.exists() else None,
    )
    if not path.exists():
        info.notes.append(f"Local file does not exist: {path}")
        return info

    sample = _read_local_sample(path, sample_size)
    info.file_type = _file_type_from_extension(str(path))
    _classify_sample(info, sample)

    if info.detected_hdf5:
        _inspect_local_hdf5(path, info)
    elif info.detected_text_ndb:
        info.file_type = "ndb"
        info.layout = "text-ndb"
    elif info.detected_spacewalk:
        info.file_type = "sw"
        info.layout = "spacewalk-sw"

    return info


def _detect_remote(
    url: str,
    *,
    timeout: float,
    sample_size: int,
    verify_ssl: bool,
) -> StructuralFileInfo:
    context = None if verify_ssl else ssl._create_unverified_context()
    info = StructuralFileInfo(
        source=url,
        is_remote=True,
        file_type=_file_type_from_extension(url),
    )

    _remote_head(url, info, timeout=timeout, context=context)
    sample = _remote_range_probe(url, info, timeout=timeout, sample_size=sample_size, context=context)
    _classify_sample(info, sample)

    if info.detected_hdf5:
        if info.file_type == "unknown":
            info.file_type = "hdf5"
        if info.range_supported is True:
            _inspect_remote_embedded_hdf5(url, info, timeout=timeout)
        else:
            info.notes.append(
                "Remote HDF5 file did not return 206 Partial Content to a Range "
                "request. Direct remote streaming is disabled for this endpoint."
            )
    elif info.detected_text_ndb:
        info.file_type = "ndb"
        info.layout = "text-ndb"
        info.notes.append(
            "Remote text NDB detection used only an initial byte sample. The full "
            "text file is not downloaded by default."
        )
    elif info.detected_spacewalk:
        info.file_type = "sw"
        info.layout = "spacewalk-sw"
        info.notes.append(
            "Remote text SpaceWalk detection used only an initial byte sample. The "
            "full text file is not downloaded by default."
        )

    if info.is_remote and info.detected_hdf5 and not info.has_embedded_index:
        info.notes.append(
            "No embedded index was detected. Direct remote frame streaming is not "
            "available unless an index exists or the file is downloaded/indexed locally."
        )

    return info


def _remote_head(
    url: str,
    info: StructuralFileInfo,
    *,
    timeout: float,
    context: ssl.SSLContext | None,
) -> None:
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=timeout, context=context) as response:
            info.file_size = _header_int(response.headers.get("Content-Length"))
            if response.headers.get("Accept-Ranges"):
                info.notes.append(f"Server advertises Accept-Ranges: {response.headers['Accept-Ranges']}")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        info.notes.append(f"HEAD probe failed: {exc}")


def _remote_range_probe(
    url: str,
    info: StructuralFileInfo,
    *,
    timeout: float,
    sample_size: int,
    context: ssl.SSLContext | None,
) -> bytes:
    stop = max(0, sample_size - 1)
    request = Request(url, headers={"Range": f"bytes=0-{stop}"})
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            status = response.getcode()
            info.range_supported = status == 206
            content_range = response.headers.get("Content-Range")
            if content_range:
                size = _size_from_content_range(content_range)
                if size is not None:
                    info.file_size = size
            elif info.file_size is None:
                info.file_size = _header_int(response.headers.get("Content-Length"))
            if status == 200:
                info.notes.append(
                    "Server returned 200 OK to a Range request. Detection read only "
                    "a small sample, but streaming is disabled to prevent accidental "
                    "full-file downloads."
                )
            elif status != 206:
                info.notes.append(f"Server returned status {status} to a Range request.")
            return response.read(sample_size)
    except HTTPError as exc:
        info.range_supported = False
        info.notes.append(f"Range probe failed with HTTP {exc.code}: {exc.reason}")
    except (URLError, TimeoutError, OSError) as exc:
        info.range_supported = False
        info.notes.append(f"Range probe failed: {exc}")
    return b""


def _inspect_local_hdf5(path: Path, info: StructuralFileInfo) -> None:
    info.detected_hdf5 = True
    if info.file_type in {"unknown", "hdf5"}:
        info.file_type = "cndb" if path.suffix.lower() == ".cndb" else "hdf5"

    try:
        with h5py.File(path, "r") as h5:
            info.has_embedded_index = "_index" in h5 or "_index_offset" in h5.attrs
            _inspect_h5py_layout(h5, info)
    except OSError as exc:
        info.notes.append(f"Could not inspect local HDF5 structure: {exc}")


def _inspect_h5py_layout(h5: h5py.File, info: StructuralFileInfo) -> None:
    simple_frames = _simple_frame_names(h5)
    nested = _nested_trajectory_names(h5)

    if simple_frames:
        info.layout = (
            "openmichrom-cndb-v2"
            if _is_openmichrom_cndb_v2_h5py(h5)
            else "openmichrom-simple-cndb"
        )
        info.frame_count = len(simple_frames)
        first = h5[simple_frames[0]]
        _record_dataset_info(first, info)
        info.coordinate_paths.append(first.name)
        if "types" in h5:
            info.bead_count = int(len(h5["types"]))
        elif len(first.shape) >= 1:
            info.bead_count = int(first.shape[0])
    elif nested:
        info.layout = "nested-ndb-swb"
        info.trajectories = nested
        first_group = h5[nested[0]]
        spatial = first_group["spatial_position"]
        frames = _frame_names(spatial)
        info.frame_count = len(frames)
        if frames:
            first = spatial[frames[0]]
            _record_dataset_info(first, info)
            info.coordinate_paths.append(first.name)
            if len(first.shape) >= 1:
                info.bead_count = int(first.shape[0])
        if "Header" in h5 or info.file_type in {"sw", "swb"}:
            info.file_type = "sw"
    elif "Header" in h5:
        info.layout = "hdf5-with-header"
        if info.file_type == "unknown":
            info.file_type = "hdf5"
    else:
        info.layout = "unknown-hdf5"

    if info.compression is None and info.chunks is None and info.coordinate_paths:
        info.direct_streaming_supported = bool(info.has_embedded_index or not info.is_remote)


def _inspect_remote_embedded_hdf5(url: str, info: StructuralFileInfo, *, timeout: float) -> None:
    try:
        file_cls = load_pyfive_file_class()
        raw_file = StrictRangeFile(url, timeout=timeout)
        range_file = BufferedStrictRangeFile(raw_file)
        probe_h5 = file_cls(range_file, index={"__dummy__": {}})
        embedded = read_embedded_object_index(probe_h5, range_file)
        object_index = embedded["index"]
        indexed_h5 = file_cls(range_file, index=object_index)
    except Exception as exc:
        info.notes.append(f"Embedded-index inspection failed: {exc}")
        return

    info.has_embedded_index = True
    root_children = object_index.get("/", {})
    simple_frames = [name for name in root_children if _is_frame_name(name)]
    trajectories = [
        name
        for name in root_children
        if f"/{name}/spatial_position" in object_index
    ]
    info.trajectories = sorted(trajectories)

    if simple_frames:
        frame_ids = _sorted_frame_names(simple_frames)
        info.layout = (
            "openmichrom-cndb-v2"
            if "Header" in root_children
            else "openmichrom-simple-cndb"
        )
        info.frame_count = len(frame_ids)
        info.coordinate_paths.append(f"/{frame_ids[0]}")
    elif trajectories:
        first_traj = info.trajectories[0]
        spatial_path = f"/{first_traj}/spatial_position"
        frame_ids = _sorted_frame_names(
            [name for name in object_index.get(spatial_path, {}) if _is_frame_name(name)]
        )
        info.layout = "nested-ndb-swb"
        info.frame_count = len(frame_ids)
        if frame_ids:
            info.coordinate_paths.append(f"{spatial_path}/{frame_ids[0]}")
    else:
        info.layout = "unknown-hdf5"

    if info.coordinate_paths:
        _inspect_pyfive_dataset(indexed_h5, info.coordinate_paths[0], info)
    info.notes.append(
        f"Embedded index contains {len(object_index)} indexed HDF5 object paths."
    )


def _inspect_pyfive_dataset(h5: Any, path: str, info: StructuralFileInfo) -> None:
    try:
        dataset = h5[path]
        shape = [int(dim) for dim in dataset.shape]
        info.bead_count = int(shape[0]) if shape else None
        info.dtype = dataset.dtype.name
        info.compression = dataset.compression
        info.filters = normalize_filter_pipeline(getattr(dataset, "filter_pipeline", None))
        info.chunks = [int(dim) for dim in dataset.chunks] if dataset.chunks is not None else None
        layout = _layout_name(dataset.id.layout_class)
        data_offset = getattr(dataset.id, "data_offset", None)
        contiguous_supported = (
            layout == "contiguous"
            and info.compression is None
            and data_offset is not None
            and len(shape) == 2
            and shape[1] == 3
        )
        chunked_supported = (
            layout == "chunked"
            and len(shape) == 2
            and shape[1] == 3
            and hdf5_filter_pipeline_supported(info.filters)
        )
        info.direct_streaming_supported = bool(contiguous_supported or chunked_supported)
    except Exception as exc:
        info.notes.append(f"First coordinate dataset inspection failed: {exc}")


def _classify_sample(info: StructuralFileInfo, sample: bytes) -> None:
    if sample.startswith(HDF5_SIGNATURE):
        info.detected_hdf5 = True
        if info.file_type == "unknown":
            info.file_type = "hdf5"
        return

    text = sample.decode("utf-8", errors="ignore")
    stripped = text.lstrip()
    if stripped.startswith("HEADER") or "CHROM" in text[:2048] or "MODEL" in text[:2048]:
        info.detected_text_ndb = True
        info.file_type = "ndb"
        info.layout = "text-ndb"
    if stripped.startswith("##format=sw") or "\ntrace " in text[:2048]:
        info.detected_spacewalk = True
        info.file_type = "sw"
        info.layout = "spacewalk-sw"


def _record_dataset_info(dataset: h5py.Dataset, info: StructuralFileInfo) -> None:
    info.dtype = dataset.dtype.name
    info.compression = dataset.compression
    info.filters = _h5py_dataset_filters(dataset)
    info.chunks = list(dataset.chunks) if dataset.chunks is not None else None
    if len(dataset.shape) >= 1:
        info.bead_count = int(dataset.shape[0])
    contiguous_supported = (
        dataset.chunks is None
        and dataset.compression is None
        and dataset.id.get_offset() is not None
        and len(dataset.shape) == 2
        and int(dataset.shape[1]) == 3
    )
    chunked_supported = (
        dataset.chunks is not None
        and len(dataset.shape) == 2
        and int(dataset.shape[1]) == 3
        and hdf5_filter_pipeline_supported(info.filters)
    )
    info.direct_streaming_supported = bool(
        (contiguous_supported or chunked_supported)
        and (not info.is_remote or info.has_embedded_index)
    )


def _is_openmichrom_cndb_v2_h5py(h5: h5py.File) -> bool:
    if "Header" not in h5 or not isinstance(h5["Header"], h5py.Group):
        return False
    attrs = h5["Header"].attrs
    format_name = _decode_attr(attrs.get("format_name"))
    format_version = _decode_attr(attrs.get("format_version"))
    return format_name == "OpenMiChroM-CNDB" and str(format_version).startswith("2")


def _h5py_dataset_filters(dataset: h5py.Dataset) -> list[dict[str, Any]]:
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
                "name": name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name),
                "client_data": [int(value) for value in cd_values],
                "flags": int(flags),
            }
        )
    return filters


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return _decode_attr(value.item())
        except (ValueError, TypeError):
            return value
    return value


def _read_local_sample(path: Path, sample_size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(sample_size)


def _simple_frame_names(h5: h5py.File) -> list[str]:
    return _sorted_frame_names(
        [
            name
            for name, obj in h5.items()
            if isinstance(obj, h5py.Dataset)
            and _is_frame_name(name)
            and len(obj.shape) == 2
            and int(obj.shape[1]) == 3
        ]
    )


def _nested_trajectory_names(h5: h5py.File) -> list[str]:
    names = []
    for name, obj in h5.items():
        if not isinstance(obj, h5py.Group) or "spatial_position" not in obj:
            continue
        if isinstance(obj["spatial_position"], h5py.Group):
            names.append(name)
    return sorted(names)


def _frame_names(group: h5py.Group) -> list[str]:
    return _sorted_frame_names(
        [
            name
            for name, obj in group.items()
            if isinstance(obj, h5py.Dataset)
            and _is_frame_name(name)
            and len(obj.shape) == 2
            and int(obj.shape[1]) == 3
        ]
    )


def _is_frame_name(name: str) -> bool:
    text = str(name)
    return text.isdigit() or (text.startswith("t_") and text[2:].isdigit())


def _sorted_frame_names(names: list[str]) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        text = str(name)
        if text.startswith("t_"):
            return int(text[2:]), text
        return int(text), text

    return sorted(names, key=key)


def _layout_name(layout_class: int) -> str:
    if layout_class == 0:
        return "compact"
    if layout_class == 1:
        return "contiguous"
    if layout_class == 2:
        return "chunked"
    return f"unknown-{layout_class}"


def _file_type_from_extension(source: str) -> str:
    parsed = urlparse(source)
    suffix = Path(unquote(parsed.path)).suffix.lower()
    if not suffix:
        file_values = parse_qs(parsed.query).get("file", [])
        if file_values:
            suffix = Path(unquote(file_values[0])).suffix.lower()
    if suffix == ".cndb":
        return "cndb"
    if suffix in {".sw", ".spw", ".swb"}:
        return "sw"
    if suffix == ".ndb":
        return "ndb"
    if suffix in {".h5", ".hdf5"}:
        return "hdf5"
    return "unknown"


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def _header_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _size_from_content_range(value: str) -> int | None:
    if "/" not in value:
        return None
    size = value.rsplit("/", 1)[1]
    if size == "*":
        return None
    try:
        return int(size)
    except ValueError:
        return None
