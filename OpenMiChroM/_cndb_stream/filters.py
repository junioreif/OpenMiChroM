"""Small HDF5 chunk filter helpers used by the streaming backend."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from .exceptions import UnsupportedLayoutError


GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
LZF_FILTER = 32000

SUPPORTED_FILTERS = {GZIP_DEFLATE_FILTER, SHUFFLE_FILTER, FLETCH32_FILTER}
FILTER_NAMES = {
    GZIP_DEFLATE_FILTER: "deflate/gzip",
    SHUFFLE_FILTER: "shuffle",
    FLETCH32_FILTER: "fletcher32",
    SZIP_FILTER: "szip",
    NBIT_FILTER: "nbit",
    SCALEOFFSET_FILTER: "scale-offset",
    LZF_FILTER: "lzf",
}


def normalize_filter_pipeline(filters: Any) -> list[dict[str, Any]]:
    """Normalize h5py/pyfive filter metadata into JSON-safe dictionaries."""

    if filters is None:
        return []
    normalized: list[dict[str, Any]] = []
    for entry in filters:
        entry_dict = dict(entry)
        filter_id = entry_dict.get("filter_id", entry_dict.get("id"))
        if filter_id is None:
            continue
        normalized.append(
            {
                "id": int(filter_id),
                "name": _filter_name(entry_dict, int(filter_id)),
                "client_data": _client_data(entry_dict),
                "flags": int(entry_dict.get("flags", 0)),
            }
        )
    return normalized


def hdf5_filter_pipeline_supported(filters: Any) -> bool:
    """Return whether every filter in a pipeline has a tested decoder."""

    return all(entry["id"] in SUPPORTED_FILTERS for entry in normalize_filter_pipeline(filters))


def unsupported_filter_message(filters: Any) -> str:
    """Human-readable unsupported filter summary."""

    normalized = normalize_filter_pipeline(filters)
    unsupported = [entry for entry in normalized if entry["id"] not in SUPPORTED_FILTERS]
    if not unsupported:
        return "All filters are supported."
    return "Unsupported HDF5 filter pipeline: " + ", ".join(
        f"{entry['name']} (id={entry['id']})" for entry in unsupported
    )


def decode_hdf5_chunk(
    raw_bytes: bytes | bytearray,
    filters: Any,
    dtype: np.dtype | str,
    chunk_shape: tuple[int, ...] | list[int],
    *,
    filter_mask: int = 0,
) -> bytes:
    """Decode raw HDF5 chunk bytes for the supported built-in filter subset.

    The HDF5 filter pipeline is decoded in reverse order. This helper is used
    directly by tests and documents the filter semantics expected from the
    vendored pyfive backend.
    """

    chunk_buffer: bytes | bytearray = raw_bytes
    normalized = normalize_filter_pipeline(filters)
    dtype = np.dtype(dtype)
    expected_nbytes = int(np.prod(chunk_shape, dtype=np.int64) * dtype.itemsize)
    num_filters = len(normalized)
    for reverse_index, entry in enumerate(reversed(normalized)):
        filter_index = num_filters - reverse_index - 1
        if filter_mask & (1 << filter_index):
            continue
        filter_id = entry["id"]
        if filter_id == GZIP_DEFLATE_FILTER:
            chunk_buffer = zlib.decompress(bytes(chunk_buffer))
        elif filter_id == SHUFFLE_FILTER:
            chunk_buffer = _unshuffle(bytes(chunk_buffer), dtype.itemsize)
        elif filter_id == FLETCH32_FILTER:
            _verify_fletcher32(bytes(chunk_buffer))
            chunk_buffer = bytes(chunk_buffer)[:-4]
        else:
            raise UnsupportedLayoutError(unsupported_filter_message([entry]))
    if len(chunk_buffer) != expected_nbytes:
        raise UnsupportedLayoutError(
            "Decoded HDF5 chunk size does not match the expected chunk shape: "
            f"decoded={len(chunk_buffer)} bytes, expected={expected_nbytes} bytes."
        )
    return bytes(chunk_buffer)


def _filter_name(entry: dict[str, Any], filter_id: int) -> str:
    for key in ("name", "filter_name"):
        value = entry.get(key)
        if value:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
    return FILTER_NAMES.get(filter_id, "unknown")


def _client_data(entry: dict[str, Any]) -> list[int]:
    value = entry.get("client_data", entry.get("client_data_values", ()))
    return [int(item) for item in value]


def _unshuffle(chunk_buffer: bytes, itemsize: int) -> bytes:
    if itemsize <= 0:
        raise UnsupportedLayoutError("Shuffle filter requires a positive dtype itemsize.")
    if len(chunk_buffer) % itemsize:
        raise UnsupportedLayoutError(
            "Shuffle-filtered chunk size is not divisible by dtype itemsize."
        )
    step = len(chunk_buffer) // itemsize
    unshuffled = bytearray(len(chunk_buffer))
    for byte_index in range(itemsize):
        start = byte_index * step
        stop = (byte_index + 1) * step
        unshuffled[byte_index::itemsize] = chunk_buffer[start:stop]
    return bytes(unshuffled)


def _verify_fletcher32(chunk_buffer: bytes) -> None:
    if len(chunk_buffer) < 4:
        raise UnsupportedLayoutError("Fletcher32 chunk is too short to contain a checksum.")
    data = chunk_buffer[:-4]
    padded = data + (b"\x00" if len(data) % 2 else b"")
    values = np.frombuffer(padded, "<u2")
    sum1 = np.uint32(0)
    sum2 = np.uint32(0)
    for value in values:
        sum1 = (sum1 + value) % 65535
        sum2 = (sum2 + sum1) % 65535
    ref_sum1, ref_sum2 = np.frombuffer(chunk_buffer[-4:], ">u2")
    if sum1 != ref_sum1 or sum2 != ref_sum2:
        raise UnsupportedLayoutError("Fletcher32 checksum is invalid for HDF5 chunk.")
