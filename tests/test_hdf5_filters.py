import zlib

import numpy as np
import pytest

from OpenMiChroM._cndb_stream.exceptions import UnsupportedLayoutError
from OpenMiChroM._cndb_stream.filters import (
    FLETCH32_FILTER,
    GZIP_DEFLATE_FILTER,
    SCALEOFFSET_FILTER,
    SHUFFLE_FILTER,
    decode_hdf5_chunk,
    hdf5_filter_pipeline_supported,
    normalize_filter_pipeline,
)


def _shuffle(raw, itemsize):
    step = len(raw) // itemsize
    shuffled = bytearray(len(raw))
    for byte_index in range(itemsize):
        shuffled[byte_index * step : (byte_index + 1) * step] = raw[byte_index::itemsize]
    return bytes(shuffled)


def _append_fletcher32(raw):
    padded = raw + (b"\x00" if len(raw) % 2 else b"")
    values = np.frombuffer(padded, "<u2")
    sum1 = np.uint32(0)
    sum2 = np.uint32(0)
    for value in values:
        sum1 = (sum1 + value) % 65535
        sum2 = (sum2 + sum1) % 65535
    checksum = np.array([sum1, sum2], dtype=">u2").tobytes()
    return raw + checksum


def test_decode_hdf5_chunk_gzip_shuffle_and_fletcher32():
    array = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    raw = array.tobytes()
    encoded = _append_fletcher32(zlib.compress(_shuffle(raw, array.dtype.itemsize)))
    filters = [
        {"id": SHUFFLE_FILTER, "name": "shuffle"},
        {"id": GZIP_DEFLATE_FILTER, "name": "deflate", "client_data": [4]},
        {"id": FLETCH32_FILTER, "name": "fletcher32"},
    ]

    decoded = decode_hdf5_chunk(encoded, filters, array.dtype, array.shape)

    np.testing.assert_array_equal(np.frombuffer(decoded, dtype=array.dtype).reshape(array.shape), array)


def test_decode_hdf5_chunk_rejects_bad_fletcher32_checksum():
    array = np.arange(12, dtype=np.float32).reshape(4, 3)
    encoded = _append_fletcher32(array.tobytes())[:-1] + b"\x00"

    with pytest.raises(UnsupportedLayoutError, match="Fletcher32"):
        decode_hdf5_chunk(
            encoded,
            [{"id": FLETCH32_FILTER, "name": "fletcher32"}],
            array.dtype,
            array.shape,
        )


def test_filter_pipeline_support_classification():
    filters = normalize_filter_pipeline(
        [
            {"filter_id": SHUFFLE_FILTER, "client_data": (4,)},
            {"filter_id": GZIP_DEFLATE_FILTER, "client_data": (4,)},
        ]
    )

    assert filters == [
        {"id": SHUFFLE_FILTER, "name": "shuffle", "client_data": [4], "flags": 0},
        {"id": GZIP_DEFLATE_FILTER, "name": "deflate/gzip", "client_data": [4], "flags": 0},
    ]
    assert hdf5_filter_pipeline_supported(filters) is True
    assert hdf5_filter_pipeline_supported([{"id": SCALEOFFSET_FILTER}]) is False
