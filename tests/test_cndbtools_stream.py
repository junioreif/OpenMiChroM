import h5py
import numpy as np
import pytest
import functools
import os
import re
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import OpenMiChroM._cndb_stream as internal_stream
from OpenMiChroM._cndb_stream import IndexedCNDB
from OpenMiChroM._cndb_stream.embedded_writer import initialize_cndb_header, write_embedded_index
from OpenMiChroM.CndbTools import CndbTools, cndbTools
from OpenMiChroM._structural_io import coalesce_indices


class _RangeRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        handle = open(path, "rb")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d+)$", range_header)
            if match is None:
                handle.close()
                self.send_error(416, "Invalid Range")
                return None
            start = int(match.group(1))
            stop_inclusive = min(int(match.group(2)), file_size - 1)
            if start >= file_size or stop_inclusive < start:
                handle.close()
                self.send_error(416, "Range Not Satisfiable")
                return None
            length = stop_inclusive - start + 1
            self.send_response(206)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{stop_inclusive}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            handle.seek(start)
            self._range_length = length
            return handle

        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        self._range_length = None
        return handle

    def copyfile(self, source, outputfile):
        if getattr(self, "_range_length", None) is None:
            return super().copyfile(source, outputfile)
        remaining = self._range_length
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


@contextmanager
def _serve_directory(directory):
    factory = functools.partial(_RangeRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _make_simple_cndb(path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("types", data=np.array([b"A1", b"B1", b"A1", b"B1"]))
        handle.create_dataset(
            "1",
            data=np.arange(12, dtype=np.float32).reshape(4, 3),
        )
        handle.create_dataset(
            "2",
            data=(100 + np.arange(12, dtype=np.float32)).reshape(4, 3),
        )


class _FakeIndexedCNDB:
    last_instance = None
    last_kwargs = None

    def __init__(self):
        self.data = {
            "1": np.arange(60, dtype=np.float32).reshape(20, 3),
            "2": (1000 + np.arange(60, dtype=np.float32)).reshape(20, 3),
        }
        self.n_frames = 2
        self.n_beads = 20
        self.frame_ids = ["1", "2"]
        self.trajectories = ["replica1_chr1"]
        self.current_trajectory = "replica1_chr1"
        self.types = ["A1", "B1"] * 10
        self.genomic_positions = np.arange(40).reshape(20, 2)
        self.index_bytes_read = 100
        self.metadata_bytes_read = 40
        self.data_bytes_read = 0
        self.index_cache_hit = True
        self.requests = []

    @property
    def bytes_read(self):
        return self.index_bytes_read + self.metadata_bytes_read + self.data_bytes_read

    @classmethod
    def from_embedded_index(cls, **kwargs):
        cls.last_kwargs = kwargs
        cls.last_instance = cls()
        return cls.last_instance

    def get_coordinates(self, frame, start=None, stop=None):
        start = 0 if start is None else start
        stop = self.n_beads if stop is None else stop
        self.requests.append((str(frame), start, stop))
        coords = self.data[str(frame)][start:stop]
        self.data_bytes_read += coords.nbytes
        return coords


def _install_fake_internal_stream(monkeypatch):
    monkeypatch.setattr(internal_stream, "IndexedCNDB", _FakeIndexedCNDB)
    _FakeIndexedCNDB.last_instance = None
    _FakeIndexedCNDB.last_kwargs = None


def test_internal_stream_backend_is_packaged():
    from OpenMiChroM._cndb_stream._vendor.hdf5_indexed_reader.pyfive.high_level import File

    assert internal_stream.IndexedCNDB is IndexedCNDB
    assert File is not None


def test_local_cndbtools_xyz_behavior_is_preserved(tmp_path):
    cndb_path = tmp_path / "toy.cndb"
    _make_simple_cndb(cndb_path)

    tools = cndbTools().load(str(cndb_path))
    xyz = tools.xyz(frames=[1, 2], beadSelection=[0, 2], XYZ=[0, 2])

    expected = np.array(
        [
            [[0, 2], [6, 8]],
            [[100, 102], [106, 108]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(xyz, expected)
    assert tools.is_remote is False
    assert tools.stream_stats()["bytes_read"] == 0


def test_from_remote_uses_internal_stream_backend(monkeypatch):
    _install_fake_internal_stream(monkeypatch)

    tools = CndbTools.from_remote(
        h5_url="https://example.org/test.cndb",
        trajectory="replica1_chr1",
        index_cache_path="/tmp/test-index.json.gz",
    )

    assert tools.is_remote is True
    assert tools.Nframes == 2
    assert tools.Nbeads == 20
    assert tools.frame_ids == ["1", "2"]
    assert tools.trajectories == ["replica1_chr1"]
    assert tools.dictChromSeq["A1"] == list(range(0, 20, 2))
    assert tools.dictChromSeq["B1"] == list(range(1, 20, 2))
    np.testing.assert_array_equal(tools.genomic_positions, np.arange(40).reshape(20, 2))
    assert _FakeIndexedCNDB.last_kwargs["h5_url"] == "https://example.org/test.cndb"
    assert _FakeIndexedCNDB.last_kwargs["trajectory"] == "replica1_chr1"
    assert _FakeIndexedCNDB.last_kwargs["index_cache_path"] == "/tmp/test-index.json.gz"


def test_from_remote_normalizes_legacy_numeric_type_codes(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    monkeypatch.setattr(
        _FakeIndexedCNDB,
        "from_embedded_index",
        classmethod(lambda cls, **kwargs: cls()),
    )
    original_init = _FakeIndexedCNDB.__init__

    def numeric_init(self):
        original_init(self)
        self.types = np.array([0, 1, 2, 3, 4, 5, 6] + [0] * 13)

    monkeypatch.setattr(_FakeIndexedCNDB, "__init__", numeric_init)

    tools = CndbTools.from_remote("https://example.org/legacy.cndb")

    assert tools.types[:7] == ["A1", "A2", "B1", "B2", "B3", "B4", "NA"]
    assert tools.dictChromSeq["A1"] == [0] + list(range(7, 20))


def test_classic_local_loader_normalizes_legacy_numeric_type_codes(tmp_path):
    path = tmp_path / "legacy.cndb"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("types", data=np.arange(7, dtype=np.int64))
        handle.create_dataset("1", data=np.zeros((7, 3), dtype=np.float32))

    tools = CndbTools()
    tools.load(path)

    assert tools.types == ["A1", "A2", "B1", "B2", "B3", "B4", "NA"]


def test_streaming_xyz_contiguous_range_reads_exact_coordinate_bytes(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1, 2], beadSelection=range(0, 10), XYZ=[0, 1, 2])

    assert xyz.shape == (2, 10, 3)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 0, 10), ("2", 0, 10)]
    assert tools.stream_data_bytes_read == 2 * 10 * 3 * np.dtype("float32").itemsize
    stats = tools.stream_stats()
    assert stats["data_bytes_read"] == 240
    assert stats["index_cache_hit"] is True
    assert stats["selection_strategy"] == "single-range"
    assert stats["range_request_count"] == 2
    assert stats["requested_coordinate_bytes"] == 240
    assert stats["transferred_coordinate_bytes"] == 240
    assert stats["overfetch_coordinate_bytes"] == 0
    np.testing.assert_array_equal(xyz[0], _FakeIndexedCNDB.last_instance.data["1"][:10])


def test_streaming_xyz_slice_selection_reads_one_range(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=slice(2, 8), XYZ=[1, 2])

    assert xyz.shape == (1, 6, 2)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 2, 8)]
    expected = _FakeIndexedCNDB.last_instance.data["1"][2:8][:, [1, 2]]
    np.testing.assert_array_equal(xyz[0], expected)
    stats = tools.stream_stats()
    assert stats["selection_strategy"] == "single-range"
    assert stats["requested_coordinate_bytes"] == 6 * 3 * np.dtype("float32").itemsize


def test_streaming_xyz_contiguous_list_reads_one_range(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=[2, 3, 4, 5], XYZ=[0, 1, 2])

    assert xyz.shape == (1, 4, 3)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 2, 6)]
    assert tools.stream_stats()["selection_strategy"] == "single-range"


def test_streaming_xyz_noncontiguous_selection_reads_coalesced_ranges(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=[1, 3, 4], XYZ=[0, 2])

    assert xyz.shape == (1, 3, 2)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 1, 2), ("1", 3, 5)]
    expected = _FakeIndexedCNDB.last_instance.data["1"][[1, 3, 4]][:, [0, 2]]
    np.testing.assert_array_equal(xyz[0], expected)
    stats = tools.stream_stats()
    assert stats["coordinate_range_requests"] == 2
    assert stats["range_request_count"] == 2
    assert stats["coalesced_range_count"] == 2
    assert stats["selection_strategy"] == "coalesced-ranges"
    assert stats["requested_data_bytes"] == 3 * 3 * np.dtype("float32").itemsize
    assert stats["transferred_data_bytes"] == stats["requested_data_bytes"]
    assert stats["overfetch_bytes"] == 0
    assert stats["requested_coordinate_bytes"] == stats["requested_data_bytes"]
    assert stats["transferred_coordinate_bytes"] == stats["transferred_data_bytes"]
    assert stats["overfetch_coordinate_bytes"] == stats["overfetch_bytes"]


def test_streaming_xyz_noncontiguous_preserves_requested_order(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=[4, 1, 4, 3], XYZ=[0, 1, 2])

    expected = _FakeIndexedCNDB.last_instance.data["1"][[4, 1, 4, 3]]
    np.testing.assert_array_equal(xyz[0], expected)
    assert tools.stream_stats()["selection_strategy"] == "coalesced-ranges"


def test_streaming_xyz_sparse_selection_can_fall_back_to_enclosing_range(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=[1, 3, 4], XYZ=[0, 2], max_ranges=1)

    assert xyz.shape == (1, 3, 2)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 1, 5)]
    stats = tools.stream_stats()
    assert stats["coordinate_range_requests"] == 1
    assert stats["range_request_count"] == 1
    assert stats["selection_strategy"] == "enclosing-range"
    assert stats["requested_data_bytes"] == 3 * 3 * np.dtype("float32").itemsize
    assert stats["transferred_data_bytes"] == 4 * 3 * np.dtype("float32").itemsize
    assert stats["overfetch_bytes"] == 1 * 3 * np.dtype("float32").itemsize


def test_coalesce_indices_groups_adjacent_blocks():
    assert coalesce_indices([0, 1, 2, 10, 11, 12]) == [(0, 3), (10, 13)]
    assert coalesce_indices([0, 10, 20], max_gap=10) == [(0, 21)]
    assert coalesce_indices([0, 10, 20], max_ranges=2) == [(0, 11), (20, 21)]


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        {},
        {"compression": "gzip"},
        {"compression": "gzip", "shuffle": True},
        {"compression": "gzip", "shuffle": True, "fletcher32": True},
    ],
)
def test_remote_chunked_cndb_streams_with_embedded_backend(tmp_path, dataset_kwargs):
    suffix = "-".join(str(key) for key, value in dataset_kwargs.items() if value) or "none"
    cndb_path = tmp_path / f"chunked-{suffix}.cndb"
    coords = np.arange(30, dtype=np.float32).reshape(10, 3)
    with h5py.File(cndb_path, "w") as handle:
        handle.create_dataset("types", data=np.array([b"A1"] * 10))
        handle.create_dataset("1", data=coords, chunks=(4, 3), **dataset_kwargs)
    write_embedded_index(cndb_path)

    with _serve_directory(tmp_path) as base_url:
        tools = CndbTools.from_remote(
            h5_url=f"{base_url}/{cndb_path.name}",
            fetch_size=512,
            cache_size=2048,
        )
        xyz = tools.xyz(frames=[1], beadSelection=range(2, 7))

    np.testing.assert_array_equal(xyz[0], coords[2:7])
    stats = tools.stream_stats()
    assert stats["requested_coordinate_bytes"] == 5 * 3 * np.dtype("float32").itemsize
    assert stats["transferred_coordinate_bytes"] >= stats["requested_coordinate_bytes"]
    assert stats["data_bytes_read"] >= stats["requested_coordinate_bytes"]
    assert stats["data_bytes_read"] < cndb_path.stat().st_size


def test_nested_header_counts_survive_embedded_index_finalization(tmp_path):
    path = tmp_path / "nested.cndb"
    with h5py.File(path, "w") as handle:
        initialize_cndb_header(
            handle,
            n_beads=4,
            coordinate_dtype="float32",
            frame_layout="nested_trajectories",
        )
        for replica in (1, 2):
            group = handle.create_group(f"replica{replica}_chr1")
            group.create_dataset("types", data=np.array([b"A1"] * 4))
            spatial = group.create_group("spatial_position")
            for frame in (1, 2, 3):
                spatial.create_dataset(
                    str(frame),
                    data=np.full((4, 3), replica * frame, dtype=np.float32),
                )

    write_embedded_index(path)

    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        assert header["n_frames"] == 6
        assert header["n_beads"] == 4
        assert header["n_trajectories"] == 2
