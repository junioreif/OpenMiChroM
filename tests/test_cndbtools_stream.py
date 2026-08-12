import io
import struct

import h5py
import numpy as np

import OpenMiChroM._cndb_stream as internal_stream
from OpenMiChroM._cndb_stream import IndexedCNDB
from OpenMiChroM.CndbTools import CndbTools, cndbTools


def _make_simple_cndb(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "cndb"
        handle.attrs["format_version"] = "1.0.0"
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
        self.format_metadata = {
            "cndb_format": "cndb",
            "cndb_format_version": "1.0.0",
            "cndb_format_status": "supported",
        }
        self.index_bytes_read = 100
        self.metadata_bytes_read = 40
        self.data_bytes_read = 0
        self.index_cache_hit = True
        self.requests = []
        self.closed = False

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

    def close(self):
        self.closed = True


def _install_fake_internal_stream(monkeypatch):
    monkeypatch.setattr(internal_stream, "IndexedCNDB", _FakeIndexedCNDB)
    _FakeIndexedCNDB.last_instance = None
    _FakeIndexedCNDB.last_kwargs = None


def test_internal_stream_backend_is_packaged():
    from OpenMiChroM._cndb_stream._vendor.hdf5_indexed_reader.pyfive.high_level import File

    assert internal_stream.IndexedCNDB is IndexedCNDB
    assert File is not None


def test_vendored_reader_accepts_short_global_heap_padding():
    from OpenMiChroM._cndb_stream._vendor.hdf5_indexed_reader.pyfive.misc_low_level import (
        GlobalHeap,
    )

    object_data = b"x" + (b"\x00" * 7)
    heap_data = struct.pack("<HHIQ", 1, 0, 0, 1) + object_data + (b"\x00" * 8)
    header = struct.pack("<4sB3sQ", b"GCOL", 1, b"\x00" * 3, 16 + len(heap_data))

    heap = GlobalHeap(io.BytesIO(header + heap_data), 0)

    assert heap.objects[1] == b"x"


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
    assert tools.format_version == "1.0.0"
    assert tools.dictChromSeq["A1"][:2] == [0, 2]
    assert _FakeIndexedCNDB.last_kwargs["h5_url"] == "https://example.org/test.cndb"
    assert _FakeIndexedCNDB.last_kwargs["trajectory"] == "replica1_chr1"
    assert _FakeIndexedCNDB.last_kwargs["index_cache_path"] == "/tmp/test-index.json.gz"


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
    np.testing.assert_array_equal(xyz[0], _FakeIndexedCNDB.last_instance.data["1"][:10])


def test_streaming_xyz_noncontiguous_selection_reads_enclosing_range(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = cndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    xyz = tools.xyz(frames=[1], beadSelection=[1, 3, 4], XYZ=[0, 2])

    assert xyz.shape == (1, 3, 2)
    assert _FakeIndexedCNDB.last_instance.requests == [("1", 1, 5)]
    expected = _FakeIndexedCNDB.last_instance.data["1"][[1, 3, 4]][:, [0, 2]]
    np.testing.assert_array_equal(xyz[0], expected)


def test_remote_resources_are_closed(monkeypatch):
    _install_fake_internal_stream(monkeypatch)
    tools = CndbTools.from_remote("https://example.org/test.cndb", trajectory="replica1_chr1")

    tools.close()
    tools.close()

    assert _FakeIndexedCNDB.last_instance.closed is True
