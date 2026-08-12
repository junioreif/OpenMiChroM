import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import h5py
import numpy as np
import pytest

from OpenMiChroM.CndbTools import CndbTools
from OpenMiChroM.CustomReporter import SaveStructure
from OpenMiChroM._cndb_stream import (
    CNDBFormatError,
    CNDBIndexError,
    FrameNotFoundError,
    LegacyCNDBVersionWarning,
    RangeRequestUnsupportedError,
    RemoteAccessError,
    UnsupportedCNDBVersionError,
)
from OpenMiChroM._cndb_stream.index import build_index
from OpenMiChroM._cndb_stream.reader import IndexedCNDB
from OpenMiChroM._cndb_stream.remote import read_range
from OpenMiChroM._cndb_stream.version import CNDB_FORMAT_VERSION


def _make_cndb(path, *, version=CNDB_FORMAT_VERSION, frames=True):
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "cndb"
        if version is not None:
            handle.attrs["format_version"] = version
        handle.create_dataset("types", data=np.array([b"A1", b"B1", b"A1", b"B1"]))
        if frames:
            handle.create_dataset("1", data=np.arange(12, dtype=np.float32).reshape(4, 3))
            handle.create_dataset(
                "3", data=(100 + np.arange(12, dtype=np.float32)).reshape(4, 3)
            )


class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/redirect.cndb":
            self.send_response(302)
            self.send_header("Location", "/trajectory.cndb")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        payload = self.server.payload
        if self.path == "/missing.cndb":
            self.send_error(404)
            return
        if self.path == "/ignore-range.cndb":
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        range_header = self.headers.get("Range")
        if range_header is None or not range_header.startswith("bytes="):
            self.send_error(400)
            return
        start_text, stop_text = range_header[6:].split("-", 1)
        start, stop = int(start_text), int(stop_text)
        selected = payload[start : stop + 1]
        declared_stop = stop
        if self.path == "/eof-clipped.cndb":
            declared_stop = min(stop, len(payload) - 1)
        if self.path == "/truncated.cndb":
            selected = selected[:-1]

        self.server.requests.append((self.path, start, stop, len(selected)))
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{declared_stop}/{len(payload)}")
        self.send_header("Content-Length", str(len(selected)))
        self.end_headers()
        self.wfile.write(selected)


@contextmanager
def _range_server(payload):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    server.daemon_threads = True
    server.payload = payload
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_load_distinguishes_pathlike_unusual_local_name_and_http_url(tmp_path, monkeypatch):
    unusual = tmp_path / "archive:trajectory.cndb"
    _make_cndb(unusual)

    with CndbTools().load(unusual) as tools:
        assert tools.is_remote is False
        assert tools.xyz(frames=1).shape == (1, 4, 3)

    monkeypatch.chdir(tmp_path)
    relative = "https:local-trajectory.cndb"
    _make_cndb(relative)
    with CndbTools().load(relative) as tools:
        assert tools.is_remote is False

    with pytest.raises(ValueError, match="scheme"):
        CndbTools.from_remote("ftp://example.org/trajectory.cndb")
    with pytest.raises(ValueError, match="host"):
        CndbTools.from_remote("https://")


def test_missing_corrupt_and_structurally_invalid_local_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        CndbTools().load(tmp_path / "missing.cndb")

    corrupt = tmp_path / "corrupt.cndb"
    corrupt.write_bytes(b"not hdf5")
    with pytest.raises(CNDBFormatError, match="Could not open"):
        CndbTools().load(corrupt)

    no_types = tmp_path / "no-types.cndb"
    with h5py.File(no_types, "w") as handle:
        handle.attrs["format"] = "cndb"
        handle.attrs["format_version"] = CNDB_FORMAT_VERSION
        handle["1"] = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(CNDBFormatError, match="types"):
        CndbTools().load(no_types)


@pytest.mark.parametrize("version", [None, "0.9"])
def test_legacy_local_versions_are_supported_with_warning(tmp_path, version):
    path = tmp_path / "legacy.cndb"
    _make_cndb(path, version=version)

    with pytest.warns(LegacyCNDBVersionWarning):
        tools = CndbTools().load(path)
    assert tools.format_status.startswith("legacy")
    tools.close()


@pytest.mark.parametrize(
    ("version", "error"),
    [("next", CNDBFormatError), ("2.0.0", UnsupportedCNDBVersionError)],
)
def test_malformed_and_future_local_versions_fail(tmp_path, version, error):
    path = tmp_path / "unsupported.cndb"
    _make_cndb(path, version=version)

    with pytest.raises(error):
        CndbTools().load(path)


def test_empty_dataset_missing_frame_and_invalid_ranges(tmp_path):
    empty = tmp_path / "empty.cndb"
    _make_cndb(empty, frames=False)
    with CndbTools().load(empty) as tools:
        assert tools.Nframes == 0
        assert tools.xyz().shape == (0, 4, 3)

    path = tmp_path / "frames.cndb"
    _make_cndb(path)
    with CndbTools().load(path) as tools:
        np.testing.assert_array_equal(tools.xyz(frames=None)[1, 0], [100, 101, 102])
        with pytest.raises(FrameNotFoundError):
            tools.xyz(frames=[2])
        with pytest.raises(IndexError):
            tools.xyz(frames=[1], beadSelection=[4])
        with pytest.raises(IndexError):
            tools.xyz(frames=[1], XYZ=[3])


def test_deterministic_http_stream_reads_only_requested_coordinate_range(tmp_path):
    path = tmp_path / "trajectory.cndb"
    index_path = tmp_path / "trajectory.index.json"
    _make_cndb(path)
    build_index(path, index_path)
    payload = path.read_bytes()

    with _range_server(payload) as (server, base_url):
        with CndbTools().load(
            f"{base_url}/trajectory.cndb",
            index_path=index_path,
        ) as tools:
            coords = tools.xyz(frames=[1], beadSelection=range(1, 3))
            np.testing.assert_array_equal(
                coords,
                np.arange(12, dtype=np.float32).reshape(4, 3)[None, 1:3],
            )
            assert tools.is_remote is True
            assert tools.stream_data_bytes_read == 24
            assert tools.stream_data_bytes_read < len(payload)
        assert tools._stream_backend.traj.closed is True
        assert len(server.requests) == 1
        assert server.requests[0][3] == 24


def test_redirect_ignored_range_truncation_and_unavailable_remote_are_explicit(tmp_path):
    path = tmp_path / "trajectory.cndb"
    _make_cndb(path)
    payload = path.read_bytes()

    with _range_server(payload) as (_, base_url):
        assert read_range(f"{base_url}/redirect.cndb", 0, 8) == payload[:8]
        assert (
            read_range(
                f"{base_url}/eof-clipped.cndb", len(payload) - 4, len(payload) + 8
            )
            == payload[-4:]
        )
        with pytest.raises(RangeRequestUnsupportedError, match="200 OK"):
            read_range(f"{base_url}/ignore-range.cndb", 0, 8)
        with pytest.raises(RemoteAccessError, match="expected exactly"):
            read_range(f"{base_url}/truncated.cndb", 0, 8)
        with pytest.raises(RemoteAccessError, match="HTTP 404"):
            read_range(f"{base_url}/missing.cndb", 0, 8)


def test_remote_index_and_cndb_versions_are_validated_before_reads(tmp_path):
    path = tmp_path / "trajectory.cndb"
    _make_cndb(path)
    index = build_index(path)

    future_index = tmp_path / "future-index.json"
    future_payload = dict(index, version="9.0")
    future_index.write_text(json.dumps(future_payload), encoding="utf-8")
    with pytest.raises(CNDBIndexError, match="index version"):
        IndexedCNDB(h5_path=path, index_path=future_index)

    future_format = tmp_path / "future-format.json"
    format_payload = dict(index, cndb_format_version="2.0.0")
    future_format.write_text(json.dumps(format_payload), encoding="utf-8")
    with pytest.raises(UnsupportedCNDBVersionError):
        IndexedCNDB(h5_path=path, index_path=future_format)


def test_cndb_reporter_writes_authoritative_format_metadata(tmp_path):
    reporter = SaveStructure(
        filePrefix="metadata",
        reportInterval=10,
        mode="cndb",
        folder=tmp_path,
        chains=[(0, 1, False)],
        typeListLetter=["A1", "B1"],
    )
    reporter.close()

    with h5py.File(tmp_path / "metadata_0.cndb", "r") as handle:
        assert handle.attrs["format"] == "cndb"
        assert handle.attrs["format_version"] == CNDB_FORMAT_VERSION
