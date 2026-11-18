import functools
import os
import re
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import h5py
import numpy as np
import pytest

from OpenMiChroM.CndbTools import CndbTools, cndbTools
from OpenMiChroM._cndb_stream.exceptions import UnsupportedCNDBVersionError
from OpenMiChroM._structural_io import detect_structural_file


openmm = pytest.importorskip("openmm")
from OpenMiChroM.CustomReporter import SaveStructure  # noqa: E402


class _FakePositions:
    def __init__(self, data):
        self._data = np.asarray(data)

    def value_in_unit(self, unit):
        return self._data


class _FakeState:
    def __init__(self, data):
        self._data = np.asarray(data)

    def getPositions(self, asNumpy=True):
        return _FakePositions(self._data)


class _RangeRequestHandler(SimpleHTTPRequestHandler):
    request_log = []

    def log_message(self, format, *args):
        return

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        self.request_log.append(
            {
                "path": self.path,
                "range": range_header,
                "status": 206 if range_header else 200,
            }
        )
        handle = open(path, "rb")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d+)$", range_header)
            if match is None:
                handle.close()
                self.send_error(416, "Invalid Range")
                return None
            start = int(match.group(1))
            stop_inclusive = int(match.group(2))
            if start >= file_size or stop_inclusive < start:
                handle.close()
                self.send_error(416, "Range Not Satisfiable")
                return None
            stop_inclusive = min(stop_inclusive, file_size - 1)
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
    handler = type("Handler", (_RangeRequestHandler,), {"request_log": []})
    factory = functools.partial(handler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", handler.request_log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_tiny_indexed_cndb(tmp_path, *, indexed=True, metadata=True):
    frames = [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        (100 + np.arange(12, dtype=np.float32)).reshape(4, 3),
    ]
    reporter = SaveStructure(
        filePrefix="toy",
        reportInterval=1,
        mode="cndb",
        folder=str(tmp_path),
        chains=[(0, 3, False)],
        typeListLetter=np.array([b"A1", b"B1", b"A1", b"B1"]),
        indexed=indexed,
        metadata=metadata,
        coordinate_dtype=np.float32,
    )
    for frame in frames:
        reporter.report(None, _FakeState(frame))
    reporter.close()
    return tmp_path / "toy_0.cndb", frames


def test_save_structure_writes_cndb_v2_header_and_embedded_index(tmp_path):
    path, frames = _write_tiny_indexed_cndb(tmp_path)

    with h5py.File(path, "r") as h5:
        assert "Header" in h5
        assert "types" in h5
        assert "0" in h5
        assert "1" in h5
        assert "_index" in h5
        assert "_index_offset" in h5.attrs
        header = h5["Header"].attrs
        assert header["format_name"] == "OpenMiChroM-CNDB"
        assert header["format_version"] == "2.0"
        assert header["frame_layout"] == "root_numeric_frames"
        assert header["n_beads"] == 4
        assert header["n_frames"] == 2
        assert bool(header["indexed"]) is True
        assert header["index_format"] == "hdf5-object-offset-json-gzip"

    info = detect_structural_file(path)
    assert info.layout == "openmichrom-cndb-v2"
    assert info.has_embedded_index is True
    assert info.direct_streaming_supported is True

    tools = CndbTools.open(str(path))
    assert (tools.format_name, tools.format_version, tools.format_status) == (
        "OpenMiChroM-CNDB", "2.0", "supported",
    )
    xyz = tools.xyz(frames=[0], beadSelection=range(1, 3))
    assert tools.Nframes == 2
    assert tools.frame_ids == ["0", "1"]
    np.testing.assert_array_equal(xyz[0], frames[0][1:3])


def test_indexed_cndb_v2_streams_through_http_range_server(tmp_path):
    path, frames = _write_tiny_indexed_cndb(tmp_path)

    with _serve_directory(tmp_path) as (base_url, request_log):
        tools = cndbTools.from_remote(h5_url=f"{base_url}/{path.name}")
        xyz = tools.xyz(frames=[0], beadSelection=range(1, 3))
        for opener in (CndbTools.open, CndbTools().load):
            with opener(f"{base_url}/{path.name}") as reader:
                assert reader.format_version == "2.0"
                assert reader.format_status == "supported"
                np.testing.assert_array_equal(reader.xyz(frames=1)[0], frames[1])

    assert xyz.shape == (1, 2, 3)
    np.testing.assert_array_equal(xyz[0], frames[0][1:3])
    stats = tools.stream_stats()
    assert stats["data_bytes_read"] == 2 * 3 * np.dtype("float32").itemsize
    assert any(entry["range"] for entry in request_log if entry["path"].endswith(path.name))
    assert all(
        entry["status"] == 206
        for entry in request_log
        if entry["path"].endswith(path.name) and entry["range"]
    )


def test_indexed_future_version_is_not_treated_as_legacy(tmp_path):
    path, _ = _write_tiny_indexed_cndb(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5["Header"].attrs.modify("format_version", "99.0")
    with pytest.raises(UnsupportedCNDBVersionError):
        CndbTools.open(path)
    with _serve_directory(tmp_path) as (base_url, _):
        with pytest.raises(UnsupportedCNDBVersionError):
            CndbTools.from_remote(f"{base_url}/{path.name}")


def test_annotated_ndb_converts_to_streamable_selected_frames(tmp_path):
    source = tmp_path / "annotated.ndb"
    source.write_text(
        "ASMBLY    galGal5\nMODEL 4\n"
        "CHROM 1 A1 C1 1 1.0 2.0 3.0 1 50000 0.2\n"
        "CHROM 2 B1 C1 2 4.0 5.0 6.0 50001 100000 0.4\nENDMDL\n"
        "MODEL 8\n"
        "CHROM 1 A1 C1 1 7.0 8.0 9.0 1 50000 0.2\n"
        "CHROM 2 B1 C1 2 10.0 11.0 12.0 50001 100000 0.4\nENDMDL\n"
        "LOOPS 1 2\nEND\n"
    )
    path = CndbTools.convert(input_path=source, output_path=tmp_path / "annotated.cndb",
                             input_format="auto", indexed=True, frames=[8])
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["loops"], [[1, 2]])
        assert h5.attrs["genome"] == "galGal5"
        np.testing.assert_allclose(h5["_ndb_sigma"], [0.2, 0.4])
    with _serve_directory(tmp_path) as (base_url, _):
        for source in (path, f"{base_url}/{path.name}"):
            with CndbTools.open(source) as reader:
                assert reader.format_version == "2.0"
                assert reader.frame_ids == ["8"]
                assert reader.types == ["A1", "B1"]
                np.testing.assert_array_equal(reader.xyz()[0], [[7, 8, 9], [10, 11, 12]])
                np.testing.assert_array_equal(reader.genomic_positions, [[1, 50000], [50001, 100000]])


def test_old_style_cndb_without_header_or_index_still_loads(tmp_path):
    path = tmp_path / "old.cndb"
    frame0 = np.arange(12, dtype=np.float32).reshape(4, 3)
    frame1 = 100 + frame0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("types", data=np.array([b"A1", b"B1", b"A1", b"B1"]))
        h5.create_dataset("0", data=frame0)
        h5.create_dataset("1", data=frame1)

    info = detect_structural_file(path)
    assert info.layout == "openmichrom-simple-cndb"
    assert info.has_embedded_index is False

    tools = CndbTools.open(str(path))
    assert tools.Nframes == 2
    np.testing.assert_array_equal(tools.xyz()[0], frame0)
