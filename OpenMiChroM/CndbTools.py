# Copyright (c) 2020-2025 The Center for Theoretical Biological Physics (CTBP) - Rice University
# This file is from the Open-MiChroM project, released under the MIT License.

R"""
The :class:`~.cndbTools` class perform analysis from **cndb** or **ndb** - (Nucleome Data Bank) file format for storing an ensemble of chromosomal 3D structures.
Details about the NDB/CNDB file format can be found at the `Nucleome Data Bank <https://ndb.rice.edu/ndb-format>`__.
"""

import h5py
import numpy as np
import os
import ssl
from urllib.request import Request, urlopen
from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.spatial import distance
from OpenMiChroM._structural_io.converters import convert_structure_file

from OpenMiChroM._cndb_stream.exceptions import CNDBFormatError, FrameNotFoundError
from OpenMiChroM._cndb_stream.remote import validate_http_url
from OpenMiChroM._cndb_stream.version import (
    metadata_from_attrs,
    validate_format_metadata,
)


def _source_kind(source):
    """Classify a path-like or string source without guessing from colons alone."""

    if isinstance(source, os.PathLike):
        return "local"
    if not isinstance(source, str):
        raise TypeError("CNDB source must be a local path or an HTTP(S) URL string.")
    if "://" not in source:
        return "local"
    validate_http_url(source)
    return "remote"


def _metadata_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_list(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [_metadata_value(value) for value in values.tolist()]
    return [_metadata_value(value) for value in list(values)]


_NUMERIC_TYPE_LABELS = {
    0: "A1",
    1: "A2",
    2: "B1",
    3: "B2",
    4: "B3",
    5: "B4",
    6: "NA",
}


def _chromatin_type_list(values):
    labels = _metadata_list(values)
    normalized = []
    for value in labels:
        if isinstance(value, (int, np.integer)):
            normalized.append(_NUMERIC_TYPE_LABELS.get(int(value), str(int(value))))
            continue
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            normalized.append(_NUMERIC_TYPE_LABELS.get(int(value), str(int(value))))
            continue
        text = str(value)
        if text.lstrip("-").isdigit():
            normalized.append(_NUMERIC_TYPE_LABELS.get(int(text), text))
            continue
        normalized.append(text)
    return normalized


def _download_remote_structural_file(
    url,
    *,
    file_size,
    max_download_size_mb,
    download_cache_path,
    timeout=30.0,
    verify_ssl=True,
):
    """Download a remote structural file only after explicit size-limited opt-in."""

    if max_download_size_mb is None:
        raise ValueError(
            "Remote download fallback requires max_download_size_mb. "
            "This prevents accidental large CNDB/HDF5 downloads."
        )
    max_bytes = int(float(max_download_size_mb) * 1024 * 1024)
    if max_bytes < 1:
        raise ValueError("max_download_size_mb is too small to permit any download.")
    if file_size is None:
        raise ValueError(
            "Remote file size is unknown; refusing download fallback. "
            "Use a URL that reports Content-Length or download the file manually."
        )
    if int(file_size) > max_bytes:
        raise ValueError(
            f"Remote file is {int(file_size)} bytes, which exceeds the configured "
            f"download limit of {max_bytes} bytes."
        )

    target_path = Path(download_cache_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    context = None if verify_ssl else ssl._create_unverified_context()
    request = Request(url)
    total = 0
    output = target_path.open("xb")
    try:
        with output:
            with urlopen(request, timeout=timeout, context=context) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            "Remote download exceeded max_download_size_mb; "
                            "aborting before opening the file."
                        )
                    output.write(chunk)
            if total != int(file_size):
                raise ValueError("Downloaded size does not match the reported file size.")
    except BaseException:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise

    return target_path


class _TypeSelections(dict):
    """Canonical chromatin labels with historical byte/integer key lookup."""

    def __missing__(self, key):
        label = _chromatin_type_list([key])[0]
        if label in self:
            return self[label]
        raise KeyError(key)


class _CNDBStreamBackend:
    """Thin adapter around the internal indexed CNDB streaming backend."""

    def __init__(
        self,
        h5_url,
        trajectory=None,
        index_cache_path=None,
        index_path=None,
        index_url=None,
        **kwargs,
    ):
        from OpenMiChroM._cndb_stream import IndexedCNDB

        validate_http_url(h5_url)
        if index_path is not None or index_url is not None:
            if index_cache_path is not None:
                raise ValueError(
                    "index_cache_path applies only to embedded indexes and cannot be "
                    "combined with index_path or index_url."
                )
            self.traj = IndexedCNDB(
                h5_url=h5_url,
                index_path=index_path,
                index_url=index_url,
                trajectory=trajectory,
                **kwargs,
            )
        else:
            self.traj = IndexedCNDB.from_embedded_index(
                h5_url=h5_url,
                trajectory=trajectory,
                index_cache_path=index_cache_path,
                **kwargs,
            )

    @property
    def n_frames(self):
        return self.traj.n_frames

    @property
    def n_beads(self):
        return self.traj.n_beads

    @property
    def frame_ids(self):
        return self.traj.frame_ids

    @property
    def trajectories(self):
        return self.traj.trajectories

    @property
    def current_trajectory(self):
        return self.traj.current_trajectory

    @property
    def types(self):
        return self.traj.types

    @property
    def genomic_positions(self):
        return self.traj.genomic_positions

    @property
    def format_metadata(self):
        return self.traj.format_metadata

    def get_coordinates(self, frame, start=None, stop=None):
        return self.traj.get_coordinates(frame=frame, start=start, stop=stop)

    def close(self):
        self.traj.close()

    def stats(self):
        return {
            "index_bytes_read": self.traj.index_bytes_read,
            "metadata_bytes_read": self.traj.metadata_bytes_read,
            "data_bytes_read": self.traj.data_bytes_read,
            "bytes_read": self.traj.bytes_read,
            "index_cache_hit": self.traj.index_cache_hit,
        }


class _CNDBLocalIndexedBackend(_CNDBStreamBackend):
    """Thin adapter around the internal indexed reader for local nested files."""

    def __init__(self, h5_path, trajectory=None):
        from OpenMiChroM._cndb_stream import IndexedCNDB
        from OpenMiChroM._cndb_stream.index import build_index

        index = build_index(h5_path, trajectories=[trajectory] if trajectory else None)
        self.traj = IndexedCNDB(h5_path=h5_path, _index=index, trajectory=trajectory)


class cndbTools:

    def __init__(self):
        self.Type_conversion = {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
        self.Type_conversionInv = {y:x for x,y in self.Type_conversion.items()}
        self._stream_backend = None
        self.is_remote = False
        self.cndb = None
        self._temporary_conversion = None
        self._temporary_download = None
        self.ChromSeq = []
        self.uniqueChromSeq = set()
        self.dictChromSeq = {}
        self.Nbeads = 0
        self.Nframes = 0
        self.frame_ids = []
        self.trajectories = []
        self.current_trajectory = None
        self.format_name = None
        self.format_version = None
        self.format_status = None
        self.closed = False
        self.genomic_positions = None
        self.types = []
        self._reset_selection_stats()

    def _reset_selection_stats(self):
        self._stream_selection_stats = {
            "coordinate_range_requests": 0,
            "requested_data_bytes": 0,
            "transferred_data_bytes": 0,
            "overfetch_bytes": 0,
            "range_request_count": 0,
            "requested_coordinate_bytes": 0,
            "transferred_coordinate_bytes": 0,
            "overfetch_coordinate_bytes": 0,
            "coalesced_range_count": 0,
            "selection_strategy": None,
        }

    @classmethod
    def from_remote(
        cls,
        h5_url,
        trajectory=None,
        index_cache_path=None,
        index_path=None,
        index_url=None,
        **kwargs,
    ):
        R"""
        Open a remote indexed CNDB/HDF5 file using the internal streaming backend.

        This mode reads embedded HDF5 index metadata and selected coordinate byte
        ranges with HTTP Range requests. Existing local ``load()`` behavior is
        unchanged.

        Args:
            h5_url (str, required):
                HTTP(S) URL to the remote CNDB/HDF5 file.
            trajectory (str, optional):
                Nested trajectory group, for example ``"replica1_chr1"``.
            index_cache_path (str, optional):
                Local JSON.gz cache for the parsed embedded index.
            **kwargs:
                Additional keyword arguments passed to
                ``OpenMiChroM._cndb_stream.IndexedCNDB.from_embedded_index``.
        """
        tool = cls()
        tool._stream_backend = _CNDBStreamBackend(
            h5_url=h5_url,
            trajectory=trajectory,
            index_cache_path=index_cache_path,
            index_path=index_path,
            index_url=index_url,
            **kwargs,
        )
        tool._load_backend_metadata(is_remote=True)
        return tool

    def _set_types(self, values):
        self.ChromSeq = _chromatin_type_list(values)
        self.types = self.ChromSeq
        self.uniqueChromSeq = set(self.ChromSeq)
        self.dictChromSeq = _TypeSelections()
        for i, label in enumerate(self.ChromSeq):
            self.dictChromSeq.setdefault(label, []).append(i)

    def _load_backend_metadata(self, is_remote):
        backend = self._stream_backend
        self._reset_selection_stats()
        self.is_remote = is_remote
        self.closed = False
        self.cndb = None
        self._set_types(backend.types)
        self.Nbeads, self.Nframes = backend.n_beads, backend.n_frames
        self.frame_ids = backend.frame_ids
        self.trajectories = backend.trajectories
        self.current_trajectory = backend.current_trajectory
        self.genomic_positions = backend.genomic_positions
        self._set_format_metadata(getattr(backend, "format_metadata", {}))

    @classmethod
    def open(
        cls,
        source,
        trajectory=None,
        index_cache_path=None,
        mode="auto",
        allow_remote_without_index=False,
        allow_download=False,
        max_download_size_mb=None,
        download_cache_path=None,
        **kwargs,
    ):
        R"""
        Open a local or remote structural trajectory file.

        This is a conservative routing layer. Local CNDB files continue to use
        the existing ``load()`` implementation. Remote HDF5/CNDB/SW files use
        streaming only when an embedded index and direct coordinate byte reads
        are detected. Non-indexed remote HDF5 files are rejected by default.
        Set ``allow_download=True`` with ``max_download_size_mb`` to explicitly
        download a small remote file and open it locally.
        """
        if mode != "auto":
            raise ValueError("Only mode='auto' is currently supported by CndbTools.open().")

        from OpenMiChroM._structural_io import detect_structural_file

        detect_kwargs = {
            key: kwargs[key]
            for key in ("timeout", "sample_size", "verify_ssl")
            if key in kwargs
        }
        stream_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"sample_size", "verify_ssl"}
        }

        kind = _source_kind(source)
        if kind == "remote" and any(kwargs.get(key) is not None for key in ("index_path", "index_url")):
            return cls.from_remote(source, trajectory=trajectory, index_cache_path=index_cache_path, **stream_kwargs)
        if kind == "local" and (index_cache_path is not None or any(
            kwargs.get(key) is not None for key in ("index_path", "index_url")
        )):
            raise ValueError("Remote index options cannot be used with a local path.")
        info = detect_structural_file(source, **detect_kwargs)
        if info.is_remote:
            if info.detected_hdf5 and info.has_embedded_index and info.direct_streaming_supported:
                return cls.from_remote(
                    h5_url=source,
                    trajectory=trajectory,
                    index_cache_path=index_cache_path,
                    **stream_kwargs,
                )
            if allow_remote_without_index:
                raise ValueError(
                    "Remote fallback reading is not implemented yet. The file was detected "
                    f"as file_type={info.file_type!r}, layout={info.layout!r}."
                )
            if allow_download:
                tmp = None
                if download_cache_path is None:
                    tmp = TemporaryDirectory(prefix="openmichrom-structural-")
                    suffix = Path(str(source).split("?", 1)[0]).suffix or ".structural"
                    download_cache_path = Path(tmp.name) / ("download" + suffix)
                try:
                    local_path = _download_remote_structural_file(
                        source, file_size=info.file_size, max_download_size_mb=max_download_size_mb,
                        download_cache_path=download_cache_path, timeout=detect_kwargs.get("timeout", 30.0),
                        verify_ssl=detect_kwargs.get("verify_ssl", True),
                    )
                    tool = cls.open(local_path, trajectory=trajectory, mode=mode)
                except BaseException:
                    if tmp is not None:
                        tmp.cleanup()
                    raise
                tool._temporary_download = tmp
                tool.downloaded_remote_source = source
                tool.downloaded_remote_path = str(local_path)
                return tool
            raise ValueError(
                "Remote structural file cannot be streamed safely. "
                f"file_type={info.file_type!r}, layout={info.layout!r}, "
                f"range_supported={info.range_supported!r}, "
                f"has_embedded_index={info.has_embedded_index!r}, "
                f"direct_streaming_supported={info.direct_streaming_supported!r}. "
                "Use a file with an embedded index, or download/index the file locally."
            )

        if info.detected_hdf5 and info.file_type in {"cndb", "hdf5", "sw"}:
            return cls().load(source, trajectory=trajectory)
        if info.detected_text_ndb:
            if trajectory is not None:
                raise ValueError("Text NDB does not contain nested trajectories.")
            from OpenMiChroM._structural_io.readers import NDBTextReader

            tool = cls()
            tool._stream_backend = NDBTextReader(source)
            tool._load_backend_metadata(is_remote=False)
            return tool
        raise ValueError(
            f"Could not open structural file {source!r}. "
            f"Detected file_type={info.file_type!r}, layout={info.layout!r}."
        )
    
    def load(
        self,
        fileName=None,
        *,
        filename=None,
        trajectory=None,
        index_cache_path=None,
        index_path=None,
        index_url=None,
        **kwargs,
    ):
        R"""
        Receives the path to **cndb** or **ndb** file to perform analysis.
        
        Args:
            fileName (file, required):
                Path to a CNDB or NDB file. For an NDB input, a sibling CNDB
                is created if absent. If it already exists, the conversion is
                temporary: loading an NDB never replaces an existing trajectory.
        """
        if fileName is not None and filename is not None:
            raise TypeError("Pass only one of fileName or filename.")
        source = fileName if fileName is not None else filename
        if source is None:
            raise TypeError("load() requires a local path or HTTP(S) URL.")
        if _source_kind(source) == "remote":
            remote = type(self).from_remote(
                str(source),
                trajectory=trajectory,
                index_cache_path=index_cache_path,
                index_path=index_path,
                index_url=index_url,
                **kwargs,
            )
            self.close()
            self.__dict__.update(remote.__dict__)
            return self
        if any(value is not None for value in (index_cache_path, index_path, index_url)):
            raise ValueError("Remote index and trajectory options cannot be used with a local path.")
        if kwargs:
            raise TypeError(f"Unexpected local CNDB options: {sorted(kwargs)}")

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local CNDB/NDB file does not exist: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Local CNDB/NDB source is not a regular file: {path}")
        self.close()
        try:
            if path.suffix.lower() == ".ndb":
                try:
                    path = self.ndb_to_cndb(path, path.with_suffix(".cndb"))
                except FileExistsError:
                    self._temporary_conversion = TemporaryDirectory()
                    dest = Path(self._temporary_conversion.name) / "trajectory.cndb"
                    path = self.ndb_to_cndb(path, dest)
            try:
                self.cndb = h5py.File(path, "r")
            except (OSError, ValueError) as exc:
                raise CNDBFormatError(f"Could not open {path} as a CNDB/HDF5 file: {exc}") from exc
            if not any(str(key).isdigit() for key in self.cndb) and any(
                isinstance(group, h5py.Group) and "spatial_position" in group
                for group in self.cndb.values()
            ):
                self.cndb.close()
                self.cndb = None
                self._stream_backend = _CNDBLocalIndexedBackend(path, trajectory=trajectory)
                self._load_backend_metadata(is_remote=False)
            else:
                if trajectory is not None:
                    raise ValueError("A trajectory name is only valid with a nested CNDB/SWB file.")
                self._load_local_metadata(path)
        except BaseException:
            self.close()
            raise
        return self

    def _load_local_metadata(self, path):
        self._stream_backend = None
        self.is_remote = False
        self.closed = False

        if "types" not in self.cndb or not isinstance(self.cndb["types"], h5py.Dataset):
            raise CNDBFormatError(f"CNDB file {path} is missing the required 'types' dataset.")

        metadata = metadata_from_attrs(self.cndb.attrs)
        if not any(metadata.values()) and "Header" in self.cndb:
            header = self.cndb["Header"]
            if isinstance(header, h5py.Group):
                metadata = metadata_from_attrs(header.attrs)
        self._set_format_metadata(validate_format_metadata(metadata, source=str(path)))

        self._set_types(self.cndb["types"])
        self.genomic_positions = None
        for key in ("_ndb_genomic_intervals", "genomic_position"):
            if key in self.cndb:
                self.genomic_positions = np.asarray(self.cndb[key])
                break
        self.Nbeads = len(self.ChromSeq)
        self.frame_ids = sorted(
            [key for key in self.cndb.keys() if str(key).isdigit()],
            key=lambda frame: int(frame),
        )
        self.Nframes = len(self.frame_ids)
        self.trajectories = []
        self.current_trajectory = None
        
    def _set_format_metadata(self, metadata):
        self.format_name = metadata.get("cndb_format")
        self.format_version = metadata.get("cndb_format_version")
        self.format_status = metadata.get("cndb_format_status")

    def close(self):
        """Close local or remote CNDB resources; safe to call repeatedly."""

        backend = getattr(self, "_stream_backend", None)
        if backend is not None and hasattr(backend, "close"):
            backend.close()
        local_file = getattr(self, "cndb", None)
        if local_file is not None:
            try:
                local_file.close()
            finally:
                self.cndb = None
        if self._temporary_conversion is not None:
            self._temporary_conversion.cleanup()
            self._temporary_conversion = None
        if self._temporary_download is not None:
            self._temporary_download.cleanup()
            self._temporary_download = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    

    @staticmethod
    def convert(source=None, output=None, *, input_path=None, output_path=None, **kwargs):
        """Convert a local trajectory, inferring the formats from the paths.

        The eight original NDB converter routes preserve supported annotations.
        NDB to CNDB also supports ``indexed=True`` and frame selection without
        discarding annotations. Structural options such as bead subsetting
        and additional routes (e.g. SWB or PDB to CNDB) use the structural
        converter. For NDB or annotated flat CNDB input and NDB/CNDB output,
        a bead window retains the selected types, chains and chain indices,
        genomic intervals, and sigma values. Only loops with both endpoints
        inside the window survive; their bead numbers are relative to the
        new window. This export does not preserve assembly or title metadata.
        """
        from OpenMiChroM.Converters import FORMAT_ALIASES, SUPPORTED_CONVERSIONS, convert

        if source is not None and input_path is not None or output is not None and output_path is not None:
            raise TypeError("Specify only one name for each input and output path.")
        source = source if source is not None else input_path
        output = output if output is not None else output_path
        if source is None:
            raise TypeError("convert() requires a local input path.")
        input_fmt = kwargs.get("input_format") or kwargs.get("from_format") or Path(source).suffix
        if input_fmt == "auto" and Path(source).suffix.lower() == ".ndb":
            input_fmt = Path(source).suffix
            kwargs.pop("input_format", None)
        output_fmt = kwargs.get("output_format") or kwargs.get("to_format") or (Path(output).suffix if output else "")
        route = tuple(str(fmt).lower().lstrip(".") for fmt in (input_fmt, output_fmt))
        route = tuple(FORMAT_ALIASES.get(fmt, fmt) for fmt in route)
        structural = {"indexed", "metadata", "trajectory", "frames", "start", "stop", "max_frames",
                      "max_memory_mb", "allow_large", "pdb_atom_name", "pdb_residue_name",
                      "pdb_chain_id", "pdb_element"}
        structural_options = structural.intersection(kwargs)
        if route == ("ndb", "cndb"):
            structural_options -= {"indexed", "metadata", "frames", "max_frames"}
        if route in SUPPORTED_CONVERSIONS and not structural_options:
            return convert(source, output, **kwargs)
        for alias, key in (("from_format", "input_format"), ("to_format", "output_format")):
            if alias in kwargs:
                value = kwargs.pop(alias)
                if key in kwargs and kwargs[key] != value:
                    raise ValueError(f"{alias} and {key} disagree.")
                kwargs[key] = value
        return convert_structure_file(source, output, **kwargs)

    @staticmethod
    def ndb_to_cndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import ndb_to_cndb

        return ndb_to_cndb(source, output, **kwargs)

    @staticmethod
    def cndb_to_ndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import cndb_to_ndb

        return cndb_to_ndb(source, output, **kwargs)

    @staticmethod
    def ndb_to_pdb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import ndb_to_pdb

        return ndb_to_pdb(source, output, **kwargs)

    @staticmethod
    def pdb_to_ndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import pdb_to_ndb

        return pdb_to_ndb(source, output, **kwargs)

    @staticmethod
    def ndb_to_spw(source, output=None, **kwargs):
        from OpenMiChroM.Converters import ndb_to_spw

        return ndb_to_spw(source, output, **kwargs)

    @staticmethod
    def spw_to_ndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import spw_to_ndb

        return spw_to_ndb(source, output, **kwargs)

    @staticmethod
    def gro_to_ndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import gro_to_ndb

        return gro_to_ndb(source, output, **kwargs)

    @staticmethod
    def csv_to_ndb(source, output=None, **kwargs):
        from OpenMiChroM.Converters import csv_to_ndb

        return csv_to_ndb(source, output, **kwargs)

    def ndb2cndb(self, fileName):
        R"""Compatibility wrapper for the historical extensionless NDB stem API.

        New code should call :meth:`ndb_to_cndb`, which returns a
        :class:`pathlib.Path` and refuses to replace existing output by default.
        This wrapper retains the original string return type and replacement
        behavior for callers that explicitly request this legacy conversion.
        """

        source = Path(fileName).expanduser()
        if source.suffix.lower() != ".ndb":
            source = Path(str(source) + ".ndb")
        destination = source.with_suffix(".cndb")
        return str(self.ndb_to_cndb(source, destination, overwrite=True))
    
    def xyz(self, frames=None, beadSelection=None, XYZ=(0, 1, 2), coalesce=True, max_gap=0, max_ranges=128):
        R"""
        Get the selected beads' 3D position from a **cndb** or **ndb** for multiple frames.
        
        Args:
            frames (list, required):
                Define which frames to extract the position of the selected beads. The list should include all frames to iterate over, or a iterator that goes over the desired frames. (Default value: `None`, all frames)
            beadSelection (list of ints, required):
                List of beads to extract the 3D position for each frame. The list is defined by `beadSelection=[0,1,2,...,N-1]`. (Default value: `None`, all beads) 
            XYZ (list, required):
                List of the axis in the Cartesian coordinate system that the position of the bead will get extracted for each frame. The list is defined by `XYZ=[0,1,2]`. where 0, 1 and 2 are the axis X, Y and Z, respectively. (Default value: `XYZ=[0,1,2]`) 
    
        Returns:
            (:math:`N_{frames}`, :math:`N_{beads}`, 3) :class:`numpy.ndarray`: Returns an array of the 3D position of the selected beads for different frames.
        """
        if self.closed:
            raise ValueError("Cannot read coordinates from closed CNDBTools resources.")
        if self._stream_backend is not None:
            return self._xyz_stream(frames, beadSelection, XYZ, coalesce, max_gap, max_ranges)
        if self.cndb is None or self.closed:
            raise ValueError("Load a CNDB file before requesting coordinates.")

        selected_frames = self._normalize_frames(frames)
        selection = self._normalize_bead_selection(beadSelection)
        axes = self._normalize_axes(XYZ)
        if not selected_frames:
            return np.empty((0, selection.size, axes.size), dtype=float)

        frame_list = []
        for frame_id in selected_frames:
            try:
                dataset = self.cndb[frame_id]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != (self.Nbeads, 3):
                    raise CNDBFormatError(
                        f"Frame {frame_id!r} is not a coordinate dataset shaped ({self.Nbeads}, 3)."
                    )
                frame_list.append(np.take(np.take(np.asarray(dataset), selection, axis=0), axes, axis=1))
            except (OSError, ValueError) as exc:
                raise CNDBFormatError(f"Could not read CNDB frame {frame_id!r}: {exc}") from exc
        return np.asarray(frame_list)

    def _xyz_stream(
        self,
        frames=None,
        beadSelection=None,
        XYZ=(0, 1, 2),
        coalesce=True,
        max_gap=0,
        max_ranges=128,
    ):
        R"""
        Streaming implementation of ``xyz`` using the internal CNDB backend.

        Contiguous bead ranges are read with exact byte ranges. Non-contiguous
        selections are coalesced into ordered ranges and reassembled in the
        requested order. If too many sparse ranges would be needed, the reader
        falls back to the smallest enclosing bead interval and subsets in
        memory.
        """
        frame_list = []
        frames = self._normalize_frames(frames)
        if beadSelection is not None:
            beadSelection = self._normalize_bead_selection(beadSelection)

        ranges, post_selection, requested_rows, strategy = self._stream_bead_plan(
            beadSelection,
            coalesce=coalesce,
            max_gap=max_gap,
            max_ranges=max_ranges,
        )
        axis_selection = self._normalize_axes(XYZ)
        if not frames:
            return np.empty((0, requested_rows, axis_selection.size), dtype=float)

        for frame in frames:
            data_bytes_before = self.stream_data_bytes_read
            coords = self._stream_read_ranges(frame, ranges)
            data_bytes_after = self.stream_data_bytes_read
            if post_selection is not None:
                coords = np.take(coords, post_selection, axis=0)
            frame_list.append(np.take(coords, axis_selection, axis=1))
            if coords.size:
                itemsize = coords.dtype.itemsize
            else:
                itemsize = np.dtype(np.float32).itemsize
            transferred_rows = sum(stop - start for start, stop in ranges)
            planned_transferred_bytes = transferred_rows * 3 * itemsize
            measured_transferred_bytes = max(0, data_bytes_after - data_bytes_before)
            transferred_bytes = max(planned_transferred_bytes, measured_transferred_bytes)
            requested_bytes = requested_rows * 3 * itemsize
            self._stream_selection_stats["coordinate_range_requests"] += len(ranges)
            self._stream_selection_stats["requested_data_bytes"] += requested_bytes
            self._stream_selection_stats["transferred_data_bytes"] += transferred_bytes
            self._stream_selection_stats["overfetch_bytes"] += max(
                0,
                transferred_bytes - requested_bytes,
            )
            self._stream_selection_stats["range_request_count"] += len(ranges)
            self._stream_selection_stats["requested_coordinate_bytes"] += requested_bytes
            self._stream_selection_stats["transferred_coordinate_bytes"] += transferred_bytes
            self._stream_selection_stats["overfetch_coordinate_bytes"] += max(
                0,
                transferred_bytes - requested_bytes,
            )
            if strategy == "coalesced-ranges":
                self._stream_selection_stats["coalesced_range_count"] += len(ranges)
            self._stream_selection_stats["selection_strategy"] = strategy
        return(np.array(frame_list))

    def _stream_read_ranges(self, frame, ranges):
        blocks = []
        for start, stop in ranges:
            blocks.append(self._stream_backend.get_coordinates(frame=frame, start=start, stop=stop))
        if not blocks:
            return np.empty((0, 3), dtype=np.float32)
        if len(blocks) == 1:
            return blocks[0]
        return np.concatenate(blocks, axis=0)

    def _stream_bead_plan(self, beadSelection, coalesce=True, max_gap=0, max_ranges=128):
        if beadSelection is None:
            return [(0, self.Nbeads)], None, self.Nbeads, "full-frame"

        selection = beadSelection
        if selection.size == 0:
            return [], None, 0, "empty"

        sorted_selection = np.sort(selection)
        start = int(sorted_selection[0])
        stop = int(sorted_selection[-1]) + 1

        if np.array_equal(selection, np.arange(start, stop)):
            return [(start, stop)], None, int(selection.size), "single-range"

        if coalesce:
            from OpenMiChroM._structural_io import coalesce_indices

            ranges = coalesce_indices(selection, max_gap=max_gap)
            if len(ranges) <= max_ranges:
                offsets = {}
                cursor = 0
                for range_start, range_stop in ranges:
                    for absolute in range(range_start, range_stop):
                        offsets[absolute] = cursor + absolute - range_start
                    cursor += range_stop - range_start
                post_selection = np.array([offsets[int(index)] for index in selection], dtype=int)
                return ranges, post_selection, int(selection.size), "coalesced-ranges"

        return [(start, stop)], selection - start, int(selection.size), "enclosing-range"

    def _normalize_frames(self, frames):
        if frames is None:
            selected = list(self.frame_ids)
        elif isinstance(frames, (int, np.integer, str)):
            selected = [str(frames)]
        else:
            selected = [str(frame) for frame in frames]
        available = set(self.frame_ids)
        selected = [
            f"t_{frame}" if frame not in available and f"t_{frame}" in available else frame
            for frame in selected
        ]
        missing = [frame for frame in selected if frame not in available]
        if missing:
            raise FrameNotFoundError(
                f"Frames {missing!r} were not found. Available frames include {self.frame_ids[:10]!r}."
            )
        return selected

    def _normalize_bead_selection(self, beadSelection):
        if beadSelection is None:
            return np.arange(self.Nbeads, dtype=int)
        if isinstance(beadSelection, slice):
            return np.arange(self.Nbeads, dtype=int)[beadSelection]
        selection = np.atleast_1d(np.asarray(beadSelection, dtype=int))
        if np.any(selection < 0):
            selection = np.where(selection < 0, selection + self.Nbeads, selection)
        if np.any(selection < 0) or np.any(selection >= self.Nbeads):
            raise IndexError(
                f"Bead selection contains an index outside [0, {self.Nbeads - 1}]."
            )
        return selection

    @staticmethod
    def _normalize_axes(XYZ):
        axes = np.atleast_1d(np.asarray(XYZ, dtype=int))
        if np.any(axes < 0) or np.any(axes > 2):
            raise IndexError("XYZ axes must contain only 0, 1, or 2.")
        return axes

    @property
    def stream_data_bytes_read(self):
        if self._stream_backend is None:
            return 0
        backend = getattr(self._stream_backend, "traj", self._stream_backend)
        return getattr(backend, "data_bytes_read", 0)

    @property
    def stream_index_bytes_read(self):
        if self._stream_backend is None:
            return 0
        backend = getattr(self._stream_backend, "traj", self._stream_backend)
        return getattr(backend, "index_bytes_read", 0)

    @property
    def stream_metadata_bytes_read(self):
        if self._stream_backend is None:
            return 0
        backend = getattr(self._stream_backend, "traj", self._stream_backend)
        return getattr(backend, "metadata_bytes_read", 0)

    @property
    def stream_bytes_read(self):
        if self._stream_backend is None:
            return 0
        backend = getattr(self._stream_backend, "traj", self._stream_backend)
        return getattr(backend, "bytes_read", 0)

    @property
    def stream_index_cache_hit(self):
        if self._stream_backend is None:
            return False
        backend = getattr(self._stream_backend, "traj", self._stream_backend)
        return getattr(backend, "index_cache_hit", False)

    def stream_stats(self):
        R"""
        Return byte-accounting diagnostics for remote streaming mode.

        Local h5py mode returns zeros and ``index_cache_hit=False``.
        """
        if self._stream_backend is None:
            return {
                "index_bytes_read": 0,
                "metadata_bytes_read": 0,
                "data_bytes_read": 0,
                "bytes_read": 0,
                "index_cache_hit": False,
                "coordinate_range_requests": 0,
                "requested_data_bytes": 0,
                "transferred_data_bytes": 0,
                "overfetch_bytes": 0,
                "range_request_count": 0,
                "requested_coordinate_bytes": 0,
                "transferred_coordinate_bytes": 0,
                "overfetch_coordinate_bytes": 0,
                "coalesced_range_count": 0,
                "selection_strategy": None,
            }
        stats = self._stream_backend.stats()
        stats.update(self._stream_selection_stats)
        return stats
    
    
    
#########################################################################################
#### Analysis start here!
#########################################################################################

    def compute_Orientation_OP(self, xyz, chrom_start=0, chrom_end=1000, vec_length=4,
                               *, normalization="legacy"):
        """Compute tangent correlations for one frame of Cartesian coordinates.

        The selected region is a zero-based half-open interval (the end is
        clipped to the frame length). A tangent joins beads i and i+vec_length.
        Use ``normalization="pairs"`` for zero-based bead lags and the mean over
        all available pairs at each lag; the zero-lag value is one. The default
        ``"legacy"`` retains the original offset axis, omitted first tangent
        and weighting so previously reported curves can be reproduced.
        """
        from operator import index

        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("xyz must be one finite frame of shape (beads, 3)")
        start, width = index(chrom_start), index(vec_length)
        stop = len(xyz) if chrom_end is None else min(index(chrom_end), len(xyz))
        if not 0 <= start < stop or not 0 < width < stop - start:
            raise ValueError("The selected region must contain more beads than vec_length")
        if normalization not in ("legacy", "pairs"):
            raise ValueError("normalization must be 'legacy' or 'pairs'")
        region = xyz[start:stop]
        tangent = region[width:] - region[:-width]
        if normalization == "legacy":
            tangent = tangent[1:]
        norms = np.linalg.norm(tangent, axis=1)
        if np.any(norms == 0):
            raise ValueError("A zero-length tangent has no defined orientation")
        tangent = tangent / norms[:, None]
        lags = np.arange(len(tangent))
        sums = np.array([np.einsum("ij,ij->", tangent[:len(tangent)-d], tangent[d:])
                         for d in lags])
        if normalization == "legacy":
            return lags + 1, sums / (len(region) - lags)
        return lags, sums / (len(tangent) - lags)

    def compute_FFT_from_Oij(self, Oijy, lowcut=1, highcut=500, order=5):
        """Return the historical half-spectrum in cycles per bead and normalized FFT magnitude.

        Cutoffs are in cycles per input record, as in the original routine.
        Set both to None for an unfiltered transform. An upper cutoff at or
        above the Nyquist frequency leaves the upper band unrestricted.
        """
        from scipy.signal import butter, lfilter

        signal = np.asarray(Oijy, dtype=float)
        if signal.ndim != 1 or len(signal) < 2 or not np.isfinite(signal).all():
            raise ValueError("Oijy must be a finite 1D signal with at least two samples")
        n = len(signal)
        if highcut is not None and (not np.isfinite(highcut) or highcut <= 0):
            raise ValueError("highcut must be positive and finite")
        if lowcut is not None and not 0 < lowcut < n // 2:
            raise ValueError("lowcut must be positive and below the Nyquist frequency")
        if highcut is not None and lowcut is not None and highcut <= lowcut:
            raise ValueError("highcut must exceed lowcut")
        # Retain the historical integer Nyquist convention for filtered curves.
        nyquist = n // 2
        upper = highcut if highcut is not None and highcut < nyquist else None
        if lowcut is not None or upper is not None:
            if lowcut is not None and upper is not None:
                cutoff, kind = [lowcut / nyquist, upper / nyquist], "bandpass"
            elif lowcut is not None:
                cutoff, kind = lowcut / nyquist, "highpass"
            else:
                cutoff, kind = upper / nyquist, "lowpass"
            b, a = butter(order, cutoff, btype=kind)
            signal = lfilter(b, a, signal)
        return np.arange(n // 2) / n, np.abs(np.fft.rfft(signal)[:n // 2]) / n
    

    def compute_Chirality(self,xyz,neig_beads=4):
        R"""
        Calculates the Chirality parameter :math:`\Psi`. Details are decribed in "Zhang, B. and Wolynes, P.G., 2016. Shape transitions and chiral symmetry breaking in the energy landscape of the mitotic chromosome. Physical review letters, 116(24), p.248101."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.
            neig_beads (int, required):
                Number of neighbor beads to consider in the calculation (Default value = 4).  
                       
        Returns:
            :class:`numpy.ndarray`:
                Returns the Chirality parameter :math:`\Psi` for each bead.
        """
        Psi=[]
        for frame in range(len(xyz)):
            XYZ = xyz[frame]
            Psi_per_bead=[]
            for i in range(0,np.shape(XYZ)[0] - np.ceil(1.25*neig_beads).astype('int')):
                a=i
                b=int(np.round(i+0.5*neig_beads))
                c=int(np.round(i+0.75*neig_beads))
                d=int(np.round(i+1.25*neig_beads))

                AB = XYZ[b]-XYZ[a]
                CD = XYZ[d]-XYZ[c]
                E = (XYZ[b]-XYZ[a])/2.0 + XYZ[a]
                F = (XYZ[d]-XYZ[c])/2.0 + XYZ[c]
                Psi_per_bead.append(np.dot((F-E),np.cross(CD,AB))/(np.linalg.norm(F-E)*np.linalg.norm(AB)*np.linalg.norm(CD)))
            Psi.append(Psi_per_bead)
            
        return np.asarray(Psi)

    def compute_RG(self, xyz):
        R"""
        Calculate the radius of gyration for each frame.

        Args:
            xyz (numpy.ndarray): Coordinates with shape ``(frames, beads, 3)``.

        Returns:
            numpy.ndarray: Radius of gyration for each frame in units of
            :math:`\sigma`.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        xyz_rel_to_cm= xyz - np.tile(rcm,(xyz.shape[1],1))
        rg=np.sqrt(np.mean(np.linalg.norm(xyz_rel_to_cm,axis=2)**2,axis=1))
        return rg

    def compute_GyrTensorEigs(self, xyz):
        R"""
        Calculate the three sorted eigenvalues of each gyration tensor.

        Args:
            xyz (numpy.ndarray): Coordinates with shape ``(frames, beads, 3)``.

        Returns:
            numpy.ndarray: Sorted eigenvalues with shape ``(frames, 3)``.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        sorted_eigenvals=[]
        for frame in xyz-rcm:
            gyr=np.matmul(np.transpose(frame),frame)/xyz.shape[1]
            sorted_eigenvals.append(np.sort(np.linalg.eig(gyr)[0]))
        return np.array(sorted_eigenvals)


    def compute_MSD(self,xyz):
        R"""
        Calculate mean-squared displacement with a fast Fourier transform.

        Args:
            xyz (numpy.ndarray): Coordinates with shape ``(frames, beads, 3)``.

        Returns:
            numpy.ndarray: MSD for each bead, with shape ``(beads, frames)``.

        """
        
        msd=[self._msd_fft(xyz[:,mono_id,:]) for mono_id in range(xyz.shape[1])]
        return np.array(msd)
        

    def _autocorrFFT(self, x):
        R"""
        Internal function. 
        """
        N=len(x)
        F = np.fft.fft(x,n=2*N)  #2*N because of zero-padding
        res = np.fft.ifft(F * F.conjugate()) #autocorrelation using Weiner Kinchin theorem
        res = (res[:N]).real   
        return res/(N-np.arange(0,N)) #this is the normalized autocorrelation

        #r is an (T,3) ndarray: [time stamps,dof]
    def _msd_fft(self, r):
        R"""
        Internal function. 
        """
        N=len(r)
        D=np.square(r).sum(axis=1)
        D=np.append(D,0)
        S2=sum([self._autocorrFFT(r[:, i]) for i in range(r.shape[1])])
        Q=2*D.sum()
        S1=[]
        for m in range(N):
            Q=Q-D[m-1]-D[N-m]
            S1.append(Q/(N-m))
        return np.array(S1) - 2*S2

    def compute_RadNumDens(self, xyz, dr=1.0, ref='centroid', center=None, *, normalization="shell-volume"):

        R"""
        Calculates the radial number density of monomers; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of monomers.
        
        Args:
            xyz (numpy.ndarray): Coordinates with shape ``(frames, beads, 3)``.

            dr (float, required):
                mesh size of radius for calculating the radial distribution. 
                can be arbitrarily small, but leads to empty bins for small values.
                bins are computed from the maximum values of radius and dr.
            
            ref (string):
                defines reference for centering the disribution. It can take three values:
                
                'origin': radial distance is calculated from the center

                'centroid' (default value): radial distributioin is computed from the centroid of the cloud of points at each time step

                'custom': user defined center of reference. 'center' is required to be specified when 'custom' reference is chosen

            center (list of float, len 3):
                defines the reference point in custom reference. required when ref='custom'
            normalization (str):
                The default, ``"shell-volume"``, includes the furthest bead
                even when ``dr > 1``. Select ``"legacy"`` to reproduce the
                historical bin edges.
                       
        Returns:
            num_density (numpy.ndarray):
                the number density
            
            bins (numpy.ndarray):
                bins corresponding to the number density

        """

        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or not xyz.shape[0] or not xyz.shape[1]:
            raise ValueError("xyz must contain frames and beads and have shape (frames, beads, 3).")
        if not np.isfinite(dr) or dr <= 0:
            raise ValueError("dr must be finite and positive.")
        if ref == 'origin':
            origin = 0
        elif ref == 'centroid':
            origin = xyz.mean(axis=1, keepdims=True)
        elif ref == 'custom':
            origin = np.asarray(center, dtype=float)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise ValueError("center must contain three finite coordinates for ref='custom'.")
        else:
            raise ValueError("ref must be 'origin', 'centroid', or 'custom'.")
        rad_vals = np.linalg.norm(xyz - origin, axis=2)
        if not np.all(np.isfinite(rad_vals)):
            raise ValueError("xyz must contain finite coordinates.")
        if normalization == "legacy":
            edges = np.arange(0, rad_vals.max() + 1, dr)
        elif normalization == "shell-volume":
            shell_count = max(1, int(np.ceil(rad_vals.max() / dr)))
            edges = np.arange(shell_count + 1) * dr
            edges[-1] = max(edges[-1], rad_vals.max())
        else:
            raise ValueError("normalization must be 'legacy' or 'shell-volume'.")
        rdp_hist, bin_edges = np.histogram(rad_vals, bins=edges)

        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        bin_vols = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        num_density = rdp_hist/(xyz.shape[0]*bin_vols)

        return (num_density, bin_mids)

        
    def compute_RDP(self, xyz, beadSelection=None, radius=20.0, bins=200, *, normalization="shell-volume"):
        R"""
        Calculate mean bead number per shell volume about the per-frame centroid.
        
        Args:
            xyz (numpy.ndarray): Coordinates with shape ``(frames, beads, 3)``.
            beadSelection (numpy.ndarray): Bead indices included in the RDP.
            radius (float, required):
                Radius of the sphere in units of :math:`\sigma` to be considered in the calculations. The radius value should be modified depending on your simulated chromosome length. (Default value = 20.0).
            bins (int, required):
                Number of slices to be considered as spherical shells. (Default value = 200).
            normalization (str):
                The default, ``"shell-volume"``, uses exact shell volumes and
                matching midpoints, including beads at zero and at ``radius``.
                Select ``"legacy"`` to reproduce the historical thin-shell
                approximation and radii, starting at one bin width.
                       
        Returns:
            tuple: Shell radii in units of :math:`\sigma` and the RDP for each
            spherical shell.
        """
        
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or not xyz.shape[0] or not xyz.shape[1]:
            raise ValueError("xyz must contain frames and beads and have shape (frames, beads, 3).")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive.")
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError("bins must be a positive integer.")
        if not np.all(np.isfinite(xyz)):
            raise ValueError("xyz must contain finite coordinates.")
        centered = xyz - xyz.mean(axis=1, keepdims=True)
        if beadSelection is not None:
            centered = centered[:, beadSelection]
        distances = np.linalg.norm(centered, axis=-1)
        if normalization == "legacy":
            width = radius / bins
            r = width
            values = np.zeros(bins)
            for k in range(bins):
                if r > radius:
                    break
                count = np.count_nonzero((distances >= r) & (distances < r + width))
                values[k] = count / (len(xyz) * 4 * np.pi * width * r**2)
                r += width
            return np.arange(0, int(radius + width), width).tolist(), values
        if normalization != "shell-volume":
            raise ValueError("normalization must be 'legacy' or 'shell-volume'.")
        edges = np.linspace(0, radius, bins + 1)
        counts, _ = np.histogram(distances, bins=edges)
        volumes = (4 * np.pi / 3) * np.diff(edges**3)
        return (edges[1:] + edges[:-1]) / 2, counts / (len(xyz) * volumes)

    def traj2HiC(self, xyz, mu=3.22, rc = 1.78):
        R"""
        Calculates the *in silico* Hi-C maps (contact probability matrix) using a chromatin dyamics trajectory.   
        
        The parameters :math:`\mu` (mu) and rc are part of the probability of crosslink function :math:`f(r_{i,j}) = \frac{1}{2}\left( 1 + tanh\left[\mu(r_c - r_{i,j}\right] \right)`, where :math:`r_{i,j}` is the spatial distance between loci (beads) *i* and *j*.
        
        Args:

            mu (float, required):
                Parameter in the probability of crosslink function. (Default value = 3.22).
            rc (float, required):
                Parameter in the probability of crosslink function, :math:`f(rc) = 0.5`. (Default value = 1.78).
        
         Returns:
            :math:`(N, N)` :class:`numpy.ndarray`:
                Returns the *in silico* Hi-C maps (contact probability matrix).
        """
        def calc_prob(data, mu, rc):
            return 0.5 * (1.0 + np.tanh(mu * (rc - distance.cdist(data, data, 'euclidean'))))
        
        size = len(xyz[0])
        P = np.zeros((size, size))
        Ntotal = 0
        
        for i in range(len(xyz)):
            data = xyz[i]
            P += calc_prob(data, mu, rc)
            Ntotal += 1
            if i % 500 == 0:
                print("Reading frame {:} of {:}".format(i, len(xyz)))
        
        return(np.divide(P , Ntotal))    
            
        
    def __repr__(self):
        return '<{0}.{1} object at {2}>\nCndb file has {3} frames, with {4} beads and {5} types '.format(
      self.__module__, type(self).__name__, hex(id(self)), self.Nframes, self.Nbeads, self.uniqueChromSeq)


CndbTools = cndbTools
