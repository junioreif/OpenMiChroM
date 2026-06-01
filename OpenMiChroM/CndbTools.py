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
import tempfile
from scipy.spatial import distance
from pathlib import Path
from urllib.request import Request, urlopen

from OpenMiChroM._structural_io.converters import convert_structure_file


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


def _download_remote_structural_file(
    url,
    *,
    file_size,
    max_download_size_mb,
    download_cache_path=None,
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

    if download_cache_path is None:
        suffix = Path(str(url).split("?", 1)[0]).suffix or ".structural"
        handle = tempfile.NamedTemporaryFile(
            prefix="openmichrom-structural-",
            suffix=suffix,
            delete=False,
        )
        target_path = Path(handle.name)
        handle.close()
    else:
        target_path = Path(download_cache_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    context = None if verify_ssl else ssl._create_unverified_context()
    request = Request(url)
    total = 0
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            with target_path.open("wb") as output:
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
    except Exception:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise

    return target_path


class _CNDBStreamBackend:
    """Thin adapter around the internal indexed CNDB streaming backend."""

    def __init__(self, h5_url, trajectory=None, index_cache_path=None, **kwargs):
        from OpenMiChroM._cndb_stream import IndexedCNDB

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

    def get_coordinates(self, frame, start=None, stop=None):
        return self.traj.get_coordinates(frame=frame, start=start, stop=stop)

    def stats(self):
        return {
            "index_bytes_read": self.traj.index_bytes_read,
            "metadata_bytes_read": self.traj.metadata_bytes_read,
            "data_bytes_read": self.traj.data_bytes_read,
            "bytes_read": self.traj.bytes_read,
            "index_cache_hit": self.traj.index_cache_hit,
        }


class _CNDBLocalIndexedBackend:
    """Thin adapter around the internal indexed reader for local nested files."""

    def __init__(self, h5_path, trajectory=None):
        from OpenMiChroM._cndb_stream import IndexedCNDB
        from OpenMiChroM._cndb_stream.index import build_index

        index = build_index(h5_path, trajectories=[trajectory] if trajectory else None)
        self.traj = IndexedCNDB(h5_path=h5_path, _index=index, trajectory=trajectory)

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

    def get_coordinates(self, frame, start=None, stop=None):
        return self.traj.get_coordinates(frame=frame, start=start, stop=stop)

    def stats(self):
        return {
            "index_bytes_read": 0,
            "metadata_bytes_read": 0,
            "data_bytes_read": self.traj.data_bytes_read,
            "bytes_read": self.traj.bytes_read,
            "index_cache_hit": False,
        }


class cndbTools:

    def __init__(self):
        self.Type_conversion = {'A1':0, 'A2':1, 'B1':2, 'B2':3, 'B3':4, 'B4':5, 'NA':6}
        self.Type_conversionInv = {y:x for x,y in self.Type_conversion.items()}
        self._stream_backend = None
        self.is_remote = False
        self.frame_ids = []
        self.trajectories = []
        self.current_trajectory = None
        self.genomic_positions = None
        self.types = []
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
    def from_remote(cls, h5_url, trajectory=None, index_cache_path=None, **kwargs):
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
            **kwargs,
        )
        tool.is_remote = True
        tool.cndb = None
        tool.ChromSeq = _metadata_list(tool._stream_backend.types)
        tool.Nbeads = tool._stream_backend.n_beads
        tool.Nframes = tool._stream_backend.n_frames
        tool.frame_ids = tool._stream_backend.frame_ids
        tool.trajectories = tool._stream_backend.trajectories
        tool.current_trajectory = tool._stream_backend.current_trajectory
        tool.types = tool.ChromSeq
        tool.genomic_positions = tool._stream_backend.genomic_positions
        tool.uniqueChromSeq = set(tool.ChromSeq)
        tool.dictChromSeq = {}
        for tt in tool.uniqueChromSeq:
            tool.dictChromSeq[tt] = [i for i, value in enumerate(tool.ChromSeq) if value == tt]
        return tool

    @staticmethod
    def convert(input_path, output_path=None, **kwargs):
        R"""
        Convert a small local structural trajectory file.

        This is a convenience wrapper around
        ``OpenMiChroM._structural_io.convert_structure_file``. Supported
        conversions include simple NDB, CNDB/HDF5, PDB, and supported HDF5
        SW/SWB layouts. Remote URLs are not downloaded by the converter.
        """

        return convert_structure_file(input_path, output_path, **kwargs)

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
                local_path = _download_remote_structural_file(
                    source,
                    file_size=info.file_size,
                    max_download_size_mb=max_download_size_mb,
                    download_cache_path=download_cache_path,
                    timeout=detect_kwargs.get("timeout", 30.0),
                    verify_ssl=detect_kwargs.get("verify_ssl", True),
                )
                tool = cls.open(local_path, trajectory=trajectory, mode=mode, **kwargs)
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

        if info.detected_hdf5 and info.layout == "nested-ndb-swb":
            tool = cls()
            tool._stream_backend = _CNDBLocalIndexedBackend(source, trajectory=trajectory)
            tool.is_remote = False
            tool.cndb = None
            tool.ChromSeq = _metadata_list(tool._stream_backend.types)
            tool.Nbeads = tool._stream_backend.n_beads
            tool.Nframes = tool._stream_backend.n_frames
            tool.frame_ids = tool._stream_backend.frame_ids
            tool.trajectories = tool._stream_backend.trajectories
            tool.current_trajectory = tool._stream_backend.current_trajectory
            tool.types = tool.ChromSeq
            tool.genomic_positions = tool._stream_backend.genomic_positions
            tool.uniqueChromSeq = set(tool.ChromSeq)
            tool.dictChromSeq = {}
            for tt in tool.uniqueChromSeq:
                tool.dictChromSeq[tt] = [i for i, value in enumerate(tool.ChromSeq) if value == tt]
            return tool
        if info.detected_hdf5 and info.file_type in {"cndb", "hdf5", "sw"}:
            return cls().load(source)
        if info.detected_text_ndb:
            from OpenMiChroM._structural_io.readers import NDBTextReader

            tool = cls()
            tool._stream_backend = NDBTextReader(source)
            tool.is_remote = False
            tool.cndb = None
            tool.ChromSeq = _metadata_list(tool._stream_backend.types)
            tool.Nbeads = tool._stream_backend.n_beads
            tool.Nframes = tool._stream_backend.n_frames
            tool.frame_ids = tool._stream_backend.frame_ids
            tool.trajectories = tool._stream_backend.trajectories
            tool.current_trajectory = tool._stream_backend.current_trajectory
            tool.types = tool.ChromSeq
            tool.genomic_positions = tool._stream_backend.genomic_positions
            tool.uniqueChromSeq = set(tool.ChromSeq)
            tool.dictChromSeq = {}
            for tt in tool.uniqueChromSeq:
                tool.dictChromSeq[tt] = [i for i, value in enumerate(tool.ChromSeq) if value == tt]
            return tool
        raise ValueError(
            f"Could not open structural file {source!r}. "
            f"Detected file_type={info.file_type!r}, layout={info.layout!r}."
        )
    
    def load(self, fileName):
        R"""
        Receives the path to **cndb** or **ndb** file to perform analysis.
        
        Args:
            fileName (file, required):
                Path to cndb or ndb file. If an ndb file is given, it is converted to a cndb file and saved in the same directory.
        """
        f_name, file_extension = os.path.splitext(fileName)
        
        if file_extension == ".ndb":
            fileName = Chrom_utils.ndb2cndb(f_name)   

        self.cndb = h5py.File(fileName, 'r')
        self._stream_backend = None
        self.is_remote = False
        
        self.ChromSeq = list(self.cndb['types'])
        self.uniqueChromSeq = set(self.ChromSeq)
        self.types = self.ChromSeq
        self.genomic_positions = None
        
        self.dictChromSeq = {}
        
        for tt in self.uniqueChromSeq:
            self.dictChromSeq[tt] = ([i for i, e in enumerate(self.ChromSeq) if e == tt])
        
        self.Nbeads = len(self.ChromSeq)
        self.frame_ids = sorted(
            [key for key in self.cndb.keys() if str(key).isdigit()],
            key=lambda frame: int(frame),
        )
        self.Nframes = len(self.frame_ids)
        self.trajectories = []
        self.current_trajectory = None
        
        return(self)
    
    

    def ndb2cndb(self, fileName):
        R"""
        Converts an **ndb** file format to **cndb**.
        
        Args:
            filename (path, required):
                    Path to the ndb file to be converted to cndb.
        """
        Main_chrom      = ['ChrA','ChrB','ChrU'] # Type A B and Unknow
        Chrom_types     = ['ZA','OA','FB','SB','TB','LB','UN']
        Chrom_types_NDB = ['A1','A2','B1','B2','B3','B4','UN']
        Res_types_PDB   = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS', 'HIS', 'GLY']
        Type_conversion = {'A1': 0,'A2' : 1,'B1' : 2,'B2' : 3,'B3' : 4,'B4' : 5,'UN' : 6}
        title_options = ['HEADER','OBSLTE','TITLE ','SPLT  ','CAVEAT','COMPND','SOURCE','KEYWDS','EXPDTA','NUMMDL','MDLTYP','AUTHOR','REVDAT','SPRSDE','JRNL  ','REMARK']
        model          = "MODEL     {0:4d}"
        atom           = "ATOM  {0:5d} {1:^4s}{2:1s}{3:3s} {4:1s}{5:4d}{6:1s}   {7:8.3f}{8:8.3f}{9:8.3f}{10:6.2f}{11:6.2f}          {12:>2s}{13:2s}"
        ter            = "TER   {0:5d}      {1:3s} {2:1s}{3:4d}{4:1s}"

        file_ndb = fileName + str(".ndb")
        name     = fileName + str(".cndb")

        cndbf = h5py.File(name, 'w')
        
        ndbfile = open(file_ndb, "r")
        
        loop = 0
        types = []
        types_bool = True
        loop_list = []
        x = []
        y = [] 
        z = []

        frame = 0

        for line in ndbfile:

            entry = line[0:6]

            info = line.split()


            if 'MODEL' in entry:
                frame += 1

                inModel = True

            elif 'CHROM' in entry:

                subtype = line[16:18]

                types.append(subtype)
                x.append(float(line[40:48]))
                y.append(float(line[49:57]))
                z.append(float(line[58:66]))

            elif 'ENDMDL' in entry:
                if types_bool:
                    typelist = [Type_conversion[x] for x in types]
                    cndbf['types'] = typelist
                    types_bool = False

                positions = np.vstack([x,y,z]).T
                cndbf[str(frame)] = positions
                x = []
                y = []
                z = []

            elif 'LOOPS' in entry:
                loop_list.append([int(info[1]), int(info[2])])
                loop += 1
        
        if loop > 0:
            cndbf['loops'] = loop_list

        cndbf.close()
        return(name)
    
    def xyz(
        self,
        frames=None,
        beadSelection=None,
        XYZ=[0,1,2],
        coalesce=True,
        max_gap=0,
        max_ranges=128,
    ):
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
        if self._stream_backend is not None:
            return self._xyz_stream(
                frames=frames,
                beadSelection=beadSelection,
                XYZ=XYZ,
                coalesce=coalesce,
                max_gap=max_gap,
                max_ranges=max_ranges,
            )

        frame_list = []
        
        if beadSelection == None:
            selection = np.arange(self.Nbeads)
        else:
            selection = np.array(beadSelection)
            
        if frames == None:
            frames = self.frame_ids
        
        for i in frames:
            frame_list.append(np.take(np.take(np.array(self.cndb[str(i)]), selection, axis=0), XYZ, axis=1))
        return(np.array(frame_list))

    def _xyz_stream(
        self,
        frames=None,
        beadSelection=None,
        XYZ=[0,1,2],
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
        if frames is None:
            frames = self.frame_ids
        elif isinstance(frames, (int, np.integer, str)):
            frames = [frames]

        ranges, post_selection, requested_rows, strategy = self._stream_bead_plan(
            beadSelection,
            coalesce=coalesce,
            max_gap=max_gap,
            max_ranges=max_ranges,
        )
        axis_selection = np.array(XYZ)

        for frame in frames:
            coords = self._stream_read_ranges(frame, ranges)
            if post_selection is not None:
                coords = np.take(coords, post_selection, axis=0)
            frame_list.append(np.take(coords, axis_selection, axis=1))
            if coords.size:
                itemsize = coords.dtype.itemsize
            else:
                itemsize = np.dtype(np.float32).itemsize
            transferred_rows = sum(stop - start for start, stop in ranges)
            transferred_bytes = transferred_rows * 3 * itemsize
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

        if isinstance(beadSelection, slice):
            step = 1 if beadSelection.step is None else beadSelection.step
            start = 0 if beadSelection.start is None else beadSelection.start
            stop = self.Nbeads if beadSelection.stop is None else beadSelection.stop
            if step == 1:
                return [(start, stop)], None, max(0, stop - start), "single-range"
            selection = np.arange(start, stop, step, dtype=int)

        elif isinstance(beadSelection, range):
            if beadSelection.step == 1:
                return (
                    [(beadSelection.start, beadSelection.stop)],
                    None,
                    max(0, beadSelection.stop - beadSelection.start),
                    "single-range",
                )
            selection = np.array(list(beadSelection), dtype=int)
        else:
            selection = np.array(beadSelection, dtype=int)

        if selection.size == 0:
            return [(0, 0)], None, 0, "empty"

        if np.any(selection < 0):
            selection = np.where(selection < 0, selection + self.Nbeads, selection)

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

    @property
    def stream_data_bytes_read(self):
        if self._stream_backend is None:
            return 0
        return self._stream_backend.traj.data_bytes_read

    @property
    def stream_index_bytes_read(self):
        if self._stream_backend is None:
            return 0
        return self._stream_backend.traj.index_bytes_read

    @property
    def stream_metadata_bytes_read(self):
        if self._stream_backend is None:
            return 0
        return self._stream_backend.traj.metadata_bytes_read

    @property
    def stream_bytes_read(self):
        if self._stream_backend is None:
            return 0
        return self._stream_backend.traj.bytes_read

    @property
    def stream_index_cache_hit(self):
        if self._stream_backend is None:
            return False
        return self._stream_backend.traj.index_cache_hit

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

    def compute_Orientation_OP(self,xyz,chrom_start=0,chrom_end=1000,vec_length=4):
        from collections import OrderedDict
        import itertools
        R"""
        Calculates the Orientation Order Parameter OP. Details are decribed in "Zhang, Bin, and Peter G. Wolynes. "Topology, structures, and energy landscapes of human chromosomes." Proceedings of the National Academy of Sciences 112.19 (2015): 6062-6067."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.
            chrom_start (int, required):
                First bead to consider in the calculations (Default value = 0).
            chrom_end (int, required):
                Last bead to consider in the calculations (Default value = 1000).  
            vec_length (int, required):
                Number of neighbor beads to build the vector separation :math:`i` and :math:`i+4` if vec_length is set to 4. (Default value = 4).  
           
                       
        Returns:
            Oijx:class:`numpy.ndarray`:
                Returns the genomic separation employed in the calculations.
            Oijy:class:`numpy.ndarray`:
                Returns the Orientation Order Parameter OP as a function of the genomic separation.
        """

        vec_rij=[] 
        for i in range(chrom_start,chrom_end-vec_length):
            vec_rij.append(xyz[i+vec_length]-xyz[i]) # from a trajectory, gets the vector ri,i+vec_length

        dot_ri_rj=[]
        ij=[]
        for i in itertools.combinations_with_replacement(range(1,chrom_end-vec_length),2):  
            dot_ri_rj.append(np.dot(vec_rij[i[0]]/np.linalg.norm(vec_rij[i[0]]),vec_rij[i[1]]/np.linalg.norm(vec_rij[i[1]]))) # dot product between all vector r,r+_vec_length
            ij.append(i[1]-i[0]) # genomic separation

        d = OrderedDict()
        for k, v in zip(ij, dot_ri_rj):
            d[k] = d.get(k, 0) + v # sum the values v for each genomic separation k
        
        Oijx=[]
        Oijy=[]
        for i in range(1,np.size(list(d.keys()))+1):
            Oijx.append(list(d.keys())[i-1]+1)
            Oijy.append(list(d.values())[i-1]/(chrom_end-list(d.keys())[i-1])) # gets Oij normalized by the number of elements. For example, in a chromosome of length 100, genomic distance 1 has more elements (99) considered than genomic distance 100 (1).

        return np.asarray(Oijx),np.asarray(Oijy)


    def compute_FFT_from_Oij(self, Oijy, lowcut=1, highcut=500, order=5):
        from scipy.signal import butter, lfilter
        from scipy.fftpack import fft
        R"""
        Calculates the Fourier transform of the Orientation Order Parameter OP. Details are decribed in "Zhang, Bin, and Peter G. Wolynes. "Topology, structures, and energy landscapes of human chromosomes." Proceedings of the National Academy of Sciences 112.19 (2015): 6062-6067."
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.
            lowcut (int, required):
                Filter to cut low frequencies (Default value = 1).
            highcut (int, required):
                Filter to cut high frequencies (Default value = 500).  
            order (int, required):
                Order of the Butterworth filter obtained from `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`__. (Default value = 5).  
           
                       
        Returns:
            xf:class:`numpy.ndarray`:
                Return frequencies.
            yf:class:`numpy.ndarray`:
                Returns the Fourier transform of the Orientation Order Parameter OP in space of 1/Chrom_Length.
        """

        def _butter_bandpass(lowcut, highcut, fs, order=5):
            R"""
            Internal function for selecting frequencies.
            """
            nyq = fs//2
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            R"""
            Internal function for filtering bands.
            """
            b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        N=np.shape(Oijy)[0]
        y = _butter_bandpass_filter(Oijy, lowcut, highcut, N, order=order)
        xf = np.linspace(1, N//2 , N//2)
        yf=fft(y)/len(y)
        return (xf[0:N//2]-1)/N,np.abs(yf[0:N//2])
    

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
            for i in range(0,np.shape(xyz)[0] - np.ceil(1.25*neig_beads).astype('int')):
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
        Calculates the Radius of Gyration. 
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: Tx1):
                Returns the Radius of Gyration in units of :math:`\sigma`.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        xyz_rel_to_cm= xyz - np.tile(rcm,(xyz.shape[1],1))
        rg=np.sqrt(np.mean(np.linalg.norm(xyz_rel_to_cm,axis=2)**2,axis=1))
        return rg

    def compute_GyrTensorEigs(self, xyz):
        R"""
        Calculates the eigenvalues of the Gyration tensor:
        For a cloud of N points with positions: {[xi,yi,zi]},gyr tensor is a symmetric matrix defined as,
        
        gyr= (1/N) * [[sum_i(xi-xcm)(xi-xcm)  sum_i(xi-xcm)(yi-ycm) sum_i(xi-xcm)(zi-zcm)],
                      [sum_i(yi-ycm)(xi-xcm)  sum_i(yi-ycm)(yi-ycm) sum_i(yi-ycm)(zi-zcm)],
                      [sum_i(zi-zcm)(xi-xcm)  sum_i(zi-zcm)(yi-ycm) sum_i(zi-zcm)(zi-zcm)]]
        
        the three non-negative eigenvalues of gyr are used to define shape parameters like radius of gyration, asphericity, etc

        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: Tx3):
                Returns the sorted eigenvalues of the Gyration Tensor.
        """
        rcm=np.mean(xyz, axis=1,keepdims=True)
        sorted_eigenvals=[]
        for frame in xyz-rcm:
            gyr=np.matmul(np.transpose(frame),frame)/xyz.shape[1]
            sorted_eigenvals.append(np.sort(np.linalg.eig(gyr)[0]))
        return np.array(sorted_eigenvals)


    def compute_MSD(self,xyz):
        R"""
        Calculates the Mean-Squared Displacement using Fast-Fourier Transform. 
        Uses Weiner-Kinchin theorem to compute the autocorrelation, and a recursion realtion from the following reference:
        see Sec. 4.2 in Calandrini V, et al. (2011) EDP Sciences (https://doi.org.10.1051/sfn/201112010).
        Also see this stackoverflow post: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.  
                       
        Returns:
            :class:`numpy.ndarray` (dim: NxT):
                Returns the MSD of each particle over the trajectory.

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

    def compute_RadNumDens(self, xyz, dr=1.0, ref='centroid',center=None):

        R"""
        Calculates the radial number density of monomers; which when integrated over 
        the volume (with the appropriate kernel: 4*pi*r^2) gives the total number of monomers.
        
        Args:
            xyz (:math:`(frames, beadSelection, XYZ)` :class:`numpy.ndarray` (dim: TxNx3), required):
                Array of the 3D position of the selected beads for different frames extracted by using the `xyz()` function.  

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
                       
        Returns:
            num_density:class:`numpy.ndarray`:
                the number density
            
            bins:class:`numpy.ndarray`:
                bins corresponding to the number density

        """

        if ref=='origin':
            rad_vals = np.ravel(np.linalg.norm(xyz,axis=2))

        elif ref=='centroid':
            rcm=np.mean(xyz,axis=1, keepdims=True)
            rad_vals = np.ravel(np.linalg.norm(xyz-rcm,axis=2))

        elif ref == 'custom':
            try:
                if len(center)!=3: raise TypeError
                center=np.array(center,dtype=float)
                center_nd=np.tile(center,(xyz.shape[0],1,1))
                rad_vals=np.ravel(np.linalg.norm(xyz-center_nd,axis=2))

            except (TypeError,ValueError):
                print("FATAL ERROR!!\n Invalid 'center' for ref='custom'.\n\
                        Please provide a valid center: [x0,y0,z0]")
                return ([0],[0])
        else:
            print("FATAL ERROR!! Unvalid 'ref'\n\
                'ref' can take one of three values: 'origin', 'centroid', and 'custom'")
            return ([0],[0])

        rdp_hist,bin_edges=np.histogram(rad_vals, 
                                bins=np.arange(0,rad_vals.max()+1,dr),
                                density=False)

        bin_mids=0.5*(bin_edges[:-1] + bin_edges[1:])
        bin_vols = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        num_density = rdp_hist/(xyz.shape[0]*bin_vols)

        return (num_density, bin_mids)

        
    def compute_RDP(self, xyz, beadSelection=None, radius=20.0, bins=200):
        R"""
        Calculates the RDP - Radial Distribution Probability. Details can be found in the following publications: 
        
            - Oliveira Jr., A.B., Contessoto, V.G., Mello, M.F. and Onuchic, J.N., 2021. A scalable computational approach for simulating complexes of multiple chromosomes. Journal of Molecular Biology, 433(6), p.166700.
            - Di Pierro, M., Zhang, B., Aiden, E.L., Wolynes, P.G. and Onuchic, J.N., 2016. Transferable model for chromosome architecture. Proceedings of the National Academy of Sciences, 113(43), pp.12168-12173.
        
        Args:
            xyz (:math:`(frames, XYZ)` :class:`numpy.ndarray`, required):
                Array of the 3D position of the frames extracted by using the `xyz()` function. 
            beadSelection (:math:`(beadSelection)` :class:`numpy.ndarray`):
                The index of the beads to be sliced from `xyz` that you want to compute RDP. Usualy, you can use the internal selection using the `dictChromSeq['types']` with 'types' been the selection that you want. 
            radius (float, required):
                Radius of the sphere in units of :math:`\sigma` to be considered in the calculations. The radius value should be modified depending on your simulated chromosome length. (Default value = 20.0).
            bins (int, required):
                Number of slices to be considered as spherical shells. (Default value = 200).
                       
        Returns:
            :math:`(N, 1)` :class:`numpy.ndarray`:
                Returns the radius of each spherical shell in units of :math:`\sigma`.
            :math:`(N, 1)` :class:`numpy.ndarray`:
                Returns the RDP - Radial Distribution Probability for each spherical shell.
        """
        
        def calcDist(a,b):
            R"""
            Internal function that calculates the distance between two beads. 
            """
            return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2  )
        
        def calc_gr(ref, pos, R, dr):
            R"""
            Internal function that calculates the distance RDP - Radial Distribution Probability. 
            """
            g_r =  np.zeros(int(R/dr))
            dd = []
            for i in range(len(pos)):
                dd.append(calcDist(pos[i],ref))
            raddi =dr
            k = 0
            while (raddi <= R):
                for i in range(0,len(pos)):
                    if (dd[i] >= raddi and dd[i] < raddi+dr):
                        g_r[k] += 1

                g_r[k] = g_r[k]/(4*np.pi*dr*raddi**2)
                raddi += dr
                k += 1
            return g_r 
        
        R_nucleus = radius
        deltaR = R_nucleus/bins                  
   
        n_frames = 0 
        g_rdf = np.zeros(bins)

        if beadSelection == None:
            beadSelection = np.arange(len(xyz[0]))
            
 
        for i in range(len(xyz)):
            frame = xyz[i]
            centroide = np.mean(frame, axis=0)[None,:][0]
            n_frames += 1
            g_rdf += calc_gr(centroide, frame[beadSelection], R_nucleus, deltaR)
        
        Rx = []   
        for i in np.arange(0, int(R_nucleus+deltaR), deltaR):
            Rx.append(i)
        return(Rx, g_rdf/n_frames) 

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
