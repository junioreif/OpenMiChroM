Structural File I/O Explained
=============================

This document explains the structural-file I/O and CNDB streaming code in the
``feature/structural-io-polish-advanced`` branch. It is intentionally
implementation-focused: the goal is to make the actual code paths clear enough
to maintain or extend them later.

The most relevant modules are:

- ``OpenMiChroM/CndbTools.py``
- ``OpenMiChroM/_cndb_stream/reader.py``
- ``OpenMiChroM/_cndb_stream/embedded.py``
- ``OpenMiChroM/_cndb_stream/remote.py``
- ``OpenMiChroM/_cndb_stream/index.py``
- ``OpenMiChroM/_cndb_stream/filters.py``
- ``OpenMiChroM/_cndb_stream/embedded_writer.py``
- ``OpenMiChroM/_structural_io/detect.py``
- ``OpenMiChroM/_structural_io/formats.py``
- ``OpenMiChroM/_structural_io/converters.py``
- ``OpenMiChroM/CustomReporter.py``
- ``OpenMiChroM/ChromDynamics.py``

1. Overview
-----------

The new structural-file I/O system extends CNDBTools without replacing the
historical local ``h5py`` workflow.

It adds five main capabilities:

1. Conservative local/remote file detection through
   ``OpenMiChroM._structural_io.detect_structural_file``.
2. Automatic opening through ``CndbTools.open(...)``.
3. Remote indexed HDF5/CNDB/SW coordinate streaming through
   ``CndbTools.from_remote(...)`` and the internal ``OpenMiChroM._cndb_stream``
   backend.
4. CNDB v2 writing with ``/Header``, ``/_index``, and root attribute
   ``_index_offset`` through ``SaveStructure`` and the embedded-index writer.
5. Small local converters among NDB, CNDB/HDF5, PDB, and simple SpaceWalk text
   files through ``convert_structure_file(...)``.

The relationship between modules is:

``OpenMiChroM.CndbTools``
  The public user API. It keeps ``load(...)`` and local ``xyz(...)`` behavior
  intact. New APIs ``open(...)`` and ``from_remote(...)`` route to internal
  backends. Streaming byte accounting is exposed through ``stream_stats()``.

``OpenMiChroM._structural_io``
  Local/remote structural detection, text NDB reading, bead-selection
  coalescing, and local converters. This layer decides whether a file is local,
  remote, HDF5, text NDB, SpaceWalk, simple CNDB, nested SWB/NDB, indexed, or
  unsafe for direct remote streaming.

``OpenMiChroM._cndb_stream``
  The internal indexed HDF5 reader. It knows how to load external/local indexes
  or embedded HDF5 object indexes, inspect dataset metadata, compute coordinate
  byte ranges, and read those bytes locally or remotely.

Vendored pyfive / hdf5-indexed-reader backend
  Located under ``OpenMiChroM/_cndb_stream/_vendor/hdf5_indexed_reader``. It is
  used to inspect HDF5 object headers and dataset metadata from a seekable
  file-like object. For remote files, that file-like object is backed by HTTP
  Range requests.

``h5py``
  Used for normal local CNDB behavior, local index building, local HDF5
  inspection, and writing CNDB files.

``urllib``
  Used for all HTTP access. ``OpenMiChroM._cndb_stream.remote.read_range``
  sends Range headers and rejects unsafe responses.

2. Main user-facing entry points
--------------------------------

Open any supported local or safely streamable remote file:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   tools = CndbTools.open("trajectory.cndb")
   xyz = tools.xyz(frames=[1], beadSelection=range(0, 10))

``CndbTools.open(...)`` calls ``detect_structural_file(...)`` and then chooses a
safe backend:

- local simple CNDB/HDF5/SW: historical ``load(...)``/``h5py`` path
- local nested HDF5/SW layout: internal local indexed backend
- local text NDB: internal ``NDBTextReader``
- remote indexed HDF5 with safe Range support: ``from_remote(...)``
- remote unsafe or non-indexed HDF5: error by default
- small remote fallback: only with explicit ``allow_download=True`` and
  ``max_download_size_mb=...``

Explicit remote streaming:

.. code-block:: python

   tools = CndbTools.from_remote(
       h5_url="https://example.org/trajectory.cndb",
       trajectory="replica1_chr1",
       index_cache_path="trajectory.embedded-index.json.gz",
   )

   xyz = tools.xyz(frames=[1], beadSelection=range(0, 10))
   print(xyz.shape)
   print(tools.stream_stats())

``from_remote(...)`` wraps ``OpenMiChroM._cndb_stream.IndexedCNDB`` in the thin
``_CNDBStreamBackend`` adapter. The adapter exposes ``n_frames``, ``n_beads``,
``frame_ids``, ``trajectories``, metadata fields, coordinate reading, and byte
statistics to CNDBTools.

Structural conversions:

.. code-block:: python

   from OpenMiChroM.CndbTools import convert_structure_file

   convert_structure_file("input.ndb", "output.cndb")
   convert_structure_file("input.cndb", "output.ndb")
   convert_structure_file("input.ndb", "output.pdb")

   CndbTools.convert("input.cndb", "output.ndb")

The command-line wrapper is:

.. code-block:: bash

   python scripts/convert_structure_file.py input.ndb output.cndb
   python scripts/convert_structure_file.py input.cndb subset.ndb --frames 1,10 --start 0 --stop 100

3. File detection
-----------------

Detection is implemented in ``OpenMiChroM/_structural_io/detect.py``. The
public function is:

.. code-block:: python

   from OpenMiChroM._structural_io import detect_structural_file

   info = detect_structural_file(path_or_url)

It returns a ``StructuralFileInfo`` dataclass from
``OpenMiChroM/_structural_io/formats.py``. The fields are:

- ``source``
- ``is_remote``
- ``file_type``
- ``layout``
- ``has_embedded_index``
- ``range_supported``
- ``direct_streaming_supported``
- ``trajectories``
- ``frame_count``
- ``bead_count``
- ``coordinate_paths``
- ``dtype``
- ``compression``
- ``filters``
- ``chunks``
- ``file_size``
- ``detected_hdf5``
- ``detected_text_ndb``
- ``detected_spacewalk``
- ``notes``

Local detection
~~~~~~~~~~~~~~~

For local files, ``_detect_local``:

1. Records file size with ``path.stat().st_size``.
2. Reads a small leading sample.
3. Infers a preliminary type from suffix:

   - ``.cndb`` -> ``cndb``
   - ``.sw``, ``.spw``, ``.swb`` -> ``sw``
   - ``.ndb`` -> ``ndb``
   - ``.h5``, ``.hdf5`` -> ``hdf5``

4. Classifies the sample:

   - HDF5 if it starts with the HDF5 signature ``b"\x89HDF\r\n\x1a\n"``
   - text NDB if it starts with ``HEADER`` or contains ``CHROM``/``MODEL`` in
     the first sample
   - text SpaceWalk if it starts with ``##format=sw`` or contains a
     ``trace`` marker

5. If the file is HDF5, opens it with ``h5py`` and inspects the layout.

Remote detection
~~~~~~~~~~~~~~~~

For remote URLs, ``_detect_remote`` is deliberately conservative:

1. Runs a ``HEAD`` request to collect ``Content-Length`` and any
   ``Accept-Ranges`` note.
2. Runs a small ``Range: bytes=0-N`` probe.
3. Marks ``range_supported=True`` only if the response status is
   ``206 Partial Content``.
4. If the server returns ``200 OK`` to the Range probe, it records a safety
   note and disables direct streaming for that endpoint.
5. If the sample is HDF5 and Range is supported, it tries to inspect the
   embedded index remotely with pyfive and ``StrictRangeFile``.

This matters because many servers accept a Range header but ignore it. A
``200 OK`` response can mean "here is the whole file", which is unacceptable
for 100 GB to TB-scale CNDB files.

HDF5 layout identification
~~~~~~~~~~~~~~~~~~~~~~~~~~

The detection code recognizes:

``openmichrom-simple-cndb``
  Root-level numeric frame datasets with shape ``(n_beads, 3)`` and optional
  ``/types``.

``openmichrom-cndb-v2``
  Same root-level numeric frame layout, plus a ``/Header`` group whose
  attributes identify ``format_name == "OpenMiChroM-CNDB"`` and
  ``format_version`` starting with ``2``.

``nested-ndb-swb``
  Top-level trajectory groups containing a ``spatial_position`` group. Frame
  datasets live below ``/<trajectory>/spatial_position/<frame>``.

``hdf5-with-header``
  HDF5 file with a ``Header`` group but no recognized coordinate layout.

``unknown-hdf5``
  HDF5 file without a recognized simple or nested coordinate layout.

For both local ``h5py`` and remote pyfive inspection, detection records the
first coordinate dataset shape, dtype, compression, filters, chunks, bead
count, and whether direct streaming is supported.

Direct streaming is supported when:

- coordinate shape is two-dimensional and second dimension is ``3``
- the file is local or has an embedded index for remote use
- and either:

  - the dataset is contiguous, uncompressed, has a data offset, or
  - the dataset is chunked and its filter pipeline is supported

4. HDF5/CNDB layouts
--------------------

Old/simple OpenMiChroM CNDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The historical local CNDB layout is:

.. code-block:: text

   /
   ├── types
   ├── 0
   ├── 1
   ├── 2
   └── ...

or, in some files, frames may start at ``1``:

.. code-block:: text

   /types
   /1
   /2
   /3

Frame datasets are HDF5 datasets shaped:

.. code-block:: text

   (n_beads, 3)

``CndbTools.load(...)`` still opens these local files with ``h5py.File`` and
uses ``np.array(self.cndb[str(i)])`` in local ``xyz(...)``.

New OpenMiChroM CNDB v2
~~~~~~~~~~~~~~~~~~~~~~~

The new writer keeps the historical coordinate layout and adds metadata:

.. code-block:: text

   /
   ├── Header
   ├── types
   ├── 0
   ├── 1
   ├── 2
   ├── _index
   └── attrs["_index_offset"]

``/Header`` stores attributes created by
``initialize_cndb_header(...)`` and updated by ``finalize_cndb_header(...)``:

- ``format_name`` = ``OpenMiChroM-CNDB``
- ``format_version`` = ``2.0``
- ``creator`` = ``OpenMiChroM``
- ``openmichrom_version``
- ``creation_time``
- ``coordinate_units``
- ``coordinate_dtype``
- ``frame_layout`` = ``root_numeric_frames``
- ``n_beads``
- ``n_frames``
- ``indexed``
- ``index_format`` = ``hdf5-object-offset-json-gzip`` when indexed
- ``index_dataset`` = ``/_index`` when indexed
- ``notes``
- after index writing, ``index_compressed_nbytes`` and ``index_object_count``

The root ``_index_offset`` attribute stores the HDF5 object-header offset for
``/_index``.

Nested NDB/SWB-style HDF5
~~~~~~~~~~~~~~~~~~~~~~~~~

Nested files are detected as trajectory groups:

.. code-block:: text

   /
   ├── replica1_chr1
   │   ├── types
   │   ├── genomic_position
   │   ├── time
   │   └── spatial_position
   │       ├── 1
   │       ├── 2
   │       └── ...
   └── replica1_chr2
       └── spatial_position
           ├── t_0
           ├── t_1
           └── ...

The code accepts numeric frame names and ``t_<number>`` names. It sorts frame
IDs numerically rather than lexicographically.

Text NDB
~~~~~~~~

Text NDB support lives in ``OpenMiChroM/_structural_io/readers.py``. The
``NDBTextReader`` parses:

- ``MODEL`` lines as frame boundaries
- ``CHROM`` lines as bead coordinates, type labels, and genomic intervals
- ``ENDMDL`` as frame commit

It stores frames in memory as NumPy arrays and exposes a small backend-like API:
``n_frames``, ``n_beads``, ``frame_ids``, ``types``,
``genomic_positions``, ``get_coordinates(...)``, and zero byte counters.

SpaceWalk ``.sw/.spw/.swb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two SpaceWalk-related paths:

HDF5 SW/SWB
  Treated as HDF5, usually nested ``/<trajectory>/spatial_position/<frame>``.
  Local files can be read by the nested local backend or converted with
  ``convert_structure_file``. Remote files can stream if they expose an
  embedded index and compatible coordinate datasets.

Text SpaceWalk
  A simple clean-room dialect is detected and converted. Supported text files
  have an optional ``##format=sw1`` header, optional column header, ``trace``
  frame markers, and rows like:

  .. code-block:: text

     chromosome start end x y z
     trace 0
     chr1 1 50000 0.0 1.0 2.0

  Text SpaceWalk does not encode OpenMiChroM compartment labels, so converted
  bead types are ``UN``.

5. Embedded indexation
----------------------

The embedded index follows the hdf5-indexer idea and is implemented internally
in ``OpenMiChroM/_cndb_stream/embedded.py`` and
``OpenMiChroM/_cndb_stream/embedded_writer.py``.

The file contains:

``/_index``
  An opaque HDF5 dataset containing gzip-compressed JSON.

``attrs["_index_offset"]``
  A root attribute storing the HDF5 object-header offset for ``/_index``.

The JSON maps HDF5 group paths to child object-header offsets. For example:

.. code-block:: json

   {
     "/": {
       "Header": 800,
       "types": 7096,
       "0": 7368,
       "1": 7640
     },
     "/Header": {}
   }

Object-header offsets are useful because pyfive can jump directly to the HDF5
object header for a group or dataset. Without an object offset index, a remote
reader would have to walk the HDF5 link tree and read many pieces of metadata,
which is slow and may be impossible to do safely without broad reads.

How the embedded reader loads the index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``EmbeddedIndexProvider`` uses a vendored pyfive ``File`` class. For remote
files, the File reads from ``BufferedStrictRangeFile``, which reads from
``StrictRangeFile``, which reads from ``RemoteByteReader``.

The loading flow is:

1. Create a pyfive probe file with a dummy index.
2. Read ``h5_probe.attrs["_index_offset"]``. If absent, fall back to the
   ``_index`` root link.
3. Use low-level pyfive ``DataObjects`` at that object-header offset.
4. Confirm the object is a dataset.
5. Parse the dataset's HDF5 layout message in ``_contiguous_payload_span``.
6. Seek to the contiguous payload offset.
7. Read exactly the compressed index payload size.
8. If the payload starts with gzip magic bytes, decompress it.
9. Parse JSON into ``object_index``.
10. Reopen pyfive with ``index=object_index``.

The index cache
~~~~~~~~~~~~~~~

``IndexedCNDB.from_embedded_index(...)`` accepts:

.. code-block:: python

   index_cache_path="ENCFF161DID.embedded-index.json.gz"
   use_index_cache=True

If the cache exists, ``load_embedded_index_cache`` loads the parsed object
index from local JSON.gz and avoids re-reading the remote ``/_index`` payload.
The cache stores index metadata only. It does not cache coordinate data.

How the writer creates the index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``write_embedded_index(h5_path)``:

1. Removes any existing ``/_index`` dataset and ``_index_offset`` attribute.
2. Calls ``build_object_offset_index(...)``.
3. Uses vendored pyfive to walk HDF5 groups and collect child object-header
   offsets.
4. Excludes ``/_index`` itself from the JSON.
5. Serializes the object-offset map as compact JSON.
6. Gzip-compresses the JSON.
7. Stores the compressed bytes as a one-element opaque HDF5 dataset named
   ``/_index``.
8. Uses pyfive to find the object-header offset of ``/_index``.
9. Writes that offset to root attribute ``_index_offset``.
10. Updates ``/Header`` with compressed index size and object count.

6. Remote streaming step-by-step
--------------------------------

Example:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   ENCODE_URL = "https://encode-public.s3.amazonaws.com/2023/02/02/7f75d816-342a-4b49-adbd-aaa499dc5201/ENCFF161DID.cndb"

   tools = CndbTools.from_remote(
       h5_url=ENCODE_URL,
       trajectory="replica1_chr1",
       index_cache_path="/tmp/ENCFF161DID.embedded-index.json.gz",
   )

   coords = tools.xyz(frames=[1], beadSelection=range(0, 10))

Step by step:

1. ``CndbTools.from_remote`` creates ``_CNDBStreamBackend``.
2. ``_CNDBStreamBackend`` calls ``IndexedCNDB.from_embedded_index``.
3. ``IndexedCNDB.from_embedded_index`` creates ``EmbeddedIndexProvider``.
4. ``EmbeddedIndexProvider`` creates a strict remote Range-backed file object.
5. If the index cache exists and matches the URL, the object-offset index is
   loaded locally. Otherwise, the provider reads ``/_index`` from the remote
   file by byte range, decompresses it, parses JSON, and optionally writes the
   cache.
6. The provider discovers trajectories from the embedded object-index map. In
   the ENCODE file, there are many nested trajectories such as
   ``replica1_chr1``.
7. The provider creates a lazy cndb-stream index shell. It resolves only the
   first frame metadata initially.
8. ``tools.xyz(...)`` plans the bead selection. ``range(0, 10)`` becomes one
   half-open range ``(0, 10)`` with strategy ``single-range``.
9. ``IndexedCNDB.get_coordinates(frame=1, start=0, stop=10)`` resolves frame
   ``1`` to the path ``/replica1_chr1/spatial_position/1``.
10. If that frame metadata is not already cached, the provider uses the
    object-header offset from the embedded index to inspect the HDF5 dataset.
11. The reader determines shape, dtype, layout, chunks, compression, filters,
    data offset, and whether direct contiguous reads are supported.
12. For a contiguous uncompressed frame, the reader computes the exact
    coordinate byte range.
13. ``RemoteByteReader`` sends:

    .. code-block:: text

       Range: bytes=start-end

14. ``remote.read_range`` requires ``206 Partial Content``. ``200 OK`` raises
    ``RangeRequestUnsupportedError``.
15. Returned bytes are converted with ``np.frombuffer(raw, dtype=dtype)`` and
    reshaped to ``(rows, 3)``.
16. CNDBTools stacks selected frames and returns shape ``(1, 10, 3)``.
17. ``stream_stats()`` reports index, metadata, coordinate bytes, selection
    strategy, and overfetch diagnostics.

7. Exact byte offset calculation
--------------------------------

For a contiguous uncompressed dataset, ``IndexedCNDB.get_coordinates`` uses:

.. code-block:: python

   row_size = shape[1] * dtype.itemsize
   byte_start = data_offset + start_row * row_size
   byte_stop = byte_start + rows * row_size

For coordinate frames:

.. code-block:: text

   shape = (n_beads, 3)
   dtype = float32
   itemsize = 4
   row_size = 3 * 4 = 12 bytes

For beads ``0:10``:

.. code-block:: text

   rows = 10
   bytes = (stop - start) * 3 * itemsize
         = 10 * 3 * 4
         = 120 bytes

The byte range is:

.. code-block:: text

   byte_start = dataset_data_offset + 0 * 12
   byte_stop  = byte_start + 120

The reader fetches those 120 coordinate payload bytes, not the full frame and
not the full HDF5 file.

For the ENCODE ``replica1_chr1`` example:

.. code-block:: text

   full frame = 4980 beads * 3 coordinates * 4 bytes
              = 59,760 coordinate bytes

So:

- 10 beads -> 120 coordinate bytes
- 100 beads -> 1,200 coordinate bytes
- full frame -> 59,760 coordinate bytes

8. Why this is faster or more bandwidth-efficient
-------------------------------------------------

The main win is bandwidth and feasibility, not always single-request latency.

The ENCODE example file is about 139 GB. Downloading it just to inspect a few
beads is not practical. The embedded index is about 25.5 MB, which is much
larger than one tiny coordinate query but tiny compared with the whole file.

Observed benchmark scale:

- cold open: reads embedded index once, about 25.5 MB for this ENCODE file
- warm open: can use ``index_cache_path`` and avoid re-reading the embedded
  index payload
- 10 beads: 120 coordinate bytes
- 100 beads: 1,200 coordinate bytes
- full frame: 59,760 coordinate bytes
- 4 frames x 100 beads: 4,800 coordinate bytes
- full ENCODE file: about 139 GB

For a single tiny query, the wall-clock time can be dominated by HTTP latency
and HDF5 metadata reads. Reading 10 beads and 1,000 beads may feel similar if
both require one HTTP request. The difference becomes substantial when users
read many frames, repeatedly open the same dataset with an index cache, or
select small genomic windows from a very large remote file.

9. HTTP Range safety
--------------------

Remote HDF5 streaming requires byte-accurate HTTP Range support.

The safe response is:

.. code-block:: text

   206 Partial Content

The unsafe response is:

.. code-block:: text

   200 OK

to a request that included ``Range: bytes=start-end``.

``OpenMiChroM._cndb_stream.remote.read_range`` rejects ``200 OK`` because that
usually means the server ignored the Range header and may be returning the full
file. The error is ``RangeRequestUnsupportedError``.

This is why some public endpoints are detected but not streamable. Older Rice
NDB CNDB URLs and some NCBI GEO download endpoints may return ``200 OK`` to a
Range request. OpenMiChroM refuses to stream those URLs directly because doing
otherwise could accidentally download a very large HDF5 file.

10. Local vs remote behavior
----------------------------

Local simple CNDB
  Historical ``CndbTools.load(...)`` behavior remains. Local coordinate reads
  use ``h5py`` and NumPy.

Local nested HDF5/SW
  ``CndbTools.open(...)`` builds an in-memory local index with
  ``build_index(...)`` and uses ``IndexedCNDB`` with a ``LocalByteReader``.
  Chunked local layouts can fall back to ``h5py`` row slicing.

Local text NDB
  ``CndbTools.open(...)`` uses ``NDBTextReader``.

Remote indexed HDF5/CNDB/SW
  Uses ``CndbTools.from_remote(...)`` and the embedded-index Range backend.

Remote non-indexed HDF5
  Rejected by default. The code does not remotely walk the HDF5 tree without an
  embedded object index.

Remote text NDB or text SpaceWalk
  Detection uses only a small initial byte sample. Full remote text parsing is
  not attempted by default because it would require downloading the text file.

Explicit small remote download fallback
  ``CndbTools.open(..., allow_download=True, max_download_size_mb=...)`` can
  download a small non-streamable remote file only after size checks. The code
  requires a known file size and refuses downloads above the configured limit.

11. Non-contiguous bead selections
----------------------------------

``CndbTools.xyz(...)`` supports streaming bead selections beyond simple ranges.
The planning code is ``_stream_bead_plan``.

Examples:

``range(0, 10)`` or ``slice(0, 10)``
  One exact range: ``(0, 10)``. Strategy ``single-range``.

``[0, 1, 2, 10, 11, 12]``
  With coalescing enabled, this becomes two half-open ranges:
  ``(0, 3)`` and ``(10, 13)``. The code reads each range and then reassembles
  the rows in the requested order.

Sparse selections with too many ranges
  If coalescing would produce more than ``max_ranges`` ranges, CNDBTools reads
  the smallest enclosing range and subsets in memory. Strategy
  ``enclosing-range``.

``stream_stats()`` records:

- requested coordinate bytes
- transferred coordinate bytes
- overfetch bytes
- range request count
- coalesced range count
- final selection strategy

For chunked datasets, transferred bytes can exceed requested bytes because the
reader may need whole HDF5 chunks.

12. Chunked and compressed HDF5
-------------------------------

Contiguous uncompressed datasets are the simplest and fastest path because the
reader can compute exact byte spans directly from ``data_offset``.

Chunked datasets are different:

- HDF5 chunks are the smallest independently readable unit.
- A bead subset may intersect one or more chunks.
- The backend reads the relevant chunk payloads, decodes them, and slices rows
  in memory.
- Transferred bytes may exceed requested coordinate bytes.

The code classifies filter support in ``OpenMiChroM/_cndb_stream/filters.py``.
Supported built-in filters are:

- no filter
- gzip/deflate, filter id ``1``
- shuffle, filter id ``2``
- Fletcher32 checksum, filter id ``3``

The helper ``decode_hdf5_chunk`` decodes the supported filter subset in reverse
pipeline order and verifies Fletcher32. The vendored pyfive backend performs
the actual chunk reads in normal streaming. The helper documents and tests the
expected filter semantics.

Unsupported filters include:

- SZIP
- n-bit
- scale-offset
- unavailable LZF codec
- unknown/custom plugin filters

Unsupported filters are not silently decoded. They cause an
``UnsupportedLayoutError`` or detection marks direct streaming unsupported.

Definitions:

``requested_coordinate_bytes``
  The mathematical size of the user-requested coordinate rows.

``transferred_coordinate_bytes``
  The coordinate or chunk bytes actually transferred by streaming reads.

``overfetch_coordinate_bytes``
  ``transferred_coordinate_bytes - requested_coordinate_bytes`` when positive.

13. New indexed CNDB writer
---------------------------

The writer path is in ``OpenMiChroM/CustomReporter.py`` and
``OpenMiChroM/_cndb_stream/embedded_writer.py``.

``SaveStructure`` now accepts:

.. code-block:: python

   SaveStructure(
       filePrefix="traj",
       reportInterval=1000,
       mode="cndb",
       indexed=True,
       metadata=True,
       coordinate_dtype=None,
   )

``MiChroM.createReporters`` forwards:

.. code-block:: python

   sim.createReporters(
       traj=True,
       trajFormat="cndb",
       trajIndexed=True,
       trajMetadata=True,
       trajCoordinateDtype=None,
   )

For ``mode="cndb"``:

1. The reporter creates one ``.cndb`` file per chain.
2. If ``metadata=True``, it calls ``initialize_cndb_header``.
3. It writes ``/types``.
4. Each report writes a numeric frame dataset.
5. ``close()`` finalizes ``n_frames`` and other ``/Header`` attributes.
6. If ``indexed=True``, ``close()`` calls ``write_embedded_index`` after the
   HDF5 file has been closed.

Inspect a written file:

.. code-block:: python

   import h5py

   with h5py.File("trajectory.cndb", "r") as f:
       print(list(f.keys()))
       print("Header" in f)
       print("_index" in f)
       print("_index_offset" in f.attrs)
       print(dict(f["Header"].attrs))

Why this preserves compatibility:

- Coordinates remain in root numeric frame datasets.
- ``/types`` remains present.
- Old local ``CndbTools.load(...)`` ignores ``/Header`` and ``/_index``.
- New files become self-describing and remotely streamable once hosted by a
  Range-compatible server.

14. Structural converters
-------------------------

Converters are implemented in ``OpenMiChroM/_structural_io/converters.py``.
They are local-file only and intentionally in-memory.

Supported conversions:

- ``ndb -> cndb``
- ``cndb -> ndb``
- ``ndb -> pdb``
- simple ``pdb -> ndb``
- HDF5 ``sw/swb -> ndb`` for nested ``spatial_position`` layouts
- simple text SpaceWalk ``sw/spw -> ndb``
- simple text SpaceWalk ``sw/spw -> cndb``
- ``ndb -> sw/spw``

The converter uses a shared ``StructureTrajectory`` dataclass:

.. code-block:: python

   frames: OrderedDict[str, np.ndarray]
   types: list[str]
   genomic_positions: np.ndarray | None
   title: str

The API accepts:

- ``frames`` for selecting frame IDs
- ``start`` and ``stop`` for selecting bead rows
- ``max_frames`` for taking only the first selected frames
- ``max_memory_mb`` for refusing large selected payloads
- ``allow_large=True`` to explicitly bypass the memory guard
- PDB options ``pdb_atom_name``, ``pdb_residue_name``, ``pdb_chain_id``, and
  ``pdb_element``

Examples:

.. code-block:: python

   from OpenMiChroM.CndbTools import convert_structure_file

   convert_structure_file("input.ndb", "output.cndb")
   convert_structure_file("input.cndb", "output.ndb")
   convert_structure_file("input.ndb", "output.pdb")
   convert_structure_file("input.cndb", "subset.ndb", frames=[1, 10], start=0, stop=100)

CLI:

.. code-block:: bash

   python scripts/convert_structure_file.py input.ndb output.cndb
   python scripts/convert_structure_file.py input.cndb subset.ndb --frames 1,10 --start 0 --stop 100

Provenance note: NDB-Converters was inspected only as inspiration because the
available copy had no license file. The converter code in OpenMiChroM is a
clean-room implementation from OpenMiChroM examples and public file shapes.

15. Public URL compatibility
----------------------------

Current documented support status:

ENCODE indexed CNDB
  Supported. The S3 URL supports Range requests, has an embedded index, and
  direct coordinate streaming works for compatible frame datasets.

NCBI Mammoth Direct SW
  Supported when the direct endpoint supports Range requests, an embedded
  index is present, and datasets are compatible.

Rice NDB older CNDB URLs
  Detected as HDF5/CNDB, but tested endpoints returned ``200 OK`` to Range.
  They are unsafe for direct remote streaming unless the server begins honoring
  Range requests and an embedded index is available.

NCBI GEO download SW endpoint
  May return ``200 OK`` to Range. The generic download endpoint is unsafe for
  direct streaming unless redirected to a direct Range-compatible file URL.

Bintu NDB
  Detected as text NDB. Remote full parsing is not attempted by default because
  it would require downloading the text file.

Inspect a URL:

.. code-block:: bash

   python scripts/inspect_structural_file.py --url URL
   python scripts/inspect_structural_file.py --url URL --json

For multiple URLs:

.. code-block:: bash

   python scripts/inspect_structural_file.py --urls urls.txt --json-output results.json

16. ``stream_stats``
--------------------

``tools.stream_stats()`` returns backend byte counters plus CNDBTools selection
diagnostics. The actual fields currently returned are:

``index_bytes_read``
  Compressed embedded ``/_index`` payload bytes read. This is ``0`` when the
  embedded index came from ``index_cache_path`` or for local/non-streaming
  backends.

``metadata_bytes_read``
  HDF5 metadata bytes read by the embedded backend, excluding counted index
  payload bytes and counted data bytes. This includes object headers,
  superblock/root metadata, and cache overfetch from pyfive metadata reads.

``data_bytes_read``
  Coordinate payload bytes read by the direct byte-range reader plus chunk
  payload bytes counted through the embedded backend.

``bytes_read``
  Total backend bytes: data + index + metadata.

``index_cache_hit``
  Whether the embedded object index was loaded from local cache.

``coordinate_range_requests``
  Number of coordinate ranges CNDBTools asked the backend to read.

``requested_data_bytes``
  Requested coordinate bytes using the older counter name.

``transferred_data_bytes``
  Transferred coordinate bytes using the older counter name.

``overfetch_bytes``
  Overfetch bytes using the older counter name.

``range_request_count``
  Number of planned coordinate range requests.

``requested_coordinate_bytes``
  Mathematical bytes requested by the bead selection.

``transferred_coordinate_bytes``
  Bytes planned/measured as transferred for coordinates or chunks.

``overfetch_coordinate_bytes``
  Positive difference between transferred and requested coordinate bytes.

``coalesced_range_count``
  Number of ranges used when strategy is ``coalesced-ranges``.

``selection_strategy``
  One of ``full-frame``, ``single-range``, ``coalesced-ranges``,
  ``enclosing-range``, or ``empty`` depending on the most recent streaming
  selection.

For local h5py mode, all byte counts are zero and ``index_cache_hit`` is
``False``.

17. Safety and correctness guarantees
-------------------------------------

The implementation enforces several safety rules:

- Remote HDF5 byte reads require ``206 Partial Content``.
- ``200 OK`` to a Range request raises ``RangeRequestUnsupportedError``.
- ``StrictRangeFile.read(size=-1)`` is intentionally disabled.
- Remote non-indexed HDF5 files are rejected by default.
- Download fallback is opt-in and size-limited.
- Unsupported HDF5 layouts or filters raise errors rather than returning
  potentially wrong data.
- Chunk/filter support is tested with synthetic HDF5 fixtures.
- Normal pytest does not depend on public network files.
- Optional public URL tests are gated by environment variables.
- Local h5py reads are used as references in tests for synthetic files.

18. Tests
---------

Normal local tests:

.. code-block:: bash

   pytest -q

Tutorial sync:

.. code-block:: bash

   python scripts/sync_tutorials.py --check --clear-outputs

Optional ENCODE smoke test:

.. code-block:: bash

   OPENMICHROM_RUN_ENCODE_STREAM_TESTS=1 pytest -q tests/test_cndbtools_encode_stream.py

Optional public structural URL integration tests:

.. code-block:: bash

   OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION=1 pytest -q tests/test_structural_io_integration.py

Build checks:

.. code-block:: bash

   python -m build
   python -m twine check dist/*

Important tutorial execution checks:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute docs/source/Tutorials/stream_remote_cndb.ipynb --output /tmp/stream_remote_cndb.executed.ipynb --ExecutePreprocessor.timeout=600
   jupyter nbconvert --to notebook --execute docs/source/Tutorials/write_indexed_cndb.ipynb --output /tmp/write_indexed_cndb.executed.ipynb
   jupyter nbconvert --to notebook --execute docs/source/Tutorials/structural_file_conversion.ipynb --output /tmp/structural_file_conversion.executed.ipynb

19. Remaining limitations
-------------------------

The current limitations are deliberate and should not be hidden:

- Direct remote coordinate streaming requires an embedded HDF5 object index for
  remote HDF5 files.
- Coordinate datasets must have shape ``(n_beads, 3)``.
- Contiguous, uncompressed datasets are the fastest exact byte-range path.
- Chunked uncompressed and gzip-compressed datasets may read whole chunks and
  report overfetch.
- Unsupported HDF5 filters are rejected. This includes scale-offset, SZIP,
  n-bit, unavailable LZF codecs, and unknown/custom plugin filters.
- Remote non-indexed HDF5 files are not streamed by default.
- Some servers return ``200 OK`` to Range requests. Those endpoints are unsafe
  for direct streaming and are rejected.
- Remote text NDB and text SpaceWalk files are not fully parsed by default,
  because that would require downloading the text file.
- Converters are local-file only.
- Converters use an in-memory ``StructureTrajectory`` representation and are
  not general streaming large-file rewriters.
- The converter memory guard estimates the selected payload after reading the
  source into memory; it makes large conversions explicit but does not turn the
  converter into a streaming converter.
- PDB conversion is coarse-grained and approximate. It is intended for simple
  CA-like bead records and visualization workflows, not exact biological PDB
  semantics.
- Text SpaceWalk support is narrow: simple trace-style
  ``chromosome start end x y z`` rows are supported, richer dialects may need
  adapters and licensed examples.
- ``get_coordinates_many`` currently loops over frames; future versions may
  coalesce adjacent byte ranges.
- Metadata byte accounting includes backend cache overfetch and is therefore a
  practical diagnostic, not a perfect HDF5 metadata lower bound.
