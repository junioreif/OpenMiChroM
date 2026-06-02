How Remote CNDB Streaming Works
===============================

This page gives a shorter user-facing explanation of the remote CNDB streaming
backend used by ``CndbTools.from_remote(...)``. For implementation details, see
``Developer/structural_file_io_explained`` in the documentation tree.

The core idea
-------------

Large CNDB/HDF5 files can be too large to download just to analyze a few
frames or loci. CNDBTools can now open an indexed remote CNDB/HDF5 file and
read only the bytes needed for the requested coordinates.

For example:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   tools = CndbTools.from_remote(
       h5_url=ENCODE_URL,
       trajectory="replica1_chr1",
       index_cache_path="/tmp/ENCFF161DID.embedded-index.json.gz",
   )

   xyz = tools.xyz(frames=[1], beadSelection=range(0, 10))
   print(xyz.shape)
   print(tools.stream_stats())

For a ``float32`` coordinate dataset, 10 beads require:

.. code-block:: text

   10 beads * 3 coordinates * 4 bytes = 120 coordinate bytes

The full ENCODE file used in the tutorial is about 139 GB. The call above does
not download that file.

What is read from the remote file?
----------------------------------

There are three categories of bytes:

``index_bytes_read``
  The embedded HDF5 index payload. For the ENCODE example this is about
  25.5 MB on a cold open. If ``index_cache_path`` is used, later opens can load
  this metadata from the local cache instead.

``metadata_bytes_read``
  HDF5 object headers and dataset metadata needed to find coordinate datasets.
  This may include small cache overfetch from the internal HDF5 metadata
  reader.

``data_bytes_read``
  Coordinate or chunk payload bytes. For contiguous uncompressed datasets this
  is the exact coordinate byte count for the requested bead range.

``bytes_read``
  The total of those categories.

Why HTTP Range requests matter
------------------------------

The backend sends HTTP requests with headers such as:

.. code-block:: text

   Range: bytes=123456-123575

The server must return:

.. code-block:: text

   206 Partial Content

If the server returns ``200 OK`` to a Range request, CNDBTools rejects the
response. A ``200 OK`` response often means the server ignored the byte range
and may be sending the whole file. Rejecting that response prevents accidental
large downloads.

What the embedded index does
----------------------------

Indexed CNDB/HDF5 files contain:

.. code-block:: text

   /_index
   root attribute: _index_offset

``/_index`` stores a gzip-compressed JSON map of HDF5 object offsets. The root
``_index_offset`` attribute tells the reader where the ``/_index`` object is in
the HDF5 file. This lets CNDBTools jump directly to the metadata for paths such
as:

.. code-block:: text

   /replica1_chr1/spatial_position/1

Without that embedded object index, a remote HDF5 reader would have to walk the
file structure remotely, which is unsafe and inefficient for very large files.

How coordinate bytes are computed
---------------------------------

For a contiguous uncompressed frame dataset:

.. code-block:: text

   shape = (n_beads, 3)
   dtype = float32
   itemsize = 4
   row_size = 3 * itemsize = 12 bytes

For beads ``start:stop``:

.. code-block:: text

   byte_start = data_offset + start * row_size
   byte_length = (stop - start) * row_size

So for beads ``0:10``:

.. code-block:: text

   byte_length = 10 * 3 * 4 = 120 bytes

For the ENCODE ``replica1_chr1`` frame with 4,980 beads:

.. code-block:: text

   full frame = 4980 * 3 * 4 = 59,760 coordinate bytes

This is why selected bead ranges can be much cheaper than reading full frames.

Why small reads may not always feel faster
------------------------------------------

Remote reads have HTTP latency. A 10-bead read and a 1,000-bead read may take
similar wall time if each requires one HTTP request. The benefit is that the
coordinate payload is much smaller, and the savings become larger when reading
many frames or repeatedly analyzing selected loci.

Use an index cache for repeated work:

.. code-block:: python

   tools = CndbTools.from_remote(
       h5_url=ENCODE_URL,
       trajectory="replica1_chr1",
       index_cache_path="/tmp/ENCFF161DID.embedded-index.json.gz",
   )

The first open may read the embedded index from the remote file. Later opens
can reuse the cache and skip that remote index read.

Contiguous and non-contiguous bead selections
---------------------------------------------

Contiguous selections are most efficient:

.. code-block:: python

   tools.xyz(frames=[1], beadSelection=range(0, 100))

Sparse selections are coalesced when possible:

.. code-block:: python

   tools.xyz(frames=[1], beadSelection=[0, 1, 2, 10, 11, 12])

The backend can read two ranges, ``0:3`` and ``10:13``, then reassemble the
requested order in memory. If a sparse selection would require too many tiny
HTTP requests, CNDBTools may read the smallest enclosing range and subset in
memory. ``stream_stats()`` reports the strategy and any overfetch.

Chunked and compressed datasets
-------------------------------

Contiguous uncompressed frame datasets are the exact-byte fast path. Chunked
datasets are different because HDF5 chunks are the smallest decodable unit.
When a bead range intersects a chunk, the backend may need to read the whole
chunk and then slice the requested rows.

Supported chunk filters include:

- no filter
- gzip/deflate
- shuffle + gzip
- shuffle + gzip + Fletcher32

Unsupported custom filters are rejected rather than decoded incorrectly.

Quick checklist
---------------

For safe remote streaming, the remote file should:

- be HDF5/CNDB/SW
- be hosted by an HTTP endpoint that returns ``206 Partial Content`` to Range
  requests
- contain ``/_index`` and ``_index_offset``
- contain compatible coordinate datasets shaped ``(n_beads, 3)``
- use contiguous uncompressed frames, or supported chunked/gzip filter
  pipelines

Inspect a URL before opening it:

.. code-block:: bash

   python scripts/inspect_structural_file.py --url URL
   python scripts/inspect_structural_file.py --url URL --json
