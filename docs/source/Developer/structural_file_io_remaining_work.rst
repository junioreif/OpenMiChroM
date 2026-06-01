Structural File I/O Remaining Work
==================================

This note records the state of the structural-file I/O branch after the first
integration pass and before additional completion work. It is meant to help
future maintainers see which behavior is implemented, which behavior is
deliberately conservative, and where each part of the implementation lives.

Current working features
------------------------

``OpenMiChroM.CndbTools`` is the public entry point for users. Existing local
``cndbTools().load(...)`` behavior is preserved and still uses ``h5py.File``
for historical root-level CNDB files. New routing is provided by
``CndbTools.open(...)``:

- local simple CNDB/HDF5 files route to the existing ``load(...)`` path;
- local nested HDF5 CNDB/SW/NDB-style files route through the local indexed
  backend;
- local text NDB files route through ``NDBTextReader``;
- remote HDF5 files stream only when they have an embedded object index and the
  server honors HTTP Range requests with ``206 Partial Content``.

The internal remote streaming backend lives in ``OpenMiChroM._cndb_stream``.
The important modules are:

- ``reader.py``: ``IndexedCNDB`` public-internal reader, byte accounting,
  trajectory selection, frame reads, and analysis helpers.
- ``embedded.py``: remote embedded-index loading, JSON.gz index cache, pyfive
  metadata access, and lazy frame metadata resolution.
- ``remote.py``: strict HTTP Range reader that rejects ``200 OK`` Range
  responses.
- ``index.py``: local JSON index builder for simple root-level and nested
  ``spatial_position`` layouts.
- ``embedded_writer.py``: CNDB v2 ``/Header`` metadata and embedded
  ``/_index``/``_index_offset`` writing.
- ``_vendor/``: MIT-licensed hdf5-indexed-reader/pyfive subset used for remote
  HDF5 metadata inspection.

Structural detection, local readers, selection helpers, and converters live in
``OpenMiChroM._structural_io``:

- ``detect.py``: local/remote format detection, HDF5 layout inspection, Range
  probing, and embedded-index support classification.
- ``formats.py``: ``StructuralFileInfo`` data model.
- ``readers.py``: local text NDB reader used by ``CndbTools.open(...)`` and
  converters.
- ``selections.py``: bead-index coalescing helper used by streaming
  ``xyz(...)``.
- ``converters.py``: clean-room, local-file converters among small NDB, CNDB,
  PDB, and supported HDF5 SW/SWB layouts.

CNDB writing is integrated into ``OpenMiChroM.CustomReporter.SaveStructure``.
``OpenMiChroM.ChromDynamics.createReporters(...)`` exposes the CNDB v2 writer
options that create ``/Header``, ``/_index``, and ``_index_offset`` metadata.

Current test coverage
---------------------

The core behavior is covered by small local fixtures:

- ``tests/test_cndbtools_stream.py`` covers local CNDBTools compatibility,
  internal streaming backend packaging, remote ``from_remote(...)`` routing,
  contiguous streaming reads, sparse bead coalescing, fallback overfetch, and
  stream byte-accounting.
- ``tests/test_cndbtools_encode_stream.py`` is skipped by default and validates
  a real ENCODE streaming smoke test when
  ``OPENMICHROM_RUN_ENCODE_STREAM_TESTS=1`` is set.
- ``tests/test_indexed_cndb_writer.py`` covers CNDB v2 writer metadata,
  embedded index writing, local detection, and remote Range smoke tests against
  small fixtures.
- ``tests/test_structural_io_detect.py`` covers local and mocked remote
  structural-file detection, including unsafe ``200 OK`` Range behavior.
- ``tests/test_structural_io_converters.py`` covers tiny NDB/CNDB/PDB/SW
  conversion paths and rejects remote conversion inputs.
- ``tests/test_structural_io_integration.py`` is skipped by default and probes
  public URLs when ``OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION=1`` is set.

Known gaps and risks
--------------------

Chunked and compressed HDF5 datasets remain more complex than contiguous
datasets. The remote reader computes exact byte offsets for contiguous,
uncompressed ``(n_beads, 3)`` datasets. Chunked uncompressed and gzip-compressed
datasets can be read through the embedded pyfive backend, which locates
intersecting chunks, reads whole chunk payloads, applies supported filters, and
reassembles requested rows. This can transfer more bytes than the requested bead
payload. Unsupported filters such as scale-offset, SZIP, or user filters should
remain unsupported until they are explicitly decoded and tested. The safe
behavior is to raise a clear unsupported-layout error rather than downloading
the full file.

Remote non-indexed HDF5 files are intentionally conservative. If a remote CNDB
or SW file lacks ``/_index``/``_index_offset``, the code does not walk the whole
HDF5 object graph remotely. If a server ignores Range requests and returns
``200 OK``, OpenMiChroM rejects streaming to avoid accidental full-file
downloads. Any future download fallback must require explicit user opt-in and a
size limit.

Converters are local-file only and in-memory. They are suitable for tiny
fixtures and small interoperability tasks, not large production rewrites. PDB
conversion is approximate because bead type labels and genomic coordinates are
not native PDB concepts. Text SpaceWalk conversion is not implemented yet
because the exact text dialect needs representative examples and should remain
clean-room. NDB-Converters has no license file in the inspected copy, so no
code has been copied from it.

Public URL status
-----------------

The current public examples are classified as follows. These statuses are
endpoint-dependent and should be regenerated before publication or production
use.

.. list-table::
   :header-rows: 1

   * - Example
     - Current classification
   * - NDB Rice older/non-indexed CNDB files
     - Detected as HDF5/CNDB, but tested endpoints have returned ``200 OK`` to
       Range requests. They are unsafe for direct streaming from those URLs.
   * - ENCODE indexed CNDB
     - Range-supported HDF5 with embedded index. Direct remote coordinate
       streaming is supported for contiguous uncompressed frame datasets.
   * - NCBI Mammoth Direct SW
     - Range-supported indexed HDF5/SW endpoint. Direct inspection/streaming is
       supported when the selected coordinate datasets are contiguous and
       uncompressed.
   * - NCBI GEO download SW
     - Download endpoint may return ``200 OK`` to Range requests. Use a direct
       file URL that honors Range requests when available.
   * - Bintu NDB
     - Detected as text NDB. Remote full-text parsing is not attempted by
       default because it would download the file.

Continuation branch goals
-------------------------

The continuation branch should keep local CNDBTools and current ENCODE remote
streaming stable while completing practical hardening tasks:

- document and test non-contiguous bead coalescing behavior;
- improve selection byte-accounting names and diagnostics where possible;
- expand safe remote non-indexed handling without enabling accidental
  downloads;
- add a small command-line converter wrapper;
- add reproducible public URL compatibility documentation/reporting;
- decide whether chunked uncompressed HDF5 streaming can be implemented safely
  in this pass; if not, keep chunked/compressed support detection-only and
  documented;
- keep all tutorials output-free, synced, and small.
