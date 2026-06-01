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

Advanced polish branch plan
---------------------------

The ``feature/structural-io-polish-advanced`` branch should improve remaining
limitations without weakening the safety guarantees of the completed structural
I/O branch.

Unsupported HDF5 filters
~~~~~~~~~~~~~~~~~~~~~~~~

Current behavior:
  Contiguous uncompressed, chunked uncompressed, and chunked gzip-compressed
  coordinate datasets are supported. Unknown or unsupported HDF5 filters raise
  an error during remote chunk decoding.

Why it matters:
  Public HDF5 files may use shuffle, Fletcher32, LZF, scale-offset, SZIP, or
  site-specific plugin filters. Returning incorrect coordinates would be worse
  than refusing to stream.

Available metadata:
  The vendored pyfive backend exposes filter id, filter name, client data
  values, filter mask, ``shuffle``, ``fletcher32``, ``compression`` and
  ``compression_opts`` for chunked datasets. Built-in filter ids observed in the
  vendored code include gzip/deflate ``1``, shuffle ``2``, Fletcher32 ``3``,
  SZIP ``4``, n-bit ``5``, scale-offset ``6``, and LZF ``32000``.

Plan:
  Add a small internal filter helper that documents and tests the supported
  decode path for raw chunks: no filter, gzip/deflate, shuffle plus gzip, and
  Fletcher32 validation when present. Keep SZIP, scale-offset, n-bit, LZF
  without its optional codec, and unknown filters as explicit unsupported
  layouts.

Tests needed:
  Synthetic chunked datasets for uncompressed, gzip, shuffle+gzip, and
  gzip+Fletcher32. Compare remote Range-backed reads with local ``h5py`` slices.

Feasibility:
  Feasible for gzip, shuffle+gzip, and Fletcher32 because the vendored backend
  already contains the necessary logic. Unknown plugin filters remain out of
  scope.

Text SpaceWalk conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

Current behavior:
  HDF5 SW/SWB layouts with nested ``spatial_position`` groups can be converted
  locally. Text SpaceWalk-like files are detected from small samples but not
  converted.

Why it matters:
  Users may have older text SpaceWalk files and want NDB/CNDB interoperability.

Risks:
  SpaceWalk text dialects are less well pinned down in this branch than
  CNDB/NDB. NDB-Converters was inspected only as inspiration because the
  available copy has no license file; no code should be copied.

Plan:
  Search the repository for representative text SW examples. If a minimal,
  unambiguous ``trace``-style format is present, implement a clean-room parser
  and tests. Otherwise, improve the unsupported error and document what example
  data is needed.

Tests needed:
  Tiny synthetic text SW fixtures if the format is implemented, including
  ``sw -> ndb`` and ``sw -> cndb`` conversions.

Feasibility:
  Conditional. Detection-only documentation is safer if representative examples
  are unavailable.

Converter memory use
~~~~~~~~~~~~~~~~~~~~

Current behavior:
  Converters use an in-memory ``StructureTrajectory`` representation and are
  intended for small local files.

Why it matters:
  Large CNDB/SW/NDB conversions can consume substantial memory if every frame is
  loaded before writing.

Plan:
  Add explicit frame and bead selection options plus a memory guard
  (``max_memory_mb`` and ``allow_large``). Improve HDF5-to-text conversion so
  selected frames can be written frame-by-frame where feasible. Keep full
  streaming rewrite of arbitrary text formats out of scope unless it remains
  simple.

Tests needed:
  Frame subset, bead subset, and memory-guard tests with tiny HDF5 fixtures.

Feasibility:
  Feasible for local HDF5/CNDB/SW inputs. Large text NDB streaming can remain
  future work if it complicates the clean reader.

Approximate PDB conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

Current behavior:
  NDB-to-PDB writes one CA-like bead per residue and PDB-to-NDB parses simple
  ATOM/HETATM records.

Why it matters:
  PDB is often used for visualization. Coarse-grained bead semantics should be
  clear and output formatting should be stable.

Plan:
  Tighten PDB formatting, preserve model records, chain id, residue numbering,
  atom name, residue name, and element where possible. Add explicit options for
  atom/residue naming if they can be threaded without disrupting existing
  converter calls.

Tests needed:
  Multi-model PDB parsing, formatting assertions, and coordinate round trips
  with tolerances.

Feasibility:
  Feasible for simple coarse-grained PDB files. Exact biological residue
  semantics remain out of scope.

Setuptools warnings
~~~~~~~~~~~~~~~~~~~

Current behavior:
  ``python -m build`` succeeds but reports pre-existing warnings about the
  deprecated license classifier and ``OpenMiChroM.share`` package discovery.

Why it matters:
  A quieter build is easier to trust before a public PR.

Plan:
  Update packaging metadata only if doing so is safe for the existing
  distribution. Ensure ``OpenMiChroM.share/MiChroM.ff`` remains included.

Tests needed:
  ``python -m build``, ``python -m twine check dist/*``, and wheel/sdist
  content inspection.

Feasibility:
  Likely feasible, but packaging changes should be small and reversible.
