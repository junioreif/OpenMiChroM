Structural File I/O Changelog
=============================

This changelog summarizes the structural-file I/O work added around CNDBTools.

Merge note: Structural file I/O and CNDBTools streaming
-------------------------------------------------------

This branch adds structural file I/O and CNDBTools streaming without requiring
an external ``cndb-stream`` dependency. Highlights:

- ``CndbTools.open(...)`` for conservative local/remote structural-file
  detection and routing.
- Remote indexed CNDB/SW streaming through the internal HDF5 byte-range backend.
- Local CNDB, nested HDF5/SW, and text NDB reading paths.
- CNDB v2 writer metadata with ``/Header``, ``/_index``, and
  ``_index_offset``.
- Small local structural converters.
- Student-facing tutorials and developer provenance documentation.

Continuation branch completion
------------------------------

The completion pass added:

- clearer remaining-work and public URL compatibility documentation;
- robust sparse bead-selection coalescing diagnostics in ``stream_stats()``;
- explicit, size-limited remote download fallback for small non-streamable
  files;
- a local ``scripts/convert_structure_file.py`` command-line converter wrapper;
- chunked uncompressed and gzip-compressed HDF5 coordinate streaming through
  the embedded backend, with overfetch accounting;
- a structural file detection tutorial synced into the Sphinx docs.

Structural detection and opening
--------------------------------

- Added ``detect_structural_file(...)`` for local and remote structural files.
- Added ``CndbTools.open(...)`` as a conservative auto-routing entry point.
- Kept historical local ``cndbTools().load("file.cndb")`` behavior intact.
- Added local text NDB reading through an internal lightweight reader.
- Added local nested HDF5/SW-style coordinate reading.

Remote streaming
----------------

- Added internal ``OpenMiChroM._cndb_stream`` backend.
- Added ``CndbTools.from_remote(...)``.
- Added strict HTTP Range request handling with ``206 Partial Content``
  enforcement.
- Added embedded HDF5 object-index loading.
- Added exact coordinate byte reads for contiguous, uncompressed
  ``(n_beads, 3)`` frame datasets.
- Added optional embedded-index cache and byte-accounting diagnostics.
- Added non-contiguous bead selection coalescing for streaming ``xyz(...)``.

CNDB v2 writer
--------------

- Added ``/Header`` metadata for newly written CNDB files.
- Added embedded object-index writing at ``/_index``.
- Added root ``_index_offset`` attribute.
- Added ``trajIndexed`` and ``trajMetadata`` reporter options.
- Added detection label ``openmichrom-cndb-v2``.

Converters
----------

- Added ``OpenMiChroM._structural_io.converters``.
- Added public ``convert_structure_file(...)`` import from
  ``OpenMiChroM.CndbTools``.
- Added ``CndbTools.convert(...)`` convenience wrapper.
- Added small local conversions:

  - ``ndb -> cndb``
  - ``cndb -> ndb``
  - ``ndb -> pdb``
  - simple ``pdb -> ndb``
  - supported HDF5 ``sw/swb -> ndb``

Tutorials and documentation
---------------------------

- Added remote CNDB streaming tutorial.
- Added indexed CNDB writer tutorial.
- Added structural file conversion tutorial.
- Added developer documentation for design, file formats, credits, and
  changelog.

Known limitations
-----------------

- Contiguous, uncompressed HDF5 coordinate datasets remain the fastest exact
  byte-range streaming path.
- Chunked uncompressed and gzip-compressed HDF5 coordinate datasets may read
  whole chunks and report coordinate overfetch.
- Unsupported HDF5 filters are rejected rather than decoded or downloaded
  wholesale.
- Converters are intended for local files and small fixtures.
- Text SpaceWalk conversion is not implemented yet.
- PDB conversion is approximate and focused on simple CA-like bead records.
