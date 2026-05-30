Structural File I/O Changelog
=============================

This changelog summarizes the structural-file I/O work added around CNDBTools.

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

- Remote streaming does not decode chunked or compressed HDF5 coordinate
  datasets.
- Converters are intended for local files and small fixtures.
- Text SpaceWalk conversion is not implemented yet.
- PDB conversion is approximate and focused on simple CA-like bead records.
