Structural File I/O Design
==========================

Overview
--------

OpenMiChroM now has a small structural-file I/O layer around CNDBTools. The
goal is to keep established local CNDBTools behavior intact while adding safe
remote streaming, local format detection, and small local format conversions.

The main entry points are:

.. list-table::
   :header-rows: 1

   * - API
     - Purpose
   * - ``CndbTools.open(...)``
     - Auto-detect and open local or safely streamable remote structural files
   * - ``CndbTools.from_remote(...)``
     - Explicitly open a remote indexed HDF5/CNDB file with HTTP Range requests
   * - ``convert_structure_file(...)``
     - Convert small local structural files among supported formats
   * - ``SaveStructure(..., mode="cndb")``
     - Write CNDB v2 files with ``/Header`` and an embedded object index

Detection
---------

``OpenMiChroM._structural_io.detect_structural_file`` inspects local files
directly and remote files conservatively. Remote detection uses HEAD requests
and small HTTP Range probes. A remote HDF5 file is streamable only when the
server returns ``206 Partial Content`` and the file exposes an embedded object
index.

Current local layouts include:

.. list-table::
   :header-rows: 1

   * - Layout
     - Description
   * - ``openmichrom-simple-cndb``
     - Historical root-level CNDB layout with ``/types`` and numeric frames
   * - ``openmichrom-cndb-v2``
     - Root-level CNDB layout with ``/Header`` and embedded ``/_index``
   * - ``nested-ndb-swb``
     - HDF5 layout with ``/<trajectory>/spatial_position/<frame>``
   * - ``text-ndb``
     - Text NDB records with ``MODEL`` and ``CHROM`` entries

Remote streaming
----------------

The internal ``OpenMiChroM._cndb_stream`` backend reads an embedded HDF5 object
index and then uses HTTP Range requests for dataset metadata and coordinate
payloads. Coordinate byte reads are exact for contiguous, uncompressed
``(n_beads, 3)`` frame datasets.

The backend refuses ``200 OK`` responses to Range requests. This is a safety
rule: a server that ignores ``Range`` may send the whole file.

Writing CNDB v2
---------------

The CNDB v2 writer keeps coordinate data in the historical root numeric frame
layout and adds:

.. code-block:: text

   /Header
   /_index
   _index_offset

``/_index`` contains gzip-compressed JSON object offsets. ``_index_offset`` is
the object-header offset of ``/_index`` and allows remote readers to find the
index without walking the whole HDF5 file.

Conversion
----------

``OpenMiChroM._structural_io.converters`` uses a simple in-memory trajectory
representation:

.. code-block:: python

   frames: OrderedDict[str, numpy.ndarray]
   types: list[str]
   genomic_positions: numpy.ndarray | None

This keeps conversions explicit and small. The converters are intended for
local files and small fixtures, not for streaming or rewriting very large
remote datasets.

Supported conversions in this initial layer:

.. list-table::
   :header-rows: 1

   * - Conversion
     - Notes
   * - ``ndb -> cndb``
     - Writes CNDB v2 by default
   * - ``cndb -> ndb``
     - Supports simple root-level CNDB
   * - ``ndb -> pdb``
     - Writes CA-like bead records
   * - ``pdb -> ndb``
     - Supports simple ATOM/HETATM PDB files
   * - ``sw/swb -> ndb``
     - Supports HDF5 nested ``spatial_position`` layouts

Limitations
-----------

- Converters are local-file only and do not download remote files.
- PDB conversion is intentionally simple and uses residue names as approximate
  bead type labels.
- Text SpaceWalk conversion is not implemented yet.
- Chunked/compressed HDF5 coordinate frames are detected but not decoded by the
  streaming byte-range backend.
- Large production conversions should be handled with explicit workflows so
  memory use is visible to the user.
