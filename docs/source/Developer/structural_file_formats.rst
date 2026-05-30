Structural File Formats
=======================

OpenMiChroM CNDB v2
-------------------

OpenMiChroM writes CNDB trajectory files as HDF5. The CNDB v2 writer keeps the
historical coordinate layout so existing local readers remain usable:

.. code-block:: text

   /Header
   /types
   /0
   /1
   /2
   ...
   /_index

``/types`` contains bead type labels. Numeric root-level datasets contain frame
coordinates with shape ``(n_beads, 3)``. For direct remote streaming, these
frame datasets must be contiguous and uncompressed.

Metadata header
---------------

The ``/Header`` group stores small metadata attributes:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``format_name``
     - ``OpenMiChroM-CNDB``
   * - ``format_version``
     - ``2.0``
   * - ``creator``
     - ``OpenMiChroM``
   * - ``openmichrom_version``
     - Writer package version when available
   * - ``creation_time``
     - UTC ISO timestamp
   * - ``coordinate_units``
     - Coordinate units, currently ``nanometer``
   * - ``coordinate_dtype``
     - Stored coordinate dtype
   * - ``frame_layout``
     - ``root_numeric_frames``
   * - ``n_beads``
     - Number of beads per frame
   * - ``n_frames``
     - Number of written frames, finalized on close
   * - ``indexed``
     - Whether an embedded object index was written
   * - ``index_format``
     - ``hdf5-object-offset-json-gzip``

Embedded object index
---------------------

The embedded index follows the hdf5-indexer convention. It is a
gzip-compressed JSON object-offset map stored in an opaque HDF5 dataset named
``/_index``. The root attribute ``_index_offset`` stores the HDF5 object-header
offset for ``/_index``.

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

The JSON intentionally does not need to include ``/_index`` itself; the
``_index_offset`` root attribute is the remote entry point for finding it.

Remote streaming
----------------

``CndbTools.from_remote(...)`` reads the embedded index and selected HDF5
metadata through HTTP Range requests. For a contiguous, uncompressed coordinate
dataset, the backend computes the exact bead range:

.. code-block:: text

   byte_start = data_offset + start * 3 * dtype.itemsize
   byte_length = (stop - start) * 3 * dtype.itemsize

If a server ignores a Range request and returns ``200 OK`` instead of
``206 Partial Content``, OpenMiChroM raises an error rather than risking an
accidental full-file download.

Writer options
--------------

``SaveStructure`` accepts:

.. code-block:: python

   SaveStructure(..., mode="cndb", indexed=True, metadata=True)

``MiChroM.createReporters`` forwards:

.. code-block:: python

   sim.createReporters(
       traj=True,
       trajFormat="cndb",
       trajIndexed=True,
       trajMetadata=True,
   )

Call ``close()`` on a direct ``SaveStructure`` instance when writing is done.
That finalization updates ``n_frames``, writes ``/_index``, and records
``_index_offset``.

Current limitations
-------------------

Direct byte-range coordinate reads currently require contiguous, uncompressed
frame datasets with shape ``(n_beads, 3)``. Chunked or compressed frame
datasets can be detected, but they are not decoded by the streaming backend.
