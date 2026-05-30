Structural File I/O Credits and Provenance
==========================================

OpenMiChroM structural I/O combines established OpenMiChroM CNDBTools behavior
with new internal detection, streaming, writing, and conversion code. This page
records provenance so future maintenance is clear.

hdf5-indexed-reader and pyfive
------------------------------

The internal remote HDF5 metadata reader vendors MIT-licensed pyfive-based code
from ``hdf5-indexed-reader`` under:

.. code-block:: text

   OpenMiChroM/_cndb_stream/_vendor/hdf5_indexed_reader/

The vendored license file is preserved in that directory. OpenMiChroM uses this
backend to inspect HDF5 object headers and dataset metadata through byte-range
reads.

hdf5-indexer behavior
---------------------

The embedded CNDB index writer reimplements the small behavior used by
``hdf5-indexer``:

- walk HDF5 groups with pyfive
- map group paths to child object-header offsets
- gzip the JSON object-offset map
- store it as an opaque HDF5 dataset named ``/_index``
- store the ``/_index`` object-header offset in root attribute
  ``_index_offset``

The implementation in OpenMiChroM is internal and does not depend on the
external ``hdf5-indexer`` package.

cndb-stream prototype
---------------------

The internal streaming backend grew from the separate ``cndb-stream`` prototype
developed to validate remote byte-range CNDB access against real ENCODE CNDB
files. That prototype demonstrated:

- embedded HDF5 index loading without a sibling checkout
- strict HTTP ``206 Partial Content`` enforcement
- exact coordinate byte reads for contiguous, uncompressed frame datasets
- index caching and byte accounting

The useful backend behavior has now been incorporated into OpenMiChroM
CNDBTools as internal code, so users do not need to install a separate
``cndb-stream`` package.

NDB-Converters inspiration
--------------------------

The structural converter layer was inspired by the practical need addressed by
NDB conversion utilities. No NDB-Converters source code was copied into
OpenMiChroM because no license file was available at the time of this work.

The converter implementation in ``OpenMiChroM._structural_io.converters`` was
written clean-room from OpenMiChroM's own file examples and the public textual
shape of NDB/PDB/CNDB records.

Core dependencies
-----------------

The structural I/O layer also uses standard project dependencies:

- Python standard-library ``urllib`` and ``http.server`` for HTTP Range access
  and tests
- ``NumPy`` for coordinate arrays
- ``h5py`` for local HDF5/CNDB writing and inspection

License notes
-------------

Vendored third-party license files must remain in the source tree and package
distributions. New OpenMiChroM structural I/O code follows the repository's
license.
