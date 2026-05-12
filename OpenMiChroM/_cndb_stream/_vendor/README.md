Vendored Dependencies
=====================

`OpenMiChroM._cndb_stream._vendor.hdf5_indexed_reader.pyfive` contains the
pyfive backend from `hdf5-indexed-reader`, used to inspect HDF5 object headers
through HTTP Range reads when a CNDB file contains an embedded hdf5-indexer
object index.

The upstream `hdf5-indexed-reader` project is MIT licensed. A copy of that
license is included at `hdf5_indexed_reader/LICENSE`.

Upstream project:

- https://github.com/jrobinso/hdf5-indexed-reader

The vendored backend is used only for HDF5 metadata inspection. Coordinate
payload reads are handled by OpenMiChroM's internal CNDB streaming backend with
strict byte-range requests.
