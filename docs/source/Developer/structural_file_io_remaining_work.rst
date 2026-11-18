Structural file I/O: practical limits
=====================================

Local reading and remote streaming have different requirements. A readable
local HDF5 file is not necessarily streamable from its public URL.

* For direct HTTP access the server must honor byte-range requests. An
  embedded object index or compatible sidecar must describe the selected
  trajectory. When a host ignores Range, download the file explicitly if you
  have enough disk space and then open it locally.
* Contiguous, uncompressed coordinate datasets allow exact byte-range reads.
  Chunked datasets may require entire chunks even for a single selected bead.
  Unsupported filters cannot be decoded by the streaming reader. See
  :doc:`structural_file_formats` and :doc:`public_structural_file_compatibility`.
* Local structural conversions assemble the selected trajectory in memory.
  Use frame and bead selections to limit the payload. The structural converter can materialize the input (including local HDF5)
  before applying the selection; the memory guard is not a bound on total
  process memory.
* PDB is a visualization format, not an archive for genomic annotations.
  Text SpaceWalk and HDF5 SpaceWalk are different formats; renaming the
  extension does not convert one into the other. Choose the eight-route
  :doc:`NDB converter </Reference/converters>` when preserving NDB chain,
  loop, and interval metadata is important.
* Byte counts, chunk overhead, and cache hits are available in
  ``stream_stats()``. For comparable measurements, distinguish cold index
  loading from repeated reads with a warm index cache:
  :doc:`remote_cndb_benchmark`.
