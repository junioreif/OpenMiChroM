=====================
CNDB format and I/O
=====================

``CndbTools.load`` accepts either a local path (including
``os.PathLike`` objects) or an HTTP(S) URL. Strings are classified as remote
only when they contain a URL delimiter and pass strict scheme/host validation;
unusual local names such as ``archive:trajectory.cndb`` remain local paths.
``CndbTools.from_remote`` remains available as the explicit remote constructor.
The historical ``cndbTools`` class name and the preferred ``CndbTools`` name are
aliases for the same class.

Remote reads require a server that honors byte-range requests with ``206
Partial Content`` and a matching ``Content-Range`` header. A server that returns
``200 OK`` is rejected to prevent an accidental full-file download. Redirects
are followed only to another valid HTTP(S) URL. Short responses, missing files,
bad indexes, corrupt layouts, invalid frames, and unsupported versions raise
specific exceptions from ``OpenMiChroM._cndb_stream``.

Format metadata
===============

Classic writers (including ``OpenMiChroM.Converters.ndb_to_cndb``) store
the following attributes on the HDF5 root:

.. code-block:: text

   format = "cndb"
   format_version = "1.0.0"

Readers use semantic major-version compatibility:

* Missing metadata and explicit 0.x versions are read as legacy files with a
  ``LegacyCNDBVersionWarning``.
* Valid 1.x and 2.x versions are supported. The version 2 layout keeps numeric
  frame datasets and types, and adds a Header with
  ``format_name="OpenMiChroM-CNDB"`` and ``format_version="2.0"``.
  The reporter can embed an index for HTTP range reads; close it to finalize
  the index before sharing the file. See :doc:`/Tutorials/write_indexed_cndb`.
* Malformed versions, unknown format markers, and future major versions fail
  before coordinate data is returned.

The index has its own format, layout, and version fields. Both index and CNDB
versions declared by the index are validated. A user-supplied sidecar must
match the exact source file; rebuild it if that file has changed.

Radial analysis compatibility
=============================

``compute_RDP`` and ``compute_RadNumDens`` use exact spherical shell
volumes by default (``normalization="shell-volume"``). To reproduce older
profiles, pass ``normalization="legacy"`` explicitly. Report the convention
when comparing results across software versions.

Caching and cleanup
===================

Embedded metadata is cached in a bounded in-memory reader by default and is
discarded at ``close()``. If ``index_cache_path`` is supplied, the parsed object
index is written atomically as JSON.gz. That persistent file is caller-owned and
is intentionally not removed by ``close()``. Temporary ``.tmp`` cache files are
removed on both success and failure. ``index_path`` selects a caller-provided
local sidecar; ``index_url`` selects a remote sidecar.

Use a context manager whenever practical:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   with CndbTools().load(url, index_cache_path="trajectory.index.json.gz") as tools:
       coordinates = tools.xyz(frames=[1], beadSelection=range(100))
       print(tools.stream_stats())

``close()`` is idempotent for local and remote readers. Access after closure
fails explicitly. ``stream_stats()`` separates index, metadata, and coordinate
bytes so callers and tests can verify that only requested ranges were read.

Default remote tests use a temporary local HTTP range server and require no
network. The optional live ENCODE check is enabled by
``OPENMICHROM_RUN_ENCODE_STREAM_TESTS=1`` or ``scripts/validate.py --network``.
