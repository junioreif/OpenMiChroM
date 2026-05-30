OpenMiChroM
===================

OpenMiChroM.ChromDynamics
-------------------------------

.. automodule:: OpenMiChroM.ChromDynamics
   :members:
   :undoc-members:
   :show-inheritance:

OpenMiChroM.Optimization
-------------------------------

.. automodule:: OpenMiChroM.Optimization
   :members:
   :undoc-members:
   :show-inheritance:
   
OpenMiChroM.CndbTools
-------------------------------

.. automodule:: OpenMiChroM.CndbTools
   :members: 
   :undoc-members: 
   :show-inheritance:

Writing streamable CNDB files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New OpenMiChroM CNDB trajectories are written in a stream-friendly CNDB v2
layout by default. The coordinate data stay in the historical root-level format
(``/types`` plus numeric frame datasets such as ``/0`` and ``/1``), while a
``/Header`` metadata group and embedded HDF5 object index make the file
directly streamable over HTTP Range requests after it is hosted.

.. code-block:: python

   sim.createReporters(
       statistics=True,
       traj=True,
       trajFormat="cndb",
       interval=1000,
       trajIndexed=True,
       trajMetadata=True,
   )

When using ``SaveStructure`` directly, call ``close()`` after writing frames so
``n_frames``, ``/_index``, and ``_index_offset`` are finalized. Use
``trajIndexed=False`` or ``SaveStructure(..., indexed=False)`` only when the
historical minimal CNDB layout is required.

Streaming remote CNDB files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CndbTools`` can stream remote indexed CNDB/HDF5 files with an internal indexed
HDF5 backend. Existing local CNDBTools behavior is unchanged; local
``cndbTools().load("trajectory.cndb")`` continues to use ``h5py``.

Install OpenMiChroM:

.. code-block:: bash

   pip install OpenMiChroM

Example:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   ENCODE_URL = "https://encode-public.s3.amazonaws.com/2023/02/02/7f75d816-342a-4b49-adbd-aaa499dc5201/ENCFF161DID.cndb"

   tools = CndbTools.from_remote(
       h5_url=ENCODE_URL,
       trajectory="replica1_chr1",
       index_cache_path="ENCFF161DID.embedded-index.json.gz",
   )

   coords = tools.xyz(
       frames=[1, 10, 100],
       beadSelection=range(0, 100),
   )

   print(coords.shape)
   print(tools.stream_stats())

The full 139 GB CNDB file is not downloaded. The embedded index is read once and
can be cached locally. Contiguous bead ranges are fetched with exact HTTP Range
requests. Non-contiguous bead selections are coalesced into small byte ranges
when practical, with a safe fallback to the smallest enclosing range. Remote
streaming currently focuses on coordinate access; metadata such as ``types`` and
``dictChromSeq`` is populated when the remote embedded index exposes small
metadata datasets. The embedded HDF5 metadata reader includes MIT-licensed
vendored components from ``hdf5-indexed-reader``/``pyfive`` under
``OpenMiChroM/_cndb_stream/_vendor``.

Structural file detection
~~~~~~~~~~~~~~~~~~~~~~~~~

``CndbTools.open(...)`` adds a conservative routing layer for local and remote
structural files:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools

   tools = CndbTools.open("trajectory.cndb")

The lower-level detector can be used to inspect files before opening them:

.. code-block:: python

   from OpenMiChroM._structural_io import detect_structural_file

   info = detect_structural_file("trajectory.cndb")
   print(info.file_type, info.layout, info.direct_streaming_supported)

For remote files, detection uses HEAD and small HTTP Range probes. It does not
download the full file. Remote HDF5 streaming is enabled only when the endpoint
supports ``206 Partial Content`` and an embedded index is available. Non-indexed
remote HDF5 files are rejected by default.

Structural file conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMiChroM includes small local converters for common structural-file
interoperability tasks:

.. code-block:: python

   from OpenMiChroM.CndbTools import CndbTools, convert_structure_file

   convert_structure_file("trajectory.ndb", "trajectory.cndb")
   convert_structure_file("trajectory.cndb", "trajectory.ndb")
   CndbTools.convert("trajectory.ndb", "trajectory.pdb")

Supported conversions currently include simple ``ndb -> cndb``,
``cndb -> ndb``, ``ndb -> pdb``, simple ``pdb -> ndb``, and supported HDF5
``sw/swb -> ndb``. Converters are local-file only and do not download remote
files.


OpenMiChroM.Integrators
-------------------------------

.. automodule:: OpenMiChroM.Integrators
   :members: 
   :undoc-members: 
   :show-inheritance:
