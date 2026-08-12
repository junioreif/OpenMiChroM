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

   tools = CndbTools().load(
       ENCODE_URL,
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
requests. Non-contiguous bead selections may read the smallest enclosing range
and then subset in memory. Remote type dictionaries are populated when the
selected index contains type metadata. The embedded HDF5 metadata reader includes
MIT-licensed vendored components from ``hdf5-indexed-reader``/``pyfive`` under
``OpenMiChroM/_cndb_stream/_vendor``.

See :doc:`Reference/cndb_stream` for format-version compatibility, caching,
resource cleanup, and deterministic testing details.


OpenMiChroM.Integrators
-------------------------------

.. automodule:: OpenMiChroM.Integrators
   :members: 
   :undoc-members: 
   :show-inheritance:
