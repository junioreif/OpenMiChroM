Public Structural File Compatibility
====================================

This page records the public structural-file examples used while validating the
OpenMiChroM structural I/O layer. The table is intentionally conservative:
support means the endpoint can be inspected with small probes and, for HDF5
files, can stream safely without accepting ``200 OK`` responses to Range
requests.

Regenerating the report
-----------------------

Put one source per line in a text file and run:

.. code-block:: bash

   python scripts/inspect_structural_file.py --urls public_structural_urls.txt --json-output report.json

Remote probes use HEAD and small HTTP Range requests. They do not download the
full file unless another tool is used explicitly.

Current public examples
-----------------------

.. list-table::
   :header-rows: 1

   * - Example
     - URL
     - Current status
     - Notes
   * - Harris et al. NDB Rice CNDB
     - ``https://ndb.rice.edu/d/Harris_etal_NatComm_2023-LCL_chr7_39.5-42.5/chr7_39.5-42.5_REP1.cndb``
     - Detected as CNDB/HDF5, direct remote streaming unsupported from tested endpoint.
     - Older/non-indexed CNDB endpoint may return ``200 OK`` to Range requests.
   * - Mello et al. NDB Rice CNDB
     - ``https://ndb.rice.edu/d/Mello_etal_2026_GM12878_500-frames/GM12878-chr01-500-frames.cndb``
     - Detected as CNDB/HDF5, direct remote streaming unsupported from tested endpoint.
     - Requires Range support and embedded index for direct streaming.
   * - Oliveira Jr. NDB Rice CNDB
     - ``https://ndb.rice.edu/d/Oliveira_Jr-C1-multichain_2020/C1_1_multichain_5.7Gb.cndb``
     - Detected as CNDB/HDF5, direct remote streaming unsupported from tested endpoint.
     - Do not allow implicit download for this large file.
   * - ENCODE indexed CNDB
     - ``https://www.encodeproject.org/files/ENCFF764BAH/@@download/ENCFF764BAH.cndb``
     - Direct remote streaming supported when endpoint responds to probes.
     - Range-supported HDF5 with embedded object index and contiguous
       uncompressed coordinate datasets.
   * - NCBI Mammoth Direct SW
     - ``https://ftp.ncbi.nlm.nih.gov/geo/series/GSE268nnn/GSE268050/suppl/GSE268050%5FWoolly%5FMammoth%5FDirect%5FInv%2Esw``
     - Direct remote indexed HDF5/SW inspection and streaming supported when
       selected coordinates are contiguous and uncompressed.
     - Use the direct file URL rather than an HTML landing page.
   * - NCBI GEO download SW
     - ``https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE268050&format=file&file=GSE268050%5FWoolly%5FMammoth%5FMiChroM%2Esw``
     - Endpoint may be unsafe for direct streaming.
     - If it returns ``200 OK`` to Range requests, OpenMiChroM refuses remote
       streaming from that URL.
   * - Bintu et al. NDB
     - ``https://ndb.rice.edu/d/Bintu_etal_Science_2018/A549_chr21-28-30Mb.ndb``
     - Detected as text NDB.
     - Remote full-text parsing is not attempted by default because it would
       download the file.

Interpretation
--------------

Direct remote streaming requires all of the following:

- the server honors HTTP Range requests with ``206 Partial Content``;
- the HDF5 file exposes an embedded object index;
- the selected coordinate datasets are directly readable by the current backend.

If any condition is missing, OpenMiChroM reports the reason and refuses direct
remote streaming. Small files can be downloaded only with explicit
``allow_download=True`` and ``max_download_size_mb=...``.
