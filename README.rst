OpenMiChroM
===========

|Citing OpenMiChroM| |PyPI| |conda-forge| |ReadTheDocs| |NDB| |Update| |Downloads| |GitHub-Stars|

.. |Citing OpenMiChroM| image:: https://img.shields.io/badge/cite-OpenMiChroM-informational
   :target: https://open-michrom.readthedocs.io/en/latest/Reference/citing.html
.. |PyPI| image:: https://img.shields.io/pypi/v/OpenMiChroM.svg
   :target: https://pypi.org/project/OpenMiChroM/
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/OpenMiChroM.svg
   :target: https://anaconda.org/conda-forge/OpenMiChroM
.. |ReadTheDocs| image:: https://readthedocs.org/projects/open-michrom/badge/?version=latest
   :target: https://open-michrom.readthedocs.io/en/latest/
.. |NDB| image:: https://img.shields.io/badge/NDB-Nucleome%20Data%20Bank-informational
   :target: https://ndb.rice.edu/
.. |Update| image:: https://anaconda.org/conda-forge/openmichrom/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/openmichrom
.. |Downloads| image:: https://anaconda.org/conda-forge/openmichrom/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/OpenMiChroM
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/junioreif/OpenMiChroM.svg?style=social
   :target: https://github.com/junioreif/OpenMiChroM

`Documentation <https://open-michrom.readthedocs.io/>`__
| `Install <https://open-michrom.readthedocs.io/en/latest/GettingStarted/installation.html>`__
| `Tutorials <https://open-michrom.readthedocs.io/en/latest/Tutorials/Tutorial_Single_Chromosome.html>`__
| `Forum <https://groups.google.com/g/open-michrom>`__

Overview
========

`OpenMiChroM <https://www.sciencedirect.com/science/article/pii/S0022283620306185>`_ is a Python library for performing chromatin dynamics simulations and analyses. OpenMiChroM uses the `OpenMM <http://openmm.org/>`_ Python API employing the `MiChroM (Minimal Chromatin Model) <https://www.pnas.org/content/113/43/12168>`_ energy function. The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures that are consistent with experimental Hi-C maps. OpenMiChroM also allows simulations of a single or multiple chromosome chains using high-performance computing on different platforms (GPUs and CPUs). It is a highly flexible framework that can be extended for chromatin modeling and simulations across different species and for general biomolecular simulations.

.. image:: https://raw.githubusercontent.com/junioreif/OpenMiChroM/main/docs/source/images/OpenMiChroM_intro_small.jpg
   :align: center
   :height: 300px

The chromatin dynamics simulations can be performed for different human cell lines, cell phases (interphase to metaphase), and various organisms from `DNAzoo <https://www.dnazoo.org/>`_. Chromatin subcompartment annotations are available at the `NDB (Nucleome Data Bank) <https://ndb.rice.edu/>`_. The OpenMiChroM package accepts the chromatin sequence of compartments and subcompartments as input to create and simulate a chromosome polymer model. Examples of running the simulations and generating *in silico* Hi-C maps can be found `here <https://open-michrom.readthedocs.io/en/latest/Tutorials/Tutorial_Single_Chromosome.html>`_.

.. image:: https://raw.githubusercontent.com/junioreif/OpenMiChroM/main/docs/source/images/A549_NDB.jpg
   :align: center
   :height: 300px

Usage
=====

The following code snippet shows how to generate a single chromosome polymer model and run a chromatin dynamics simulation using OpenMiChroM:

::

      from OpenMiChroM.ChromDynamics import MiChroM
      sim = MiChroM(name='stomach_GRCh38', temperature=1.0, timeStep=0.01)
      sim.setup(platform="cuda")
      sim.saveFolder('stomach_GRCh38_chr10_simulation')
      sim.buildClassicMichrom(ChromSeq='inputs/stomach_GRCh38.bed', chromosome='chr10')

      sim.createReporters(statistics=True, traj=True, outputName=None, trajFormat="cndb", energyComponents=True, interval=10**3)
      sim.run(nsteps=10**5, report=True, interval=10**4)

Writing streamable CNDB files
=============================

New OpenMiChroM CNDB trajectories include a small ``/Header`` metadata group
and an embedded HDF5 object index by default. The coordinate layout remains
compatible with existing CNDBTools workflows: ``/types`` plus root-level numeric
frame datasets such as ``/0``, ``/1``, and ``/2``. The embedded index is stored
as a gzip-compressed JSON dataset at ``/_index`` and the root attribute
``_index_offset`` points to that dataset's HDF5 object header.

This makes newly written CNDB files directly streamable when hosted on an HTTP
server that supports Range requests:

::

      sim.createReporters(
          statistics=True,
          traj=True,
          trajFormat="cndb",
          interval=10**3,
          trajIndexed=True,
          trajMetadata=True,
      )

      # If using SaveStructure directly, call close() when writing is done so
      # n_frames and the embedded index are finalized.

Use ``trajIndexed=False`` or ``SaveStructure(..., indexed=False)`` only when
you need to write the historical minimal CNDB layout without ``/Header`` or
``/_index``.

Streaming remote CNDB files
===========================

Remote CNDB streaming is included in CNDBTools. CNDBTools uses an internal
indexed HDF5 streaming backend to read embedded CNDB indexes and fetch
coordinate byte ranges over HTTP. This extends CNDBTools without changing
existing local ``h5py`` workflows.

Install OpenMiChroM:

::

      pip install OpenMiChroM

Example using a real ENCODE CNDB file:

::

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

The full 139 GB CNDB file is not downloaded. The embedded HDF5 index is read
once and can be cached locally. Coordinate reads use HTTP Range requests, and
contiguous bead ranges are read with exact byte ranges. Non-contiguous bead
selections are coalesced into small byte ranges when practical, with a safe
fallback to the smallest enclosing bead range. ``tools.stream_stats()`` reports
both historical byte counters and explicit coordinate-selection diagnostics such
as ``requested_coordinate_bytes``, ``transferred_coordinate_bytes``,
``overfetch_coordinate_bytes``, ``range_request_count``, and
``selection_strategy``. Remote streaming currently focuses on coordinate access;
metadata such as ``types`` and ``dictChromSeq`` is populated when the remote
embedded index exposes small metadata datasets.
The embedded HDF5 metadata reader includes MIT-licensed vendored components from
``hdf5-indexed-reader``/``pyfive`` under ``OpenMiChroM/_cndb_stream/_vendor``.

Remote files that do not stream safely are rejected by default. For small remote
files only, users may explicitly opt into a size-limited download fallback:

::

      tools = CndbTools.open(
          "https://example.org/small-old-style.cndb",
          allow_download=True,
          max_download_size_mb=50,
      )

This fallback downloads the remote file to a local temporary path and then uses
the normal local CNDBTools readers. It should not be used for large public CNDB
or SW files.

Structural file I/O
===================

CNDBTools includes an experimental structural-file detection layer for local and
remote CNDB, NDB, and SpaceWalk/SW-style files:

::

      from OpenMiChroM.CndbTools import CndbTools
      from OpenMiChroM._structural_io import detect_structural_file

      info = detect_structural_file("trajectory.cndb")
      print(info.file_type, info.layout, info.direct_streaming_supported)

      tools = CndbTools.open("trajectory.cndb")

For remote files, detection uses HEAD and small HTTP Range probes and does not
download the full file. Remote HDF5 streaming is enabled only when the endpoint
supports ``206 Partial Content`` and an embedded index is available. Non-indexed
remote HDF5 files are rejected by default so users do not accidentally download
large public datasets.

Command-line inspection is also available:

::

      python scripts/inspect_structural_file.py --path trajectory.cndb
      python scripts/inspect_structural_file.py --url https://example.org/file.cndb

Small local structural conversions are also available:

::

      from OpenMiChroM.CndbTools import convert_structure_file

      convert_structure_file("trajectory.ndb", "trajectory.cndb")
      convert_structure_file("trajectory.cndb", "trajectory.ndb")
      convert_structure_file("trajectory.ndb", "trajectory.pdb")
      convert_structure_file("trajectory.spw", "trajectory.ndb")

Command-line conversion uses the same local converter:

::

      python scripts/convert_structure_file.py trajectory.ndb trajectory.cndb

Converters are local-file only and are intended for small interoperability
workflows. They do not download remote files.

Known structural I/O limitations
================================

- Direct remote coordinate streaming requires an embedded HDF5 object index and
  coordinate datasets with shape ``(n_beads, 3)``.
- Contiguous, uncompressed coordinate datasets use exact byte ranges and are the
  fastest path.
- Chunked uncompressed and gzip-compressed coordinate datasets can be streamed
  through the embedded backend, including gzip with shuffle and Fletcher32
  checksum filters. Whole chunks may be transferred and reported as coordinate
  overfetch.
- Unsupported HDF5 filters are detected at read time and raise an error instead
  of downloading the full file. This includes scale-offset, SZIP, n-bit,
  unavailable LZF codecs, and unknown custom filters.
- Remote non-indexed HDF5 files are not streamed by default; users should
  download/index them locally or host an indexed version.
- Structural converters are local-file only.
- PDB conversion is simple and approximate, intended for CA-like bead records.
- Simple text SpaceWalk ``.sw/.spw`` conversion is supported for trace-style
  files with ``chromosome start end x y z`` rows. Text SpaceWalk does not carry
  OpenMiChroM chromatin type labels, so converted beads are assigned ``UN``.
- Large structural conversions should be run explicitly because they load the
  converted trajectory into memory.

Resources
=========

- `Reference Documentation <https://open-michrom.readthedocs.io/>`__: Examples, tutorials, and class details.
- `Installation Guide <https://open-michrom.readthedocs.io/en/latest/GettingStarted/installation.html>`__: Instructions for installing **OpenMiChroM**.
- `GitHub repository <https://github.com/junioreif/OpenMiChroM/>`__: Download the **OpenMiChroM** source code.
- `Issue tracker <https://github.com/junioreif/OpenMiChroM/issues>`__: Report issues/bugs or request features.

Tutorial notebook synchronization
=================================

Canonical user-facing tutorial notebooks live under the top-level
``Tutorials/`` folder. Sphinx uses flattened copies under
``docs/source/Tutorials/``. Keep them synchronized with:

::

      python scripts/sync_tutorials.py

Before committing tutorial changes, check consistency with:

::

      python scripts/sync_tutorials.py --check

The sync script copies mapped ``.ipynb`` files, skips checkpoint files, preserves
notebook metadata and cell IDs, and refuses source notebooks with very large
embedded outputs.

Citation
========

When using **OpenMiChroM** for chromatin dynamics simulations or analyses, please `use this citation <https://open-michrom.readthedocs.io/en/latest/Reference/citing.html>`_.  
We also thank `Polychrom <https://github.com/open2c/polychrom>`_, where part of this code was inspired. You can use this `citation <https://zenodo.org/records/3579473>`_.

Installation
============

The **OpenMiChroM** library can be installed via `pip <https://pypi.org/project/OpenMiChroM/>`__, `conda <https://conda.io/projects/conda/>`__, or compiled from source.

Install via pip
---------------

The code below will install **OpenMiChroM** from PyPI:

::

    pip install OpenMiChroM

.. note::

    **OpenMiChroM** relies on the `OpenMM <http://openmm.org/>`_ API to run the chromatin dynamics simulations.
    
    OpenMM is now available as a pip-installable package. You can install it using pip openmm[cuda12] to iinstall to use with GPU's or openmm to install for CPU's only:

    .. code-block:: bash

        pip install openmm[cuda12]

    Alternatively, if you prefer to use conda, install OpenMM from the `conda-forge channel <https://conda-forge.org/>`_ with:

    .. code-block:: bash

        conda install -c conda-forge openmm

Install via conda
-----------------

If you prefer using conda, you can install **OpenMiChroM** from
`conda-forge <https://anaconda.org/conda-forge/OpenMiChroM>`__ with the following command:

::

    conda install -c conda-forge OpenMiChroM

.. hint::
    
    Sometimes, the installation via conda may appear to be stuck. If this happens, update conda/anaconda using the command below and try installing **OpenMiChroM** again.

::

    conda update --prefix /path/to/anaconda3/ anaconda

Required Libraries
------------------

The following libraries are **required** for installing **OpenMiChroM**:

- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
- `h5py <https://www.h5py.org/>`__ (>=2.0.0)
- `pandas <https://pandas.pydata.org/>`__ (>=1.0.0)
- `scikit-learn <https://scikit-learn.org/>`__ (>=0.20.0)
