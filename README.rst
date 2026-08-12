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

The full 139 GB CNDB file is not downloaded. The embedded HDF5 index is read
once and can be cached locally. Coordinate reads use HTTP Range requests, and
contiguous bead ranges are read with exact byte ranges. Non-contiguous bead
selections may read the smallest enclosing bead range and then subset in memory.
When type metadata is present in the index, remote and local workflows both
populate ``ChromSeq`` and ``dictChromSeq``.
The embedded HDF5 metadata reader includes MIT-licensed vendored components from
``hdf5-indexed-reader``/``pyfive`` under ``OpenMiChroM/_cndb_stream/_vendor``.

Current writers store ``format="cndb"`` and ``format_version="1.0.0"`` as
authoritative HDF5 attributes. Readers warn and accept legacy files without a
version (or with a 0.x version), accept supported 1.x files, and reject malformed
or future-major versions before reading coordinates.

An ``index_cache_path`` is optional. When supplied, the parsed embedded index is
written atomically as a persistent JSON.gz file and is owned by the caller; it is
not deleted automatically. Without that option, metadata caching is in memory
only and is released by ``close()``. Use ``with CndbTools().load(...) as tools``
or call ``tools.close()`` to release local files and remote readers promptly.

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

Developer validation
====================

Create the recorded environment and install the editable package:

::

      CONDA_NO_PLUGINS=true conda env create --solver classic -f environment.yml
      conda activate openmichrom-cndbtools-integration-py310

Run frequent offline checks or the complete pre-review suite:

::

      python scripts/validate.py fast
      python scripts/validate.py complete

Both modes are CPU-safe and offline by default. Add ``--network`` for the live
ENCODE integration check or ``--gpu`` to require the OpenMM CUDA platform. The
complete notebook suite uses explicitly documented reduced validation step
counts; ``--full-science --network`` selects the production tutorial workloads.

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

    **OpenMiChroM** relies on the `OpenMM <http://openmm.org/>`_ API to run
    chromatin dynamics simulations. Install the standard package for CPU use,
    or the CUDA extra on a supported NVIDIA system:

    .. code-block:: bash

        pip install openmm
        # or: pip install "openmm[cuda12]"

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

- `Python <https://www.python.org/>`__ (>=3.10)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
- `h5py <https://www.h5py.org/>`__ (>=2.0.0)
- `pandas <https://pandas.pydata.org/>`__ (>=1.0.0)
- `scikit-learn <https://scikit-learn.org/>`__ (>=0.20.0)
