============
OpenMiChroM
============

|Citing OpenMiChroM|
|PyPI|
|conda-forge|
|ReadTheDocs|
|NDB|
|Update|
|Downloads|
|GitHub-Stars|

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

`OpenMiChroM <https://www.sciencedirect.com/science/article/pii/S0022283620306185>`_ is a Python library for performing chromatin dynamics simulations and analyses. Open-MiChroM uses the  `OpenMM <http://openmm.org/>`_ Python API employing the `MiChroM (Minimal Chromatin Model) <https://www.pnas.org/content/113/43/12168>`_ energy function. The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures that are consistent with experimental Hi-C maps. OpenMiChroM also allows simulations of a single or multiple chromosome chain using High-Performance Computing in different platforms (GPUs and CPUs).

.. raw:: html

    <p align="center">
    <img align="center" src="./docs/source/images/OpenMiChroM_intro_small.jpg" height="300px">
    </p>

The chromatin dynamics simulations can be performed for different human cell lines, cell phases (interphase to metaphase), and different organisms from  `DNAzoo <https://www.dnazoo.org/>`_. Chromatin subcompartment annotations are available at the  `NDB (Nucleome Data Bank) <https://ndb.rice.edu/>`_.
Open-MiChroM package receives the chromatin sequence of compartments and subcompartments as input to create and simulate a chromosome polymer model. Examples of running the simulations and generating the *in silico* Hi-C maps can be found `here <https://open-michrom.readthedocs.io/en/latest/Tutorials/Tutorial_Single_Chromosome.html>`_

.. raw:: html

    <p align="center">
    <img height="300px"  src="./docs/source/images/A549_NDB.jpg" >
    </p>

Resources
=========

- `Reference Documentation <https://open-michrom.readthedocs.io/>`__: Examples, tutorials, and class details.
- `Installation Guide <https://open-michrom.readthedocs.io/en/latest/GettingStarted/install.html>`__: Instructions for installing **OpenMiChroM**.
- `GitHub repository <https://github.com/junioreif/OpenMiChroM/>`__: Download the **OpenMiChroM** source code.
- `Issue tracker <https://github.com/junioreif/OpenMiChroM/issues>`__: Report issues/bugs or request features.


Citation
========

When using **OpenMiChroM** to perform chromatin dynamics simulations or analyses, please `use this citation
<https://open-michrom.readthedocs.io/en/latest/Reference/citing.html>`__.


Installation
============

The **OpenMiChroM** library can be installed via `conda <https://conda.io/projects/conda/>`_ or pip, or compiled from source.

Install via conda
-----------------

The code below will install **OpenMiChroM** from `conda-forge <https://anaconda.org/conda-forge/OpenMiChroM>`_.

.. code-block:: bash

    conda install -c conda-forge OpenMiChroM

.. hint:: Often, the installation via conda happens to be stuck. If this is the case, it is recommended to update conda/anaconda using the command line below and try to install **OpenMiChroM** again.

.. code-block:: bash

    conda update --prefix /path/to/anaconda3/ anaconda


Install via pip
-----------------

The code below will install **OpenMiChroM** from `PyPI <https://pypi.org/project/OpenMiChroM/>`_.

.. code-block:: bash

    pip install OpenMiChroM

.. note::

The **OpenMiChroM** library uses `OpenMM <http://openmm.org/>`_ API to run the chromatin dynamics simulations.
These requirements can be met by installing the OpenMM package from the `conda-forge channel <https://conda-forge.org/>`__:

.. code-block:: bash

    conda install -c conda-forge openmm
    
    
The following are libraries **required** for installing **OpenMiChroM**:

- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
- `six <https://pypi.org/project/six/>`__ (>=1.14.0)
- `h5py <https://www.h5py.org/>`__ (>=2.0.0)
- `pandas <https://pandas.pydata.org/>`__ (>=1.0.0)
- `scikit-learn <https://scikit-learn.org/>`__ (>=0.20.0)
