.. _installation:

============
Installation
============

Installing Open-MiChroM
================

The **Open-MiChroM** library can be installed via `conda <https://conda.io/projects/conda/>`_ or pip, or compiled from source.

Install via conda
-----------------

The code below will install **Open-MiChroM** from `conda-forge <https://anaconda.org/conda-forge/Open-MiChroM>`_.

.. code-block:: bash

    conda install -c conda-forge Open-MiChroM

Install via pip
-----------------

The code below will install **Open-MiChroM** from `PyPI <https://pypi.org/project/Open-MiChroM/>`_.

.. code-block:: bash

    pip install Open-MiChroM

Compile from source
-------------------

The following are **required** for installing **freud**:

- A C++14-compliant compiler
- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `Intel Threading Building Blocks <https://www.threadingbuildingblocks.org/>`__
- `Cython <https://cython.org/>`__ (>=0.29.14)
- `scikit-build <https://scikit-build.readthedocs.io/>`__ (>=0.10.0)
- `CMake <https://cmake.org/>`__ (>=3.6.3)

.. note::

    Any note here.

The **Open-MiChroM** library uses OpenMM API.
These requirements can be met by installing the following packages from the `conda-forge channel <https://conda-forge.org/>`__:

.. code-block:: bash

    conda install -c conda-forge openmm

Unit Tests
==========

The unit tests for **Open-MiChroM** are included in the repository and are configured to be run using the Python :mod:`pytest` library:

.. code-block:: bash

    # Run tests from the tests directory
    cd tests
    python -m pytest .