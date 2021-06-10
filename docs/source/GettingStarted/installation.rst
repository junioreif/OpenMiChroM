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

.. note::

The **Open-MiChroM** library uses OpenMM API to run the Chromatin dynamics simulations.
These requirements can be met by installing the OpenMM package from the `conda-forge channel <https://conda-forge.org/>`__:

.. code-block:: bash

    conda install -c conda-forge openmm
    
    
The following are libraries **required** for installing **Open-MiChroM**:

- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
- `six <https://pypi.org/project/six/>`__ (>=1.14.0)
- `h5py <https://www.h5py.org/>`__ (>=2.0.0)
- `pandas <https://pandas.pydata.org//>`__ (>=1.0.0)


