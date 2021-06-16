.. _installation:

============
Installation
============

The **OpenMiChroM** library can be installed via `conda <https://conda.io/projects/conda/>`_ or `pip <https://pypi.org/>`_, or compiled from `source (GitHub) <https://github.com/junioreif/OpenMiChroM>`_

Install via conda
-----------------

The code below will install **OpenMiChroM** from `conda-forge <https://anaconda.org/conda-forge/OpenMiChroM>`_.

.. code-block:: bash

    conda install -c conda-forge OpenMiChroM
    
.. hint:: Often, the installation via conda happens to be stuck. If this is the case, it is recommended to update conda/anaconda using the command line below and try **OpenMiChroM** again.

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
