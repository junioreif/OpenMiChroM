.. _installation:

============
Installation
============

The **OpenMiChroM** library can be installed via
`pip <https://pypi.org/project/OpenMiChroM/>`_,
`conda <https://anaconda.org/conda-forge/OpenMiChroM>`_,
or compiled from `source (GitHub) <https://github.com/junioreif/OpenMiChroM>`_.

Install via pip
-----------------

The code below will install **OpenMiChroM** from PyPI:

.. code-block:: bash

    pip3 install OpenMiChroM

.. note::

    **OpenMiChroM** relies on the `OpenMM <http://openmm.org/>`_ API to run
    chromatin dynamics simulations. OpenMM can now also be installed via pip,
    which may be preferable for some users.    OpenMM is now available as a pip-installable package. You can install it using pip openmm[cuda12] to iinstall to use with GPU's or openmm to install for CPU's only:


    .. code-block:: bash

        pip install openmm[cuda12] 

    Alternatively, if you prefer using conda, install OpenMM from the
    `conda-forge channel <https://conda-forge.org/>`_ with:

    .. code-block:: bash

        conda install -c conda-forge openmm

Install via conda
-----------------

If you prefer using conda, you can install **OpenMiChroM** from
`conda-forge <https://anaconda.org/conda-forge/OpenMiChroM>`_ with the following command:

.. code-block:: bash

    conda install -c conda-forge OpenMiChroM

.. hint::
    
    Sometimes, the installation via conda may appear to be stuck. If this happens,
    update conda/anaconda using the command below and try installing **OpenMiChroM** again.

.. code-block:: bash

    conda update --prefix /path/to/anaconda3/ anaconda

Required Libraries
------------------

The following libraries are **required** for installing **OpenMiChroM**:

- `Python <https://www.python.org/>`_ (>=3.6)
- `NumPy <https://www.numpy.org/>`_ (>=1.14)
- `SciPy <https://www.scipy.org/>`_ (>=1.5.0)
- `six <https://pypi.org/project/six/>`_ (>=1.14.0)
- `h5py <https://www.h5py.org/>`_ (>=2.0.0)
- `pandas <https://pandas.pydata.org/>`_ (>=1.0.0)
- `scikit-learn <https://scikit-learn.org/>`_ (>=0.20.0)