.. _installation:

============
Installation
============

Run the notebooks from this branch
----------------------------------

Use Git and Conda. A CPU is sufficient for the short tutorial runs. From a
directory where you want to keep the clone, run:

.. code-block:: bash

   git clone --branch cndbtools-integration-tutorial-audit --single-branch https://github.com/junioreif/OpenMiChroM.git
   cd OpenMiChroM
   conda env create --solver classic -f environment.yml
   conda activate openmichrom-tutorials

The environment installs this checkout in editable mode, with OpenMM,
Jupyter, and the tools for testing and building the documentation. No second
pip install is needed. To confirm which code is loaded:

.. code-block:: bash

   python -c "import OpenMiChroM; print(OpenMiChroM.__file__)"
   git rev-parse --short HEAD

The printed path should point into this clone. In VS Code, open the cloned
folder, install the Python and Jupyter extensions if needed, open a notebook
under ``Tutorials/``, and select the Conda environment above as its kernel.
See :doc:`validation` for short CPU runs and the full test command. For the
papers to cite with your results, see :doc:`/Reference/citing`.

Install a published release
---------------------------

If you do not need the branch tutorials, install the latest published package
in a Python 3.10+ environment using **one** of these alternatives:

.. code-block:: bash

   python -m pip install OpenMiChroM

or:

.. code-block:: bash

   conda install -c conda-forge openmichrom

These commands install the packaged release, not a development branch.
OpenMM, NumPy, SciPy, h5py, pandas, and scikit-learn are declared runtime
dependencies. For GPU-specific OpenMM setup, see the
`OpenMM installation guide <https://docs.openmm.org/latest/userguide/application/01_getting_started.html>`_.
