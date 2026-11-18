.. _installation:

============
Installation
============

Run the notebooks from this branch
----------------------------------

For the notebooks, `download the prep-2.0 ZIP
<https://github.com/junioreif/OpenMiChroM/archive/refs/heads/prep-2.0.zip>`_
and extract it. A CPU is sufficient for the short tutorial runs.
Open a terminal in the extracted folder containing ``environment.yml`` and run:

.. code-block:: bash

   conda env create --solver classic -f environment.yml
   conda activate openmichrom-tutorials

The ZIP includes all notebooks and their sample inputs. If you want to pull
updates or contribute using Git, clone the branch instead with
``git clone -b prep-2.0 --depth 1 https://github.com/junioreif/OpenMiChroM.git``.
Omit ``--depth 1`` only if you need the commit history, which contains old
simulation files. Both methods install the local source in editable mode, with OpenMM,
Jupyter, and the tools for testing and building the documentation. No second
pip install is needed. To confirm which code is loaded:

.. code-block:: bash

   python -c "import OpenMiChroM; print(OpenMiChroM.__file__)"

The printed path should point into your OpenMiChroM folder. In VS Code, open that
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
