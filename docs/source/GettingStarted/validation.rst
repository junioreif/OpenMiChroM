====================
Developer validation
====================

The repository records the integration environment in ``environment.yml``.
From a fresh clone, create it and install OpenMiChroM in editable mode with:

.. code-block:: bash

   conda env create --solver classic -f environment.yml
   conda activate openmichrom-tutorials

The fast offline suite is intended for frequent use:

.. code-block:: bash

   python scripts/validate.py fast

It checks all public imports, API compatibility, local CNDB access, deterministic
HTTP range streaming, format versions, a small CPU simulation, tutorial syntax
and synchronization, and five executable notebook smoke tests. The structural-
variant and loop-extrusion notebooks use small deterministic synthetic fixtures;
they do not download the biological datasets that motivated the workflows.

Before review, run the complete offline suite:

.. code-block:: bash

   python scripts/validate.py complete

This adds the full pytest suite, all twelve canonical notebooks in reduced CPU mode,
a Sphinx build with warnings treated as errors, source/wheel builds, an isolated
wheel install, and installed-package import checks. Temporary notebook copies,
HTTP servers, documentation output, and wheel-install directories are cleaned
automatically.

For interactive checks, open the repository in VS Code with its Python and
Jupyter extensions, select the environment above as the notebook kernel, and
open notebooks under ``Tutorials/`` (not the documentation copies). Restart the
kernel before choosing **Run All**. The input-based notebooks resolve their
bundled data from either the repository root or their own tutorial directory.
To use the same short CPU checks interactively, add this first cell before
running a notebook:

.. code-block:: python

   import os
   os.environ["OPENMICHROM_TUTORIAL_MODE"] = "fast"
   os.environ["OPENMICHROM_TUTORIAL_PLATFORM"] = "CPU"

The :doc:`DT40 annealing tutorial </Tutorials/Tutorial_DT40_Annealing_50kb>`
defaults to 2,600 steps on CPU. To compare longer trajectories from the same
archived 10-minute starting structure, set ``RUN = "extended"`` in the
notebook and repeat with different seeds (for example, 15050 and 15051).
Each replica runs 100,000 cooling steps and 30,000 cold steps; the notebook
reports peaks separately for the new trajectory and the archived 15-minute
structure. The archived coordinates yield periods of 0.32 and 3.67 Mb with
the notebook's Fourier analysis. These are a check on the bundled reference,
not expected values for an individual annealing trajectory. The `notebook and its five input files
<https://github.com/junioreif/OpenMiChroM/tree/cndbtools-integration-tutorial-audit/Tutorials/DT40_Annealing_50kb>`_
are included in the clone; no additional download is needed.

To run the full DT40 schedule from a **straight chain** on an OpenCL GPU,
set ``RUN = "slow"`` and ``PLATFORM = "OpenCL"`` in the first code cell
of the notebook. This cools the chain over 50,000,000 steps and then holds
it at reduced temperature 0.2 for 5,005,000 steps. The archived 15-minute
structure is used only for comparison. Check that the simulation cell
reports ``OpenCL`` as the backend; the default notebook run is on CPU.

Without fast mode, other simulation notebooks use production step counts, and
the remote CNDB and full-inversion examples can require network access (and
Java for the Juicer extraction step). Input-generation cells reset their files
on each run rather than appending to previous parameters.

Reduced mode executes every notebook cell while replacing production simulation
counts with small deterministic counts. It validates program flow and basic
shapes, not scientific convergence. The multi-chromosome smoke test expands its
spherical wall to contain the unequilibrated short-run structures; production
mode retains the density-derived radius and checks that collapse is sufficient
before enabling that wall. To request the original production counts
and external data, use ``python scripts/validate.py complete --full-science
--network``; this can require many hours. Add ``--network`` alone for the live
ENCODE smoke test. Default validation is offline and uses OpenMM's CPU
platform. For a separate ten-step GPU check, add ``--gpu-platform OpenCL``
on an OpenCL GPU, or ``--gpu-platform CUDA`` (also spelled ``--gpu``) on a
CUDA GPU. The check creates a context on the requested device, tests for
finite energies and forces, and verifies that the particles move. It fails
if the requested backend is unavailable; it does not switch the tutorial
suite from CPU to GPU.
