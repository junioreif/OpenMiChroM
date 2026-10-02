====================
Developer validation
====================

The repository records the integration environment in ``environment.yml``.
From a fresh clone, create it and install OpenMiChroM in editable mode with:

.. code-block:: bash

   CONDA_NO_PLUGINS=true conda env create --solver classic -f environment.yml
   conda activate openmichrom-cndbtools-integration-py310

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

This adds the full pytest suite, all eleven canonical notebooks in reduced CPU mode,
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

Without fast mode, the simulation notebooks use production step counts, and
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
ENCODE smoke test. Add ``--gpu`` to require OpenMM's CUDA platform. Default
validation is fully offline and uses OpenMM's CPU platform.
