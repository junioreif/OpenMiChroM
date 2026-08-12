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
and synchronization, and two executable notebook smoke tests.

Before review, run the complete offline suite:

.. code-block:: bash

   python scripts/validate.py complete

This adds the full pytest suite, every canonical notebook in reduced CPU mode,
a Sphinx build with warnings treated as errors, source/wheel builds, an isolated
wheel install, and installed-package import checks. Temporary notebook copies,
HTTP servers, documentation output, and wheel-install directories are cleaned
automatically.

Reduced mode executes every notebook cell while replacing production simulation
counts with small deterministic counts. It validates program flow and basic
shapes, not scientific convergence. To request the original production counts
and external data, use ``python scripts/validate.py complete --full-science
--network``; this can require many hours. Add ``--network`` alone for the live
ENCODE smoke test. Add ``--gpu`` to require OpenMM's CUDA platform. Default
validation is fully offline and uses OpenMM's CPU platform.
