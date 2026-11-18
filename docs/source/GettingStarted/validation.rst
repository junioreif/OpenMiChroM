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
and synchronization, the full offline pytest suite, and all fifteen canonical
notebooks in reduced CPU mode. The structural-
variant and loop-extrusion notebooks use small deterministic synthetic fixtures;
they do not download the biological datasets that motivated the workflows.

Numerical changes in the 2.0 candidate
--------------------------------------

* Training now accumulates the contact map of the current frame, including the
  first one. Previously a one-frame IC training run could accumulate a zero
  map. Normalizing Hi-C contact probabilities no longer doubles symmetric
  off-diagonal entries. Recompute previously trained parameters before
  comparing fitted values across versions.
* A custom ideal-chromosome (IC) table starts at ``dinit``; the last requested
  separation ``dend`` is included when physically possible. Compact tables are
  no longer silently truncated. Coordinate trajectories can change if an
  earlier run used one of the affected coefficients.
* Radial density profiles now use exact shell volumes. For comparison with
  older profiles pass ``normalization="legacy"``: :doc:`/Reference/cndb_stream`.

Complete validation
-------------------

Before review, run the complete offline suite:

.. code-block:: bash

   python scripts/validate.py complete

This adds a Sphinx build with warnings treated as errors, source/wheel builds, an isolated
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
<https://github.com/junioreif/OpenMiChroM/tree/prep-2.0/Tutorials/DT40_Annealing_50kb>`_
are included in the clone; no additional download is needed.

To run the full DT40 schedule from a **straight chain** on an OpenCL GPU,
set ``RUN = "slow"`` and ``PLATFORM = "OpenCL"`` in the first code cell
of the notebook. This cools the chain over 50,000,000 steps and then holds
it at reduced temperature 0.2 for 5,005,000 steps. The archived 15-minute
structure is used only for comparison. Check that the simulation cell
reports ``OpenCL`` as the backend; the default notebook run is on CPU.

For planning, allow **about two hours** for the full schedule on an Apple
M3 Max GPU (OpenMM 8.5.2, OpenCL, single precision). This estimate uses
the GPU portion of a run that started on CPU: the first 1,000,000 steps
were on CPU and the remaining 54,005,000 took 2 hours 1 minute on GPU.
This is not an end-to-end CPU timing or a benchmark for other devices.
For a quick installation
check, the default ``compact`` notebook took about 13 seconds on this Mac,
including kernel startup and plotting.

In this single trajectory, averaging the orientation correlations over all
1,001 cold frames before the Fourier transform gave dominant periods of
0.765 and 1.529 Mb. The 20 frames sampled by the notebook gave the same
two peak bins. Neither dominant period matched the 0.322 and 3.670 Mb
peaks of the archived structure. The
:download:`example spectra (CSV) <../../../Tutorials/DT40_Annealing_50kb/example_results/slow_spectrum.csv>`
give spatial frequency in cycles per Mb and both Fourier amplitudes.
These values describe one historically measured trajectory, not a new
production run of the 2.0 candidate or a convergence test. Its 800-row IC
input gives exactly the same 733 active coefficients (separations 3 through
735) under both table implementations.

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
ENCODE smoke test. Use ``--public-sources`` to also check the current NDB and
NCBI endpoints with native TLS certificate verification. Availability, redirects,
and byte-range support depend on those servers. Default validation is offline and uses OpenMM's CPU
platform. For a separate ten-step GPU check, add ``--gpu-platform OpenCL``
on an OpenCL GPU, or ``--gpu-platform CUDA`` (also spelled ``--gpu``) on a
CUDA GPU. The check creates a context on the requested device, tests for
finite energies and forces, and verifies that the particles move. It fails
if the requested backend is unavailable; it does not switch the tutorial
suite from CPU to GPU.
