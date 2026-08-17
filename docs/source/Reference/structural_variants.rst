======================================
Structural variants and loop extrusion
======================================

OpenMiChroM provides two complementary public modules for locus-scale
workflows. :mod:`OpenMiChroM.StructuralVariants` transforms a symmetric contact
or effective-potential matrix and optional directional motif tracks.
:mod:`OpenMiChroM.Extrusion_Bonds` generates a seeded one-dimensional loop
trajectory and activates its bonds during a MiChroM simulation.

Coordinate conventions
======================

All array and bead coordinates in these APIs are **zero-based**.
Structural-variant intervals use Python's **half-open** convention
``[start, end)``: ``start`` is included, ``end`` is excluded, and the interval
contains ``end - start`` beads. A loop pair such as ``(4, 11)`` instead names
two individual zero-based bead indices; it is not an interval.

The optional text labels produced by ``locus_labels`` are one-based human-
readable names such as ``Locus1``. They are type labels for the MiChroM input
files, not array coordinates.

Apply a structural variant
==========================

``apply_structural_variant`` is the main entry point. It accepts the canonical
names ``deletion``, ``inversion``, and ``duplication`` (plus short aliases), and
returns a
:class:`~OpenMiChroM.StructuralVariants.StructuralVariantResult`. The
``index_map`` records which input row and column produced each output row and
column.

.. code-block:: python

   import numpy as np
   from OpenMiChroM import apply_structural_variant

   bead = np.arange(8)
   matrix = 1.0 / (1.0 + np.abs(bead[:, None] - bead[None, :]))
   forward = np.array([0.0, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
   reverse = np.array([0.0, 0.0, 0.0, 0.3, 0.7, 0.1, 0.0, 0.0])

   result = apply_structural_variant(
       matrix,
       "inversion",
       2,
       5,
       forward_motifs=forward,
       reverse_motifs=reverse,
   )

   assert result.matrix.shape == (8, 8)
   assert result.index_map.tolist() == [0, 1, 4, 3, 2, 5, 6, 7]
   assert np.allclose(result.matrix, result.matrix.T)

For an inversion, the selected interval is reversed and forward/reverse motif
directions are exchanged. Deletion removes the selected rows, columns, and
motif entries. Duplication inserts a tandem copy immediately after ``end``.
Directional motif tracks must be supplied as a pair and must match the matrix
size.

Ideal-chromosome adjustment and duplication
--------------------------------------------

``adjust_ideal_chromosome=True`` first subtracts the mean value at each genomic
separation, transforms the residual matrix, and restores that separation
profile. This can keep the distance-dependent background separate from the
rearranged residual signal. It is an explicit modeling choice, not a universally
correct preprocessing step.

For a tandem duplication, ``duplicate_contacts="copy"`` copies every contact
from its source locus. ``duplicate_contacts="ideal"`` leaves contacts involving
the inserted copy at zero, or at the restored separation background when ideal-
chromosome adjustment is enabled. The latter is intended for contacts that a
later optimization will infer.

Locus-matrix files
------------------

The transformed matrix can be written in the header-plus-dense-row format
consumed by ``MiChroM.addCustomTypes``. The matching sequence uses the same
unique labels. Writers publish files atomically and refuse to replace an
existing path unless ``overwrite=True`` is explicit. Matrix files must contain
only finite values, and labels cannot contain whitespace or control characters,
because both conditions would make the downstream type table unsafe or
ambiguous.

.. code-block:: python

   from OpenMiChroM import locus_labels, write_locus_matrix, write_locus_sequence

   labels = locus_labels(result.matrix.shape[0], prefix="Variant")
   write_locus_matrix("variant_lambdas.csv", result.matrix, labels=labels)
   write_locus_sequence("variant_sequence.txt", labels)

Use ``read_locus_matrix`` to read this matrix format back with its labels.

Generate a deterministic loop trajectory
========================================

``LoopExtrusionManager`` uses a local NumPy random generator. Passing ``seed``
makes the one-dimensional trajectory reproducible without changing NumPy's
global random state.

.. code-block:: python

   import numpy as np
   from OpenMiChroM import LoopExtrusionManager

   forward_fix = np.zeros(24)
   reverse_fix = np.zeros(24)
   forward_fix[8] = 0.8
   reverse_fix[15] = 0.8

   manager = LoopExtrusionManager(
       forward_fix,
       reverse_fix,
       num_steps=5,
       extruder_count=2,
       off_rate=0.0,
       seed=2026,
   )
   trajectory = manager.simulate()

   # Five transitions plus the initial state.
   assert trajectory.shape == (6, 2, 2)
   assert np.array_equal(trajectory, manager.simulate())

The two motif arrays give per-transition fixation probabilities for the left
and right extruder feet. Extruder anchors cannot collide or pass one another.
Nested loops are allowed; interleaving loops are rejected. ``num_steps`` counts
transitions, so the returned trajectory always has ``num_steps + 1`` frames.
When a boundary or collision unloads a motor, both of its anchors are reloaded.
If a dense configuration temporarily leaves no valid loading site, all motors
are reloaded together by the default ``reload_policy="collective"``. Set
``reload_policy="error"`` to stop at that state instead. The
``collective_reload_count`` attribute records how often the collective policy
was used. These collision and reloading rules are part of the model definition
and should be included in any scientific calibration.

Activate loop bonds in one OpenMM Context
=========================================

Add the full trajectory with
:meth:`~OpenMiChroM.ChromDynamics.MiChroM.addDynamicLoopPotential` after loading
the structure and before creating the simulation. The initial frame is active
when the force is created. Between simulation segments, select another frame
with :meth:`~OpenMiChroM.ChromDynamics.MiChroM.updateDynamicLoopPotential`
without reconstructing the System or Context:

.. code-block:: python

   # Configure ``sim``, load its structure, and add the other MiChroM forces.
   updater = sim.addDynamicLoopPotential(
       trajectory, k_loop=10.0, r0_loop=1.0
   )
   sim.createSimulation()

   sim.run(nsteps=100, report=False)  # trajectory frame 0
   for frame in range(1, trajectory.shape[0]):
       sim.updateDynamicLoopPotential(frame)
       sim.run(nsteps=100, report=False)

:class:`~OpenMiChroM.Extrusion_Bonds.LoopBondUpdater` constructs one
``CustomBondForce`` containing the union of all particle pairs that occur
anywhere in the trajectory. Each bond has an ``active`` parameter.
``updateDynamicLoopPotential`` changes those parameters and calls
``updateParametersInContext`` on the existing Context; the force topology is
not rebuilt. Advanced OpenMM users can instantiate ``LoopBondUpdater`` directly
and call ``create_force`` and ``set_step``.

Limitations and interpretation
==============================

* Matrix transformations rearrange an input model; they do not call structural
  variants, infer a new contact map, or establish that one matrix-adjustment
  assumption is biologically correct. Re-equilibration or optimization and
  comparison with independent data remain the user's responsibility.
* The synthetic fixtures in the tutorials are deterministic checks of API
  behavior. They are not measurements, fitted parameters, or evidence of
  scientific convergence.
* The extrusion trajectory is an effective stochastic model. One transition is
  not assigned a physical duration, and fixation, unloading, density, force,
  and integration parameters require calibration for a biological application.
* The bond force allocates one entry per unique pair in the trajectory. Memory
  and parameter-update cost therefore grow with the union of visited pairs.
* A NumPy seed reproduces the one-dimensional bond trajectory. Full OpenMM
  dynamics need not be bitwise identical across platforms or hardware.

Provenance and scientific context
=================================

Miles Gantcher contributed the first OpenMiChroM structural-variation and loop-
extrusion workflow in `pull request 123
<https://github.com/junioreif/OpenMiChroM/pull/123>`_. The maintained modules
turn that notebook-scale contribution into validated public APIs. The
`ACS poster by Miles Gantcher, José Onuchic, and Vinícius Contessoto
<https://acs.digitellinc.com/p/s/maximum-entropy-polymer-model-predicts-the-architectural-effects-of-genetic-structural-variations-poster-board-207-650669>`_
describes the associated structural-variation modeling context.

Relevant scientific sources include:

* Sanborn *et al.*, `Chromatin extrusion explains key features of loop and
  domain formation in wild-type and engineered genomes
  <https://doi.org/10.1073/pnas.1518552112>`_, PNAS (2015).
* Bianco *et al.*, `Polymer physics predicts the effects of structural variants
  on chromatin architecture <https://doi.org/10.1038/s41588-018-0098-8>`_,
  Nature Genetics (2018).
* Andrey *et al.*, `Characterization of hundreds of regulatory landscapes in
  developing limbs reveals two regimes of chromatin folding
  <https://doi.org/10.1101/gr.213066.116>`_, Genome Research (2017).

The tutorials intentionally replace the historical biological inputs with
small synthetic arrays so they run offline and expose every assumption. See
:doc:`../Tutorials/Tutorial_Apply_Structural_Variants` and
:doc:`../Tutorials/Tutorial_Loop_Formation`.

Python API
==========

.. currentmodule:: OpenMiChroM.StructuralVariants

.. autoclass:: StructuralVariantResult
   :members:
.. autofunction:: apply_structural_variant
.. autofunction:: delete_region
.. autofunction:: invert_region
.. autofunction:: duplicate_region
.. autofunction:: ideal_chromosome_profile
.. autofunction:: remove_ideal_chromosome
.. autofunction:: add_ideal_chromosome
.. autofunction:: locus_labels
.. autofunction:: read_locus_matrix
.. autofunction:: write_locus_matrix
.. autofunction:: write_locus_sequence

.. currentmodule:: OpenMiChroM.Extrusion_Bonds

.. autoclass:: LoopExtrusionManager
   :members: simulate, get_extrusion_bonds
.. autoclass:: LoopBondUpdater
   :members: active_pairs, create_force, particle_count, set_step

The historical spelling ``Loop_Extrusion_Manager`` remains a compatibility
alias, and ``MiChroM.addHarmonicLoopPotential`` remains a single-frame
compatibility wrapper. New dynamic workflows should use
``LoopExtrusionManager`` and ``MiChroM.addDynamicLoopPotential``.
