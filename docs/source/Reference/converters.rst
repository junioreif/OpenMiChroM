============================
NDB trajectory converters
============================

OpenMiChroM provides one maintained Python interface for the trajectory
conversions that previously lived as separate command-line scripts.  The
converter functions accept :class:`pathlib.Path` objects or path strings,
return the output path, validate the complete input before publishing the
result, and do not replace an existing file unless ``overwrite=True`` is
requested explicitly.

Supported routes
================

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Input
     - Output
     - Public function
   * - NDB
     - CNDB
     - ``ndb_to_cndb``
   * - CNDB
     - NDB
     - ``cndb_to_ndb``
   * - NDB
     - PDB
     - ``ndb_to_pdb``
   * - PDB
     - NDB
     - ``pdb_to_ndb``
   * - NDB
     - SpaceWalk (SPW)
     - ``ndb_to_spw``
   * - SpaceWalk (SPW)
     - NDB
     - ``spw_to_ndb``
   * - GROMACS coordinates (GRO)
     - NDB
     - ``gro_to_ndb``
   * - Bintu et al. CSV layout
     - NDB
     - ``csv_to_ndb``

The same routes are available through ``convert``, which infers formats from
the input and output suffixes, and as static convenience methods on
``CndbTools``.

.. code-block:: python

   from OpenMiChroM.Converters import convert, ndb_to_cndb
   from OpenMiChroM.CndbTools import CndbTools

   ndb_to_cndb("ensemble.ndb", "ensemble.cndb")
   convert("ensemble.ndb", "ensemble.pdb")
   CndbTools.convert("ensemble.cndb", "roundtrip.ndb")

The installed command-line entry point uses the same implementation and safety
checks:

.. code-block:: bash

   openmichrom-convert ensemble.ndb ensemble.cndb
   openmichrom-convert bintu.csv chromosome21.ndb --chromosome 21

Run ``openmichrom-convert --help`` for route-specific metadata options.

The historical ``CndbTools().ndb2cndb(stem)`` method remains available for
older scripts.  New code should use ``ndb_to_cndb`` or ``convert`` so that the
input and output paths and overwrite policy are explicit.

Python API
==========

.. currentmodule:: OpenMiChroM.Converters

.. autofunction:: convert
.. autofunction:: ndb_to_cndb
.. autofunction:: cndb_to_ndb
.. autofunction:: ndb_to_pdb
.. autofunction:: pdb_to_ndb
.. autofunction:: ndb_to_spw
.. autofunction:: spw_to_ndb
.. autofunction:: gro_to_ndb
.. autofunction:: csv_to_ndb

Format fidelity
===============

NDB and CNDB can represent chromatin types, multiple frames, chain structure,
genomic intervals, and loops.  The converter accepts both the current unknown
type label ``NA`` and the legacy label ``UN`` and writes the current label.
Numeric type codes in older CNDB files are accepted for compatibility.

The other formats do not carry every NDB field:

* PDB is primarily a coordinate/visualization format.  OpenMiChroM writes type
  hints for its own round trips, but generic PDB files may not identify every
  chromatin type or genomic interval.
* SpaceWalk text preserves traces, chromosomes, genomic intervals, and
  coordinates, but not chromatin types.  With ``write_loops=True`` (the
  default), the companion ``.loops`` file is atomically replaced on every
  export; an empty file explicitly means that the trajectory has no loops.
* GRO imports coordinates and recognized OpenMiChroM type labels.  Genomic
  intervals are reconstructed from the requested resolution.
* ``csv_to_ndb`` is specifically for the Bintu et al. table layout
  ``model,index,z,x,y``.  It is not a general-purpose CSV schema detector.

``.spw`` and ``.swb`` are different formats.  ``.spw`` is the SpaceWalk text
exchange format handled by these converters.  ``.swb`` is an HDF5 trajectory
format produced by the OpenMiChroM SWB reporter; changing only the suffix does
not convert one into the other.

Provenance
==========

The supported route inventory and historical format behavior were informed by
the `NDB-Converters repository <https://github.com/mellofariam/NDB-Converters>`_,
developed by Matheus Mello, Vinicius Contessoto, and Antonio B. Oliveira
Junior.  That repository does not declare a software license.  OpenMiChroM's
implementation is therefore a new library implementation; no source files
from that repository are copied into this package.

See :doc:`../Tutorials/Tutorial_NDB_Converters` for a complete offline example
that exercises every supported route and checks frame, bead, coordinate, and
type invariants.
