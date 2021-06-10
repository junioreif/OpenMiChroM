============
Open-MiChroM
============

`Open-MiChroM <https://www.sciencedirect.com/science/article/pii/S0022283620306185>`_ is a Python library for performing chromatin dynamics simulations and analyses. Open-MiChroM uses the  `OpenMM <http://openmm.org/>`_ Python API employing the `MiChroM (Minimal Chromatin Model) <https://www.pnas.org/content/113/43/12168>`_ energy function. The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures that are consistent with experimental Hi-C maps. Open-MiChroM also allows simulations of a single or multiple chromosome chain using High-Performance Computing in different platforms (GPUs and CPUs).

.. raw:: html

    <p align="center">
    <img align="center" src="./docs/source/images/OpenMiChroM_intro_small.jpg" height="400px">
    </p>

The chromatin dynamics simulations can be performed for different human cell lines, cell phases (interphase to metaphase), and different organisms from  `DNAzoo <https://www.dnazoo.org/>`_. Chromatin subcompartment annotations are available at the  `NDB (Nucleome Data Bank) <https://ndb.rice.edu/>`_.
Open-MiChroM package receives the chromatin sequence of compartments and subcompartments as input to create and simulate a chromosome polymer model. Examples of running the simulations and generating the *in silico* Hi-C maps can be found `here <../Tutorials/single_chain.html>`_

.. raw:: html

    <p align="center">
    <img height="400px"  src="./docs/source/images/A549_NDB.jpg" >
    </p>

