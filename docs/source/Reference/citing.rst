========================
How to cite OpenMiChroM
========================

If you use OpenMiChroM for simulation or analysis, cite the **software paper**:
Oliveira Junior et al., *A scalable computational approach for simulating
complexes of multiple chromosomes*, Journal of Molecular Biology **433**,
166700 (2021), `doi:10.1016/j.jmb.2020.10.034 <https://doi.org/10.1016/j.jmb.2020.10.034>`__.
Report the installed version; for an unreleased checkout, also report the
commit from ``git rev-parse HEAD``.

Add references for the methods and data you actually used:

- **MiChroM energy function:** Di Pierro et al. introduced the transferable
  model of interphase chromosome architecture :cite:p:`di_pierro_transferable_2016`.
- **Parameter optimization:** cite Oliveira Junior et al. for training
  compartment and genomic-distance interactions from Hi-C
  :cite:p:`oliveira_downsampled_2021`. For the full-inversion (FI-Chrom)
  workflow that infers locus-pair interactions, cite the FI-Chrom paper
  :cite:p:`oliveira_full_inversion_2026`.
- **NDB and CNDB files:** cite the Nucleome Data Bank resource
  :cite:p:`contessoto_nucleome_2021`. If you use a deposited trajectory,
  also identify its accession and credit the dataset that supplied it.
- **Chicken DT40 and entry into mitosis:** cite Contessoto et al. for modeling
  chromosome 7 across the cell cycle :cite:p:`contessoto_cell_cycle_2025` and
  Gibcus et al. if you use the experimental time-resolved Hi-C data
  :cite:p:`gibcus_mitotic_2018`. For the axial pulling protocol demonstrated
  in the tutorial, cite Ruben et al. :cite:p:`ruben_pulling_2023`.

See the :doc:`full references <references>` or
:download:`download the BibTeX file <OpenMiChroM.bib>`.

    

  
