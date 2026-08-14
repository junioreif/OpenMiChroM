# Tutorial and example audit

All canonical notebooks use Python 3.10, current OpenMM imports, the supported
OpenMiChroM API, CPU as the portable default, deterministic NumPy seed 2026,
portable relative paths, stable cell IDs, and explicit full/fast execution
modes. Documentation copies are generated from the canonical files with
`scripts/sync_tutorials.py --clear-outputs`.

Fast mode executes every cell; it reduces simulation counts and is a workflow
smoke test, not evidence of scientific convergence. The original production
counts remain in each notebook and are selected when
`OPENMICHROM_TUTORIAL_MODE=full`.

| Canonical notebook | Purpose and data | Reduced CPU status / observed runtime | Full-mode external requirements and limitations | Main corrections |
|---|---|---|---|---|
| `Converters/Tutorial_NDB_Converters.ipynb` | All eight NDB converter routes; deterministic two-frame/four-bead fixtures | Pass, about 4 s | No network, GPU, or external data | Public library/static API, semantic round-trip assertions, legacy `UN` to current `NA`, SPW/SWB distinction, and documented lossy fields |
| `Structural_Variations/Tutorial_Apply_Structural_Variants.ipynb` | Deletion, inversion, and tandem duplication of a deterministic symmetric locus matrix and paired directional motif tracks | Pass, about 5 s | No network, GPU, or external data; synthetic matrices test bookkeeping rather than biological prediction | Public `StructuralVariants` API, explicit zero-based half-open intervals, index-map/motif/symmetry assertions, ideal-background and duplication-contact assumptions, coordinated matrix/sequence round trip in a temporary directory |
| `Structural_Variations/Tutorial_Loop_Extrusion.ipynb` | Synthetic inversion, seeded two-sided extrusion, and dynamic harmonic loop bonds in a short CPU simulation | Pass, about 5 s | No network or external data; full mode uses 40 extrusion transitions and 20 MD steps per frame but remains a demonstration, not a converged production ensemble | Public `LoopExtrusionManager` and MiChroM dynamic-loop APIs, `num_steps + 1` frame check, collision and finiteness assertions, and in-context union-bond activation without coordinate serialization or Context reconstruction |
| `Tutorials/stream_remote_cndb.ipynb` | Local/remote CNDB selection, embedded/sidecar index, exact byte ranges, cache stats | Pass, about 5 s | Real 139 GB ENCODE CNDB; network and range-capable server; reads only index/selected ranges | Deterministic local HTTP fixture in fast mode, automatic `load(URL)`, format/version and byte-count assertions, context cleanup |
| `Active_Chromosome_Dynamics/Active_Polymer_Tutorial.ipynb` | Active Brownian chromosome dynamics; bundled sequence | Pass, about 5 s | 30,000-step collapse plus 500×1,000 production steps; CPU or optional CUDA | Current integrator/platform calls, deterministic seed, reduced counts, corrected time axis and output checks |
| `Chromosome_Pulling_Tutorial/Tutorial_Pulling.ipynb` | Constant-force and constant-distance chromosome pulling; bundled DT40 inputs/lambdas | Pass, about 5 s | Thousands of blocks and multiple distance windows; CPU or CUDA with required centroid-force support | Removed obsolete duplicate notebook, fixed force/context order and parameter updates, current filenames, robust PDB coordinate parsing, fast reporters/assertions |
| `Chromosome_simulations/Tutorial_MiChroM_Simulation.ipynb` | Classic chromosome 10 simulation; bundled ENCODE-derived BED | Pass, about 4 s | 1,000,000 production steps; no required network because input is bundled | Current `buildClassicMichrom`, CPU default, portable data path, reduced steps and assertions |
| `Chromosome_simulations/Tutorial_Multiple_Chromosomes.ipynb` | Collapse and combine chromosomes 10/11; bundled type files | Pass, about 5 s | Three 1,000,000-step phases | Correct `fileName` keyword, current force/setup calls, reduced counts and shape/output checks |
| `Chromosome_simulations/Tutorial_Single_Chromosome.ipynb` | Single-chromosome collapse, production, CNDB analyses; bundled sequence | Pass, about 6 s | Two 2,000-block simulations | Reporter cadence produces fast frames, numeric frame IDs avoid metadata groups, current CNDB analysis API and sanity checks |
| `Full_Inversion_Optimization/Tutorial_Full_Inversion_Optimization.ipynb` | Adam/full inversion from bundled GM12878 dense map | Pass, about 13 s | Original workflow can invoke bundled Juicer JAR/network and 1,000/5,000 simulation blocks | Fast mode is offline, `getHiCexp` replaces deprecated `getPars`, modern state names, finite output assertions, valid image markup |
| `MiChroM_Optimization/Tutorial_MiChroM_Optimization.ipynb` | Type and ideal-chromosome custom inversion; bundled dense/eigen/lambda inputs | Pass, about 11 s | 100–5,000 blocks per phase; convergence requires many replicas/frames | Removed developer-specific Juicer path, supported snake_case calls, modern simulation lifecycle, documented fast-only non-finite Hessian handling |

The maintained structural-variation workflow is traceable to Miles Gantcher's
[OpenMiChroM pull request 123](https://github.com/junioreif/OpenMiChroM/pull/123).
Its scientific context and assumptions are linked from the notebooks and the
[structural-variants reference](../source/Reference/structural_variants.rst),
including the associated [ACS poster](https://acs.digitellinc.com/p/s/maximum-entropy-polymer-model-predicts-the-architectural-effects-of-genetic-structural-variations-poster-board-207-650669)
and the Sanborn, Bianco, and Andrey studies. The distributed notebook fixtures
are synthetic, so none of those cited biological datasets is silently treated
as bundled tutorial input.

The docs tree previously contained a second, older pulling notebook. It was
removed because the newer canonical `Tutorial_Pulling.ipynb` supersedes it; Git
history retains the old content. No important cell is silently skipped in
reduced validation. The only changed scientific interpretation is explicitly
limited to fast mode: the custom-optimization fit replaces non-finite estimates
from its intentionally inadequate two-frame Hessian with zeros so plotting code
can be exercised. Full mode retains the original estimates unchanged.

## Python example scripts

The four optimization driver scripts under
`Tutorials/MiChroM_Optimization/scripts/` now use `timeStep`, `createSimulation`,
`run`, `prob_calculation_*`, `get_lambdas_*`, and current state attributes. The
undeclared `hdf5plugin` import was removed. These HPC-oriented scripts require
site-specific schedulers/input layouts and were statically compiled rather than
submitted to a cluster.
