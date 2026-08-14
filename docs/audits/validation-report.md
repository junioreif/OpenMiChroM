# Local validation report

Validation host/date: Apple M3 Max (40-core Apple GPU), macOS, 2026-08-14.
Worktree: the isolated `OpenMiChroM-cndbtools-integration` checkout.

## Isolated environment

Environment name: `openmichrom-cndbtools-integration-py310`.
Reproducible specification: `environment.yml`. The environment was created with
the classic Conda solver because the base installation's optional libmamba
plugin reports an incompatible `libmambapy.QueryFormat`; this warning is outside
the repository and did not affect the isolated environment.

Equivalent setup commands are:

```bash
CONDA_NO_PLUGINS=true conda env create --solver classic -f environment.yml
conda activate openmichrom-cndbtools-integration-py310
python -m pip install -e .
```

| Dependency | Validated version |
|---|---:|
| Python | 3.10.20 |
| OpenMM | 8.5.2 |
| NumPy | 2.2.6 |
| SciPy | 1.15.2 |
| scikit-learn | 1.7.2 |
| h5py / HDF5 | 3.16.0 / 2.1.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| pytest | 9.1.1 |
| Jupyter / nbconvert | 1.0.0 / 7.17.1 |
| Sphinx / nbsphinx / RTD theme | 8.1.3 / 0.9.8 / 3.1.0 |
| build / pip / setuptools / wheel | 1.5.0 / 26.2.1 / 84.0.0 / 0.47.0 |

OpenMM exposes Reference, CPU, and OpenCL on this host. All recorded simulation
and notebook checks used CPU. The machine has no NVIDIA/CUDA platform, so the
optional `--gpu` check was correctly not run; the Apple GPU is not a compatible
CUDA target.

## Final offline results

| Check | Result |
|---|---|
| Public/static Python compilation | Pass |
| Full pytest | 203 passed, 1 optional live-network test skipped |
| Focused deterministic CNDB/converter/SV/extrusion/API/CPU suite | 203 passed |
| Notebook static audit and stable cell IDs | 11/11 pass, 0 issues |
| Canonical-to-Sphinx notebook sync | 11/11 pass |
| Fast executable notebooks | 5/5 pass |
| Complete reduced executable notebooks | 11/11 pass; every cell executed |
| Sphinx HTML with `-W --keep-going` | Pass, zero warnings |
| Source distribution and wheel | Pass |
| Wheel contents | Required force-field, typing marker, and vendored license present; repository tests absent |
| Isolated `--no-deps` wheel install, public imports, and converter CLI | Pass, version 1.1.1 |

Complete reduced notebook runtimes from the final recorded run were 13.0 s
(converters), 5.4 s (structural-variant transforms), 4.4 s (loop extrusion),
4.6 s (remote stream), 4.3 s (active polymer), 4.7 s (pulling), 3.7 s (classic
MiChroM), 5.0 s (multiple chromosomes), 5.4 s (single chromosome), 12.2 s
(full inversion), and 11.4 s (custom optimization). The two new notebooks also
passed their explicit full demonstration modes in 7.2 s and 5.2 s,
respectively.

The converter suite exercises all eight directed routes, every standard
MiChroM type and historical GRO atom code, numeric/byte/string CNDB types,
terminal `END` and `ENDMDL`, frame 0 and noncontiguous frames, loop sidecars,
the current reporter layout, the actual CPU simulation NDB writer, ambiguity
warnings/type overrides, and atomic failure/overwrite behavior. The converter
tutorial completed in both fast and complete modes without network, simulation
steps, or GPU use.

The structural-variant and loop-extrusion suite covers exact deletion,
inversion, and duplication index maps; directional motif transforms;
ideal-chromosome handling; finite-only locus-matrix/sequence output; seeded
trajectories; dense reload policies; collision/crossing invariants; and live
OpenMM parameter and energy updates in one persistent Context. The tutorials
use deterministic synthetic arrays rather than redistributing the original
branch's opaque biological fixtures.

The default tests and complete reduced notebook run are offline. Remote behavior
is validated by a temporary `ThreadingHTTPServer`, including redirects, ignored
ranges, truncated responses, 404s, exact byte counts, and cleanup. The separate
live ENCODE test remains opt-in with `scripts/validate.py fast --network`.
It passed on 2026-08-12 in 49.24 s, reading a ten-bead frame slice from the
139 GB object via byte ranges. The source file has no authoritative format
version, so the expected `LegacyCNDBVersionWarning` was emitted.

Full production simulations were not run because they intentionally retain up
to millions of integration steps, thousands of blocks, many replicas, and (for
the ENCODE/Juicer cases) external network access. They are not silently skipped:
the exact path is `python scripts/validate.py complete --full-science --network`.
Reduced mode tests execution and invariants, not scientific convergence.
