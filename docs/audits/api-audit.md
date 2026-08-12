# Public API, duplication, and deprecation audit

The audit covered every first-party Python module, package exports, tests,
tutorials, example scripts, Sphinx autodoc, packaging entry points, static AST
duplicates, and relevant history.

| API or duplication | Evidence | Action in this branch | Maintainer decision still needed |
|---|---|---|---|
| `AdamTraining.getPars` | Deprecated in `22a19f1` on 2023-06-27 for release 1.0.7; warning says to use `getHiCexp`. No removal release was announced. | Kept. Tutorials/examples use `getHiCexp`; an exact numerical characterization test proves the retained wrapper produces identical matrices and state. | Decide a removal release and downstream notice before deletion; likely major-version work. |
| `getPars(cutoff=...)` | The same 1.0.7 commit deprecated it in favor of `cutoff_low`/`cutoff_high`; no removal deadline exists and the method remains a compatibility path. | Kept; no first-party call site passes `cutoff`. | Decide alongside `getPars` removal. |
| Duplicate `AdamTraining.reset_Pi` definitions | `origin/main` contained identical definitions at original lines 281 and 361; the latter silently shadowed the former. | Removed the earlier duplicate without changing the surviving body. AST regression test rejects duplicate method names in first-party classes. | None. |
| Duplicate normalization algorithm in `AdamTraining` and `CustomMiChroMTraining` | The two `normalize_matrix` bodies were byte-for-byte equivalent. | Moved the algorithm to private `_normalize_matrix`; both public methods remain wrappers. A numerical characterization covers NaN, infinity, symmetry, and both classes. | None. |
| `AdamTraining.getHiCexp` vs `CustomMiChroMTraining.get_HiC_exp` | Large near-duplicate algorithms, but their default normalization differs (`True` vs `False`), their class state differs, and scientific callers depend on those defaults. | Retained independently; no opportunistic numerical refactor. | Consider a shared, explicitly parameterized implementation only with broader scientific regression fixtures. |
| Custom-training camelCase methods (`probCalculation_*`, `getLamb_types`, `getHiCSim`, etc.) | Renamed without compatibility wrappers by `a82999e` on 2023-05-08. Current package exposes snake_case methods; stale tutorials and scripts still called old names. | Standardized every current tutorial and example script on the supported snake_case API. No historical aliases were invented after the fact. Static notebook audit rejects the stale spellings. | Decide whether downstream users warrant temporary aliases in a future compatibility release. |
| `cndbTools` / `CndbTools` capitalization | Historical public class is `cndbTools`; the streaming branch introduced the PEP-8 spelling. | Both names are the same class and are exported from `OpenMiChroM`; compatibility test locks this behavior. | None. |
| `scipy.stats.stats.pearsonr` | The private/deprecated module path generated modern SciPy warnings. | Replaced with public `scipy.stats.pearsonr`; function object and numerical behavior are unchanged. | None. |
| `CLINAME=OpenMiChroM._cli:main` console entry point | Added in 2021; `_cli.py` has never existed anywhere in repository history, so every invocation fails at import. The name is a packaging placeholder rather than a usable API. | Removed from `setup.py` and `meta.yaml`. Wheel installation and public imports are tested. | If a real CLI is desired, specify its commands and add it as a new tested feature. |
| `tests/tests.py` | Not collected by current pytest conventions, ran 20,000-step OpenCL jobs at import, referenced missing fixture directories, and called removed method names. | Replaced with deterministic `test_simulation_smoke.py`, covering CPU setup, integration, CNDB reporting/loading, cleanup, and NDB/PDB/GRO/XYZ exports. | None. |
| Python support metadata | Vendored `pyfive` uses structural pattern matching, so the integrated package requires Python 3.10 even though old docs claimed 3.6. | Added `python_requires >=3.10`, classifiers, Conda constraints, Read the Docs 3.10, and consistent install docs. | Treat the support-floor change as release-note material. |

The static duplicate check intentionally excludes vendored code and same-named
methods in different classes. Same names such as `probCalc` in Adam and full
training represent different state models, not accidental duplicates.
