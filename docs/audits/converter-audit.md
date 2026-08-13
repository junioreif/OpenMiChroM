# NDB converter integration audit

Audit snapshot: 2026-08-13. The behavioral reference was
[`mellofariam/NDB-Converters`](https://github.com/mellofariam/NDB-Converters),
branch `master`, commit `8bd87e56e99c905e977455a01ba4b7552ca074f7`.
That repository has one branch, no tags, eight standalone conversion scripts,
no automated tests or notebook, and no package or reusable import API.

## Provenance and release consideration

The historical scripts credit Vinícius G. Contessoto and Matheus F. Mello; the
CNDB pair also credits Antonio B. Oliveira Junior. Most Git history is authored
by Matheus, with a 2024 numeric-CNDB compatibility change by Antonio/Junior.

No `LICENSE`, `COPYING`, package license field, or source license statement was
present at the audited commit. Authorship is not a software license. No source
from that repository was copied into OpenMiChroM. `OpenMiChroM/Converters.py`
is a new implementation of the documented file-format behavior, with explicit
acknowledgement in the module, reference page, and tutorial.

## Route inventory and integration result

| Historical script | Route | Previous OpenMiChroM state | Integration result |
|---|---|---|---|
| `ndb2cndb.py` | NDB → CNDB | One incompatible `CndbTools.ndb2cndb` helper | Replaced by validated, atomic `ndb_to_cndb`; old name retained as a wrapper |
| `cndb2ndb.py` | CNDB → NDB | Absent | `cndb_to_ndb` |
| `ndb2pdb.py` | NDB → PDB | Snapshot writer only | `ndb_to_pdb` with type-preserving hints and loop sidecar |
| `pdb2ndb.py` | PDB → NDB | Simulation loader only | `pdb_to_ndb`, accepting modern residues and legacy atom labels |
| `ndb2spw.py` | NDB → SpaceWalk text | Absent | `ndb_to_spw` with loop sidecar |
| `spw2ndb.py` | SpaceWalk text → NDB | Absent | `spw_to_ndb` |
| `gro2ndb.py` | GRO → NDB | Simulation loader only | `gro_to_ndb`, including concatenated frames and legacy labels |
| `csv2ndb.py` | Bintu-layout CSV → NDB | Absent | `csv_to_ndb`, explicitly scoped to `model,index,z,x,y` |

All routes are available as top-level functions, through the generic
`convert(...)` dispatcher, and as static methods on both the preferred
`CndbTools` name and historical `cndbTools` alias. The optional command-line
entry point is `openmichrom-convert`.

Text `.spw` remains distinct from the existing binary HDF5 `.swb` reporter.
Renaming one suffix to the other is not a conversion.

## Historical defects characterized before replacement

- The old NDB → CNDB parser finalized only `ENDMDL`, while current
  OpenMiChroM snapshot/reporter NDB files can finish a model with `END`.
- It accepted legacy `UN` but not OpenMiChroM's current `NA` label.
- Its fixed coordinate slices do not match the current `SaveStructure` reporter
  layout and can silently lose a minus sign or coordinate digit.
- A failed conversion published an empty partial `.cndb` file.
- CNDB → NDB inferred frame count from the number of HDF5 objects, assumed
  contiguous one-based keys, and therefore skipped current frame `0` while
  eventually requesting a missing frame. It also failed on byte-string types.
- Independently sorting loop columns could change loop pairs.
- The historical PDB pair was not a round trip: its writer emitted `CA` while
  its reader expected legacy atom labels. B3 and B4 also shared a residue.
- Every script parsed command-line arguments during import and overwrote its
  destination before complete validation.

The new common trajectory representation validates frame IDs, shapes, finite
coordinates, bead types, chain IDs/indices, genomic intervals, sigma, and loop
pairs before output is published. CNDB output records authoritative format and
version metadata and reserved NDB metadata datasets so NDB ↔ CNDB can preserve
chain and genomic fields.

## Compatibility and intentional information loss

- `NA`, legacy `UN`, numeric CNDB codes 0–6, byte strings, and Unicode strings
  are accepted. New output uses `NA`.
- Numeric CNDB frame keys are discovered and sorted directly, including `0`
  and noncontiguous IDs.
- NDB models end correctly on `ENDMDL`, `END`, or a validated EOF without
  double-finalizing the terminal model.
- Legacy PDB/GRO labels `ZA/OA/FB/SB/TB/LB/UN` are accepted. OpenMiChroM's
  current residue representation is also accepted. CA/HIS and CA/ARG inputs
  are ambiguous across historical residue maps; the converter warns and
  accepts an explicit `types=` sequence to restore fidelity.
- PDB rounds coordinates to three decimals and does not natively store genomic
  intervals. The OpenMiChroM writer adds reversible type and chain hints.
- PDB and SPW exports atomically replace their companion `.loops` file even
  when it is empty, preventing a stale loop list from a reused output stem.
- SPW preserves frames, chromosomes, genomic intervals, and coordinates, but
  not chromatin types or sigma. SPW imports default to type `NA`; NDB assembly
  metadata is carried into the SpaceWalk `genome` header unless overridden.
- GRO does not carry genomic intervals; they are reconstructed from explicit
  resolution/start options. Coordinate scaling is explicit and defaults to 1.
- The CSV reader is the historical Bintu layout only, not an arbitrary schema
  detector. Chromosome metadata is required rather than silently fabricated.

## OpenMiChroM branch and pull-request check

The default branch and every fetched remote branch were searched for converter
names and format routes. No branch contains the external eight-route suite.
Merged historical pull requests provide related loaders/writers, documentation,
SWB reporting, multi-chain snapshot handling, and other compatibility work, but
not a maintained converter API.

Open pull request `#117` (`ronaldojunio:ndb2cndb`, audited head `17f2784`) adds
support for terminal `END` to the old helper. It remains incomplete: it does not
accept `NA`, retains debugging output and legacy parsing assumptions, and adds
neither tests nor a tutorial. Its useful intent is superseded by this validated
implementation; it should not be merged independently without reconciliation.

## Automated verification

`tests/test_converters.py` covers all eight routes, both NDB terminators,
current reporter output, legacy/current unknown labels, numeric/string CNDB
types, frame `0` and noncontiguous IDs, loops/sidecars, format inference,
PathLike/Unicode paths, overwrite refusal, and failure cleanup. The actual CPU
simulation NDB snapshot is auto-converted and reloaded in
`tests/test_simulation_smoke.py`.

`Tutorials/Converters/Tutorial_NDB_Converters.ipynb` executes every route
offline with deterministic fixtures and assertions, including coordinate,
frame, type, genomic-interval, and loop checks.
