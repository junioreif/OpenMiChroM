# Structural File I/O Audit

This note records the pre-implementation audit for the CNDBTools structural-file
I/O unification work. It is intentionally written before code changes beyond
branch setup, so the implementation can proceed from observed repository
behavior instead of assumptions.

## OpenMiChroM CNDBTools

`OpenMiChroM/CndbTools.py` exposes both `cndbTools` and the alias
`CndbTools = cndbTools`.

Current public behavior includes:

- `cndbTools().load(fileName)` for local `.cndb` and nominal `.ndb` inputs.
- `CndbTools.from_remote(h5_url, trajectory=None, index_cache_path=None, **kwargs)`
  for remote embedded-index CNDB/HDF5 files through the internal
  `OpenMiChroM._cndb_stream` backend.
- `xyz(frames=None, beadSelection=None, XYZ=[0, 1, 2])` for coordinate extraction.
- `stream_stats()` and stream byte-count properties in remote mode.
- Analysis helpers such as `compute_RG`, `compute_RDP`, `compute_MSD`,
  `traj2HiC`, and related methods that operate on arrays returned by `xyz`.

Local `load(...)` behavior:

- Opens files with `h5py.File(fileName, "r")`.
- Assumes a simple CNDB HDF5 layout with root-level `/types` plus numeric
  root-level frame datasets such as `/1`, `/2`, ...
- Builds `ChromSeq` from `/types`.
- Builds `dictChromSeq` by grouping bead indices by each type value.
- Sets `Nbeads = len(types)`.
- Sets `Nframes = len(self.cndb.keys()) - 1`, which assumes only one non-frame
  dataset, usually `/types`.
- Sets `frame_ids` from root keys that are digit strings.
- Does not currently understand nested `/<trajectory>/spatial_position/<frame>`
  files in local `load(...)`.
- `.ndb` handling refers to `Chrom_utils.ndb2cndb(f_name)`, but `Chrom_utils`
  is not imported in the module. There is also a local `ndb2cndb(...)` method.

Local `xyz(...)` behavior:

- Defaults to all beads and frames `range(1, self.Nframes + 1)`.
- Reads each requested frame through `np.array(self.cndb[str(i)])`, so each
  selected frame dataset is loaded before bead and coordinate-axis selection.
- Applies bead selection with `np.take(..., axis=0)` and axis selection with
  `np.take(..., axis=1)`.

Remote streaming behavior:

- `from_remote(...)` creates `_CNDBStreamBackend`, which imports
  `OpenMiChroM._cndb_stream.IndexedCNDB`.
- The backend uses embedded HDF5 object indexes and HTTP Range requests.
- `xyz(...)` dispatches to `_xyz_stream(...)` when `_stream_backend` is set.
- Contiguous selections are read as exact byte ranges.
- Non-contiguous selections currently read the smallest enclosing bead interval
  and then subset in memory.
- `dictChromSeq` is intentionally empty in remote mode today; remote metadata
  and type dictionaries are not yet populated.

## Current Writer/Reporter Behavior

The structural writer is primarily `OpenMiChroM/CustomReporter.py::SaveStructure`.
It is attached by `ChromDynamics.MiChroM.createReporters(..., traj=True,
trajFormat="cndb", ...)`.

Current `.cndb` reporter behavior:

- Creates one file per chain named `{filePrefix}_{k}.cndb`.
- Writes root dataset `/types` from `typeListLetter[chain[0]:chain[1]+1]`.
- Writes each reported frame as a root-level dataset named `str(self.step)`.
- Frame IDs start at `0`, not `1`, in `SaveStructure.report(...)`.
- Uses default h5py dataset creation without explicit compression or chunking.
  For fixed-size numeric arrays, h5py normally writes contiguous uncompressed
  datasets unless defaults are changed.
- Does not write `/Header`, `_index`, or `_index_offset`.

Current `.swb` reporter behavior:

- Creates one file per chain named `{filePrefix}_{k}.swb`.
- Writes `/Header` group with attributes such as `version`, `format`, `genome`,
  `pointtype`, `title`, `author`, and `date`.
- Creates `/{filePrefix}/spatial_position`.
- Writes `/{filePrefix}/genomic_position`.
- Writes each frame under `/{filePrefix}/spatial_position/t_{self.step}`.
- Does not write `/types`, `_index`, or `_index_offset`.

Single-structure exports:

- `MiChroM.saveStructure(...)` supports `xyz`, `pdb`, `gro`, and `ndb` text-like
  outputs for the current state.
- The method does not currently write indexed CNDB files.

## Packaging and Import Behavior

Packaging currently uses `setup.py` with `find_packages()`, so
`OpenMiChroM._cndb_stream` and its nested vendored packages are included because
each directory has an `__init__.py`.

`MANIFEST.in` includes:

- `OpenMiChroM/share/MiChroM.ff`
- `OpenMiChroM/_cndb_stream/py.typed`
- `OpenMiChroM/_cndb_stream/_vendor/README.md`
- `OpenMiChroM/_cndb_stream/_vendor/hdf5_indexed_reader/LICENSE`

No external `cndb-stream` dependency or `stream` extra remains.

Important import limitation:

- `OpenMiChroM/__init__.py` imports `ChromDynamics`, `Optimization`,
  `Integrators`, and `CustomReporter` at package import time.
- Those modules import OpenMM.
- Therefore `from OpenMiChroM.CndbTools import CndbTools` may require OpenMM
  even for analysis-only users, because Python executes package `__init__.py`
  before loading the submodule.

## Internal Streaming Backend

`OpenMiChroM/_cndb_stream` currently provides:

- strict HTTP Range reads with stdlib `urllib`
- embedded object-index loading and caching
- local/external JSON index support
- simple root-frame CNDB support
- nested NDB/SWB-style support for paths like
  `/<trajectory>/spatial_position/<frame>`
- exact byte reads for contiguous, uncompressed `(n_beads, 3)` coordinate
  datasets

The backend deliberately raises on HTTP `200 OK` responses to range requests to
avoid accidental full-file downloads.

## Public URL Probe Results

These probes used only `HEAD` and small `Range: bytes=0-1023` requests. The
NDB Rice server required an unverified local SSL context in this development
environment due local CA verification failure.

| URL | Probe result |
| --- | --- |
| Harris LCL chr7 `.cndb` | HDF5 signature present. `Content-Length` 22,547,755,840 bytes. Server returned `200 OK` to Range request and no `Content-Range`; unsafe for direct remote streaming from this endpoint. |
| Mello GM12878 500-frame `.cndb` | HDF5 signature present. `Content-Length` 30,089,264 bytes. Server returned `200 OK` to Range request and no `Content-Range`; unsafe for direct remote streaming from this endpoint. |
| Oliveira C1 multichain `.cndb` | HDF5 signature present. `Content-Length` 6,024,781,964 bytes. Server returned `200 OK` to Range request and no `Content-Range`; unsafe for direct remote streaming from this endpoint. |
| ENCODE `ENCFF764BAH.cndb` | Redirects to S3. HDF5 signature present. `Content-Length` 135,272,750,514 bytes. S3 returned `206 Partial Content`; suitable for Range-based inspection/streaming if embedded index is present. |
| NCBI FTP Woolly Mammoth Direct Inv `.sw` | HDF5 signature present. `Content-Length` 1,232,374,685 bytes. FTP HTTPS endpoint returned `206 Partial Content`; candidate for Range-based inspection/streaming. |
| NCBI GEO download Woolly Mammoth MiChroM `.sw` | HDF5 signature present. `Content-Length` 32,323,363,748 bytes. The `www.ncbi.nlm.nih.gov/geo/download` endpoint returned `200 OK` to Range request; unsafe from that URL unless redirected/direct file URL supports Range. |
| Bintu A549 `.ndb` | Text NDB signature: starts with `HEADER    NDB Fi`. `Content-Length` 25,313,749 bytes. Server returned `200 OK` to Range request; remote parser should not download by default. |

## NDB-Converters Audit

The repository `https://github.com/mellofariam/NDB-Converters` was cloned to
`/Users/vc18/Work/Dev/NDB-Converters` for inspection at commit `8bd87e5`.

Files present:

- `cndb2ndb.py`
- `csv2ndb.py`
- `gro2ndb.py`
- `ndb2cndb.py`
- `ndb2pdb.py`
- `ndb2spw.py`
- `pdb2ndb.py`
- `spw2ndb.py`
- `README.md`

No `LICENSE`, `COPYING`, or `NOTICE` file is present. Because there is no
explicit license in the repository, OpenMiChroM should not vendor or copy the
converter source directly without permission/license clarification. Converter
support should be implemented as clean OpenMiChroM code based on documented
file behavior and tests, or the license should be clarified before vendoring.

Observed converter format behavior:

- CNDB-to-NDB assumes root `/types` and root numeric frame datasets.
- NDB-to-CNDB writes root `/types` and root numeric frame datasets.
- NDB records use `MODEL`, `CHROM`, `TER`, `ENDMDL`, `LOOPS`, `MASTER`, and
  `END` style records.
- Text SpaceWalk `.spw` starts with `##format=sw1` and uses `trace`, header,
  and rows like `chromosome start end x y z`.

## Implementation Implications

Recommended safe sequence:

1. Add a small structural-file detection framework and safe inspection script.
2. Add `CndbTools.open(...)` as a routing layer without changing `load(...)`.
3. Make analysis-only `CndbTools` import possible without OpenMM.
4. Populate remote metadata dictionaries where embedded-index metadata exposes
   `/types`, `/genomic_position`, and related datasets.
5. Improve non-contiguous selection via range coalescing.
6. Add writer metadata and embedded-index writing.
7. Add clean-room converters after documenting format rules and licensing risk.
8. Treat chunked/compressed reading as a later, staged implementation because it
   needs HDF5 chunk/filter metadata and careful filter handling.

The first implementation slice should avoid changing current local CNDBTools
semantics and should preserve the working ENCODE embedded-index path.
