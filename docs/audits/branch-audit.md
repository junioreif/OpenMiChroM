# Branch audit

Audit snapshot: 2026-08-12. The repository was fetched from
`https://github.com/junioreif/OpenMiChroM.git`; `origin/HEAD` resolves to
`origin/main`. The integration base is `170e33f` (tag `v1.1.1b`). All remote
heads and tags were fetched without pruning, rewriting, or deleting refs.

| Branch | Last commit (date, author) | Merged to `origin/main` | Unique commits / affected areas | Relevant missing work | Recommendation |
|---|---|---:|---|---|---|
| `main`, `origin/main` | `170e33f`, 2025-03-26, Antonio Bento de Oliveira Junior | Yes | 0; baseline package, tutorials, docs, and legacy tests | Does not contain CNDB streaming | No action; integration base |
| `origin/feature/cndbtools-stream-backend` | `e38b30a`, 2026-05-12, Vinícius Contessoto | No | 5 commits; `CndbTools`, `_cndb_stream`, vendored `hdf5-indexed-reader`/`pyfive`, manifests, remote tutorial, docs, sync script, tests | Complete relevant streaming implementation is absent from main | Integrate; all five commits were cherry-picked to preserve authorship, then hardened locally |
| `origin/copilot/add-novo-recurso` | `93288f2`, 2026-05-27, `copilot-swe-agent[bot]` | No | 2 commits (`3e93f2f`, `93288f2`); only `README.rst` and `setup.py`, adding Portuguese-language metadata/planning text | No CNDB, tutorial execution, testing, format, or deprecation work | Retain for separate review; no action in this branch |
| `codex/cndbtools-integration-tutorial-audit` | Created from `170e33f`; initial integrated tip `879f616`, Vinícius Contessoto | No | The five streaming commits plus the audit, compatibility, tutorial, documentation, packaging, and validation work recorded in this report | This is the prepared integration branch | Review locally; do not push or merge without maintainer approval |

## Unique streaming commits

| Upstream commit | Local cherry-pick | Subject |
|---|---|---|
| `12615db` | `05c5252` | Add cndb-stream backend to CNDBTools |
| `65c99f1` | `132d9ca` | Polish remote CNDB streaming tutorial |
| `72e8d48` | `418ccb5` | Synchronize tutorial notebooks with docs |
| `fd4d339` | `a3fd544` | Vendor CNDB streaming backend into CNDBTools |
| `e38b30a` | `879f616` | Harden CNDB streaming package checks |

The feature branch changes 38 files. Its useful scope is coherent: package
implementation and vendored license, package manifest, local/remote tests,
tutorial synchronization, remote notebook, and the corresponding README/Sphinx
documentation. No partial merge from the Copilot branch was warranted.

The original workspace path contained an unborn empty Git repository. It was
left untouched. Work was performed in the isolated worktree
`OpenMiChroM-cndbtools-integration`.

## Follow-up converter history audit

The converter follow-up also fetched and inspected pull-request heads and the
separate `mellofariam/NDB-Converters` repository. Merged OpenMiChroM pull
requests `#48`, `#49`, `#103`, `#106`, and `#116` contain related coordinate
load/save, documentation, SWB reporter, multi-chain export, and ring-handling
work already represented in `main`; none integrates the external converter
suite. Open pull request `#117` adds only a partial terminal-`END` repair to the
legacy NDB → CNDB helper and is superseded by the tested converter layer on this
integration branch. The external repository has only `master` at `8bd87e5` and
no alternate branch containing a packaged or tested implementation. Full
details and release/licensing notes are in `converter-audit.md`.
