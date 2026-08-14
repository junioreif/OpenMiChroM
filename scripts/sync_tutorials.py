#!/usr/bin/env python3
"""Synchronize canonical tutorial notebooks into the Sphinx docs tree."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = REPO_ROOT / "Tutorials"
DOCS_DIR = REPO_ROOT / "docs" / "source" / "Tutorials"


@dataclass(frozen=True)
class TutorialNotebook:
    source: Path
    docs_name: str

    @property
    def source_path(self) -> Path:
        return CANONICAL_DIR / self.source

    @property
    def docs_path(self) -> Path:
        return DOCS_DIR / self.docs_name


TUTORIALS = [
    TutorialNotebook(
        Path("Converters/Tutorial_NDB_Converters.ipynb"),
        "Tutorial_NDB_Converters.ipynb",
    ),
    TutorialNotebook(
        Path("Structural_Variations/Tutorial_Apply_Structural_Variants.ipynb"),
        "Tutorial_Apply_Structural_Variants.ipynb",
    ),
    TutorialNotebook(
        Path("Structural_Variations/Tutorial_Loop_Extrusion.ipynb"),
        "Tutorial_Loop_Extrusion.ipynb",
    ),
    TutorialNotebook(
        Path("Chromosome_simulations/Tutorial_MiChroM_Simulation.ipynb"),
        "Tutorial_MiChroM_Simulation.ipynb",
    ),
    TutorialNotebook(
        Path("Chromosome_simulations/Tutorial_Single_Chromosome.ipynb"),
        "Tutorial_Single_Chromosome.ipynb",
    ),
    TutorialNotebook(
        Path("Chromosome_simulations/Tutorial_Multiple_Chromosomes.ipynb"),
        "Tutorial_Multiple_Chromosomes.ipynb",
    ),
    TutorialNotebook(
        Path("MiChroM_Optimization/Tutorial_MiChroM_Optimization.ipynb"),
        "Tutorial_MiChroM_Optimization.ipynb",
    ),
    TutorialNotebook(
        Path("Full_Inversion_Optimization/Tutorial_Full_Inversion_Optimization.ipynb"),
        "Tutorial_Full_Inversion_Optimization.ipynb",
    ),
    TutorialNotebook(
        Path("Chromosome_Pulling_Tutorial/Tutorial_Pulling.ipynb"),
        "Tutorial_Pulling.ipynb",
    ),
    TutorialNotebook(
        Path("Active_Chromosome_Dynamics/Active_Polymer_Tutorial.ipynb"),
        "Tutorial_Active_Polymer.ipynb",
    ),
    TutorialNotebook(
        Path("stream_remote_cndb.ipynb"),
        "stream_remote_cndb.ipynb",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy canonical notebooks from Tutorials/ into docs/source/Tutorials/ "
            "or check that the two locations are synchronized."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not copy files; fail if any mapped docs notebook is out of sync.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Clear execution counts and outputs while syncing or checking.",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=5_000_000,
        help="Fail if a source notebook contains more output JSON than this.",
    )
    return parser


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_notebook_bytes(path: Path, *, clear_outputs: bool) -> bytes:
    notebook = load_notebook(path)
    if clear_outputs:
        strip_outputs(notebook)
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def strip_outputs(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def output_json_size(path: Path) -> int:
    notebook = load_notebook(path)
    total = 0
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            total += len(json.dumps(output, ensure_ascii=False).encode("utf-8"))
    return total


def validate_mapping(max_output_bytes: int) -> int:
    errors = 0
    mapped_docs = {notebook.docs_path for notebook in TUTORIALS}
    for path in sorted(DOCS_DIR.glob("*.ipynb")):
        if path.name.startswith(".") or ".ipynb_checkpoints" in path.parts:
            continue
        if path not in mapped_docs:
            print(f"docs notebook is not managed by sync manifest: {path.relative_to(REPO_ROOT)}")
            errors += 1

    for notebook in TUTORIALS:
        if notebook.source_path.name.startswith(".") or ".ipynb_checkpoints" in notebook.source_path.parts:
            continue
        if not notebook.source_path.exists():
            print(f"missing source: {notebook.source_path.relative_to(REPO_ROOT)}")
            errors += 1
            continue
        output_size = output_json_size(notebook.source_path)
        if output_size > max_output_bytes:
            print(
                "source notebook has very large outputs: "
                f"{notebook.source_path.relative_to(REPO_ROOT)} ({output_size} bytes)"
            )
            errors += 1
    return errors


def check_notebooks(*, clear_outputs: bool) -> int:
    errors = 0
    for notebook in TUTORIALS:
        source = notebook.source_path
        target = notebook.docs_path
        if not source.exists():
            continue
        if not target.exists():
            print(f"out of sync: missing docs copy {target.relative_to(REPO_ROOT)}")
            errors += 1
            continue
        source_bytes = normalized_notebook_bytes(source, clear_outputs=clear_outputs)
        target_bytes = normalized_notebook_bytes(target, clear_outputs=clear_outputs)
        if source_bytes != target_bytes:
            print(f"out of sync: {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
            errors += 1
        else:
            print(f"ok: {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
    return errors


def sync_notebooks(*, clear_outputs: bool) -> int:
    copied = 0
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for notebook in TUTORIALS:
        source = notebook.source_path
        target = notebook.docs_path
        if not source.exists():
            print(f"missing source: {source.relative_to(REPO_ROOT)}")
            return 1
        if clear_outputs:
            data = normalized_notebook_bytes(source, clear_outputs=True)
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
                copied += 1
                print(f"copied: {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
            else:
                print(f"unchanged: {target.relative_to(REPO_ROOT)}")
        else:
            if not target.exists() or source.read_bytes() != target.read_bytes():
                shutil.copy2(source, target)
                copied += 1
                print(f"copied: {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
            else:
                print(f"unchanged: {target.relative_to(REPO_ROOT)}")
    print(f"tutorial sync complete: {copied} file(s) copied")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    errors = validate_mapping(args.max_output_bytes)
    if errors:
        return 1
    if args.check:
        return 1 if check_notebooks(clear_outputs=args.clear_outputs) else 0
    return sync_notebooks(clear_outputs=args.clear_outputs)


if __name__ == "__main__":
    sys.exit(main())
