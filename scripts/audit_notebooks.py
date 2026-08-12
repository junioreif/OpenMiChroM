#!/usr/bin/env python3
"""Statically validate tutorial notebooks and embedded Python code."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEPRECATED_SNIPPETS = {
    "sys.path.append(": "install OpenMiChroM in the environment instead of mutating sys.path",
    "simtk.openmm": "import openmm",
    "runSimBlock(": "use MiChroM.run",
    "initStorage(": "use MiChroM.createReporters",
    ".chromRG(": "compute radius of gyration from positions or cndbTools.compute_RG",
    "time_step=": "use timeStep=",
    "kfb=": "use kFb=",
    "ka=": "use kA=",
    "Ecut=": "use eCut=",
    "kr=": "use kR=",
    "n_rad=": "use nRad=",
    ".probCalculation_": "use prob_calculation_*",
    ".getLamb_types(": "use get_lambdas_types",
    ".calc_sim_phi_types(": "use calc_phi_sim_types",
    ".calc_exp_phi_types(": "use calc_phi_exp_types",
    ".getHiCSim(": "use get_HiC_sim",
}


def notebook_paths() -> list[Path]:
    return sorted((REPO_ROOT / "Tutorials").rglob("*.ipynb"))


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def python_source(source: str) -> str | None:
    stripped = source.lstrip()
    if stripped.startswith("%%bash") or stripped.startswith("%%script"):
        return None
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


def audit(path: Path) -> list[str]:
    errors = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"cannot parse notebook JSON: {exc}"]
    if notebook.get("nbformat") != 4:
        errors.append(f"unsupported nbformat {notebook.get('nbformat')!r}; expected 4")
    cell_ids = [cell.get("id") for cell in notebook.get("cells", [])]
    if any(cell_id is None for cell_id in cell_ids):
        errors.append("one or more cells are missing stable nbformat cell IDs")
    present_ids = [cell_id for cell_id in cell_ids if cell_id is not None]
    if len(present_ids) != len(set(present_ids)):
        errors.append("duplicate nbformat cell IDs")
    for index, cell in enumerate(notebook.get("cells", [])):
        source = source_text(cell)
        if cell.get("cell_type") == "code":
            for snippet, replacement in DEPRECATED_SNIPPETS.items():
                if snippet in source:
                    errors.append(f"cell {index}: {snippet!r} is obsolete; {replacement}")
            code = python_source(source)
            if code is not None:
                try:
                    ast.parse(code or "pass", filename=f"{path}:cell-{index}")
                except SyntaxError as exc:
                    errors.append(f"cell {index}: Python syntax error: {exc.msg} (line {exc.lineno})")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"cell {index}: stored execution error {output.get('ename')}: "
                        f"{output.get('evalue')}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    failures = 0
    for path in notebook_paths():
        errors = audit(path)
        relative = path.relative_to(REPO_ROOT)
        if errors:
            failures += len(errors)
            print(f"FAIL {relative}")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"PASS {relative}")
    print(f"Notebook audit: {len(notebook_paths())} checked, {failures} issue(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
