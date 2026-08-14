#!/usr/bin/env python3
"""Execute tutorial notebooks in fast or complete validation mode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NotebookRun:
    path: Path
    cwd: Path
    fast: bool
    complete: bool
    network: bool = False
    copy_workdir: bool = True


RUNS = [
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Converters" / "Tutorial_NDB_Converters.ipynb",
        REPO_ROOT / "Tutorials" / "Converters",
        fast=True,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT
        / "Tutorials"
        / "Structural_Variations"
        / "Tutorial_Apply_Structural_Variants.ipynb",
        REPO_ROOT / "Tutorials" / "Structural_Variations",
        fast=True,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT
        / "Tutorials"
        / "Structural_Variations"
        / "Tutorial_Loop_Extrusion.ipynb",
        REPO_ROOT / "Tutorials" / "Structural_Variations",
        fast=True,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "stream_remote_cndb.ipynb",
        REPO_ROOT / "Tutorials",
        fast=True,
        complete=True,
        network=True,
        copy_workdir=False,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Active_Chromosome_Dynamics" / "Active_Polymer_Tutorial.ipynb",
        REPO_ROOT / "Tutorials" / "Active_Chromosome_Dynamics",
        fast=True,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Chromosome_Pulling_Tutorial" / "Tutorial_Pulling.ipynb",
        REPO_ROOT / "Tutorials" / "Chromosome_Pulling_Tutorial",
        fast=False,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Chromosome_simulations" / "Tutorial_MiChroM_Simulation.ipynb",
        REPO_ROOT / "Tutorials" / "Chromosome_simulations",
        fast=False,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Chromosome_simulations" / "Tutorial_Multiple_Chromosomes.ipynb",
        REPO_ROOT / "Tutorials" / "Chromosome_simulations",
        fast=False,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Chromosome_simulations" / "Tutorial_Single_Chromosome.ipynb",
        REPO_ROOT / "Tutorials" / "Chromosome_simulations",
        fast=False,
        complete=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "Full_Inversion_Optimization" / "Tutorial_Full_Inversion_Optimization.ipynb",
        REPO_ROOT / "Tutorials" / "Full_Inversion_Optimization",
        fast=False,
        complete=True,
        network=True,
    ),
    NotebookRun(
        REPO_ROOT / "Tutorials" / "MiChroM_Optimization" / "Tutorial_MiChroM_Optimization.ipynb",
        REPO_ROOT / "Tutorials" / "MiChroM_Optimization",
        fast=False,
        complete=True,
    ),
]


def prepare_notebook(run: NotebookRun, *, fast: bool, output_path: Path) -> None:
    notebook = json.loads(run.path.read_text(encoding="utf-8"))
    mode = "fast" if fast else "full"
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["openmichrom-validation-setup"]},
        "outputs": [],
        "source": [
            "import os\n",
            f"os.environ['OPENMICHROM_TUTORIAL_MODE'] = {mode!r}\n",
            "TUTORIAL_FAST = os.environ['OPENMICHROM_TUTORIAL_MODE'] == 'fast'\n",
            "TUTORIAL_PLATFORM = os.environ.get('OPENMICHROM_TUTORIAL_PLATFORM', 'CPU')\n",
            "print(f'OpenMiChroM tutorial validation: fast={TUTORIAL_FAST}, platform={TUTORIAL_PLATFORM}')\n",
        ],
    }
    notebook["cells"].insert(0, setup)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast", "complete"), default="fast")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--full-science",
        action="store_true",
        help="Run production step counts instead of the reduced validation mode.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="Execute only notebook paths containing this substring (repeatable).",
    )
    args = parser.parse_args()
    selected = [run for run in RUNS if run.fast] if args.mode == "fast" else list(RUNS)
    if args.only:
        selected = [
            run
            for run in selected
            if any(value in str(run.path.relative_to(REPO_ROOT)) for value in args.only)
        ]
        if not selected:
            parser.error("--only did not match any notebook selected by this mode")
    failures = 0
    with tempfile.TemporaryDirectory(prefix="openmichrom-notebooks-") as temporary:
        temp_dir = Path(temporary)
        # A user-level ``python3`` kernelspec often points at a different Conda
        # environment.  Register an ephemeral kernel so notebook execution uses
        # exactly the interpreter that launched this validation script.
        jupyter_root = temp_dir / "jupyter"
        kernel_dir = jupyter_root / "kernels" / "openmichrom-validation"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "OpenMiChroM validation",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        for index, run in enumerate(selected):
            relative = run.path.relative_to(REPO_ROOT)
            if args.full_science and run.network and not args.allow_network:
                print(f"SKIP {relative}: requires --allow-network")
                continue
            work_dir = temp_dir / f"work-{index}"
            if run.copy_workdir:
                shutil.copytree(run.cwd, work_dir)
            else:
                work_dir.mkdir()
            prepared = work_dir / run.path.name
            prepare_notebook(run, fast=not args.full_science, output_path=prepared)
            started = time.monotonic()
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.kernel_name=openmichrom-validation",
                    f"--ExecutePreprocessor.timeout={args.timeout}",
                    "--output",
                    str(work_dir / f"executed-{run.path.stem}.ipynb"),
                    str(prepared),
                ],
                cwd=work_dir,
                env={
                    **os.environ,
                    # Always validate the checkout that owns this harness,
                    # even when another OpenMiChroM version is installed in
                    # the active environment.
                    "PYTHONPATH": os.pathsep.join(
                        value
                        for value in (
                            str(REPO_ROOT),
                            os.environ.get("PYTHONPATH"),
                        )
                        if value
                    ),
                    "JUPYTER_PATH": os.pathsep.join(
                        value
                        for value in (str(jupyter_root), os.environ.get("JUPYTER_PATH"))
                        if value
                    ),
                    "OPENMICHROM_TUTORIAL_MODE": "full" if args.full_science else "fast",
                },
                text=True,
                capture_output=True,
            )
            elapsed = time.monotonic() - started
            if process.returncode:
                failures += 1
                print(f"FAIL {relative} ({elapsed:.1f}s)")
                print(process.stderr[-4000:])
            else:
                print(f"PASS {relative} ({elapsed:.1f}s)")
    print(f"Notebook execution: {len(selected)} selected, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
