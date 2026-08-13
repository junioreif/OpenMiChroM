#!/usr/bin/env python3
"""Run repeatable offline OpenMiChroM validation from the repository root."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUSED_TESTS = [
    "tests/test_api_compatibility.py",
    "tests/test_converters.py",
    "tests/test_cndbtools_io_contract.py",
    "tests/test_cndbtools_stream.py",
    "tests/test_simulation_smoke.py",
]


def run_stage(name: str, command: list[str], *, cwd: Path = REPO_ROOT, env=None) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    process = subprocess.run(command, cwd=cwd, env=env)
    if process.returncode:
        raise SystemExit(f"FAILED: {name} (exit {process.returncode})")
    print(f"PASS: {name}", flush=True)


def package_smoke(temporary: Path) -> None:
    dist = temporary / "dist"
    install = temporary / "wheel-install"
    run_stage(
        "source and wheel build",
        [sys.executable, "-m", "build", "--outdir", str(dist)],
    )
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"FAILED: expected one wheel in {dist}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
    required_members = {
        "OpenMiChroM/share/MiChroM.ff",
        "OpenMiChroM/_cndb_stream/py.typed",
        "OpenMiChroM/_cndb_stream/_vendor/hdf5_indexed_reader/LICENSE",
    }
    missing = required_members - members
    if missing:
        raise SystemExit(f"FAILED: wheel is missing package data: {sorted(missing)}")
    if any(member.startswith("tests/") for member in members):
        raise SystemExit("FAILED: wheel unexpectedly contains the repository test package")
    run_stage(
        "isolated wheel install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--target",
            str(install),
            str(wheels[0]),
        ],
    )
    smoke = (
        "import OpenMiChroM, OpenMiChroM.ChromDynamics, OpenMiChroM.CndbTools; "
        "import OpenMiChroM.Converters; "
        "import OpenMiChroM.CustomReporter, OpenMiChroM.Integrators, "
        "OpenMiChroM.Optimization, OpenMiChroM._cndb_stream; "
        "print(OpenMiChroM.__version__)"
    )
    smoke_env = {**os.environ, "PYTHONPATH": str(install)}
    run_stage("installed-wheel public imports", [sys.executable, "-c", smoke], cwd=temporary, env=smoke_env)
    converter_cli = install / "bin" / "openmichrom-convert"
    if not converter_cli.is_file():
        raise SystemExit(f"FAILED: installed wheel is missing converter CLI: {converter_cli}")
    run_stage(
        "installed-wheel converter CLI",
        [sys.executable, str(converter_cli), "--help"],
        cwd=temporary,
        env=smoke_env,
    )


def gpu_smoke() -> None:
    code = (
        "from openmm import Platform; "
        "platform = Platform.getPlatformByName('CUDA'); "
        "print(platform.getName())"
    )
    run_stage("optional CUDA platform smoke", [sys.executable, "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fast", "complete"), nargs="?", default="fast")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Include the optional live ENCODE streaming test.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require and smoke-test OpenMM's CUDA platform.",
    )
    parser.add_argument(
        "--full-science",
        action="store_true",
        help="Use production tutorial step counts (requires --network and can take many hours).",
    )
    parser.add_argument("--notebook-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.full_science and args.mode != "complete":
        parser.error("--full-science is available only in complete mode")
    if args.full_science and not args.network:
        parser.error("--full-science requires --network for external-data tutorial cells")

    run_stage("notebook static audit", [sys.executable, "scripts/audit_notebooks.py"])
    run_stage(
        "canonical/docs notebook synchronization",
        [sys.executable, "scripts/sync_tutorials.py", "--check", "--clear-outputs"],
    )

    test_command = [sys.executable, "-m", "pytest", "-q"]
    if args.mode == "fast":
        test_command.extend(FOCUSED_TESTS)
        if args.network:
            test_command.append("tests/test_cndbtools_encode_stream.py")
    test_env = dict(os.environ)
    if args.network:
        test_env["OPENMICHROM_RUN_ENCODE_STREAM_TESTS"] = "1"
    run_stage(f"{args.mode} pytest suite", test_command, env=test_env)

    notebook_command = [
        sys.executable,
        "scripts/execute_notebooks.py",
        "--mode",
        args.mode,
        "--timeout",
        str(args.notebook_timeout),
    ]
    if args.network:
        notebook_command.append("--allow-network")
    if args.full_science:
        notebook_command.append("--full-science")
    run_stage(f"{args.mode} notebook execution", notebook_command)

    if args.gpu:
        gpu_smoke()

    if args.mode == "complete":
        with tempfile.TemporaryDirectory(prefix="openmichrom-validation-") as temporary_name:
            temporary = Path(temporary_name)
            run_stage(
                "Sphinx documentation (warnings are errors)",
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-W",
                    "--keep-going",
                    "-b",
                    "html",
                    "docs/source",
                    str(temporary / "html"),
                ],
            )
            package_smoke(temporary)

    print(f"\nVALIDATION PASS: mode={args.mode}, network={args.network}, gpu={args.gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
