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
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
    )
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"FAILED: expected one wheel in {dist}, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = set(wheel.namelist())
    required_members = {
        "OpenMiChroM/share/MiChroM.ff",
        "OpenMiChroM/_cndb_stream/py.typed",
        "OpenMiChroM/_cndb_stream/embedded_writer.py",
        "OpenMiChroM/_structural_io/converters.py",
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
            "--no-index",
            "--quiet",
            "--no-deps",
            "--target",
            str(install),
            str(wheels[0]),
        ],
    )
    smoke = (
        "import OpenMiChroM, OpenMiChroM.ChromDynamics, OpenMiChroM.CndbTools; "
        "import OpenMiChroM.Converters, OpenMiChroM.Extrusion_Bonds, "
        "OpenMiChroM.StructuralVariants; "
        "import OpenMiChroM.CustomReporter, OpenMiChroM.Integrators, "
        "OpenMiChroM.Optimization, OpenMiChroM._cndb_stream; "
        "from pathlib import Path; from importlib.metadata import version; "
        "import os; assert Path(OpenMiChroM.__file__).resolve().is_relative_to(Path(os.environ['PYTHONPATH'])); "
        "from OpenMiChroM.CndbTools import CndbTools; "
        "assert version('OpenMiChroM') == OpenMiChroM.__version__; "
        "assert callable(CndbTools.open) and callable(CndbTools.convert); "
        "assert callable(OpenMiChroM.apply_structural_variant); "
        "assert callable(OpenMiChroM.LoopExtrusionManager); "
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


def _check_opencl_gpu(platform_index: int, device_index: int, expected_name: str) -> None:
    """OpenCL also runs on CPUs. Verify the actual context's device via its ICD."""
    import ctypes as ct
    from ctypes.util import find_library

    path = find_library("OpenCL")
    if not path:
        raise RuntimeError("Cannot verify OpenCL GPU device type: OpenCL runtime not found")
    lib = (ct.WinDLL if os.name == "nt" else ct.CDLL)(path)
    ptr, uint, ulong, size = ct.c_void_p, ct.c_uint, ct.c_uint64, ct.c_size_t
    lib.clGetPlatformIDs.argtypes = [uint, ct.POINTER(ptr), ct.POINTER(uint)]
    lib.clGetDeviceIDs.argtypes = [ptr, ulong, uint, ct.POINTER(ptr), ct.POINTER(uint)]
    lib.clGetDeviceInfo.argtypes = [ptr, uint, size, ptr, ct.POINTER(size)]

    def check(code):
        if code:
            raise RuntimeError(f"Unable to query OpenCL device type (OpenCL error {code})")

    def handles(query, *args):
        count = uint()
        check(query(*args, 0, None, ct.byref(count)))
        items = (ptr * count.value)()
        check(query(*args, count, items, None))
        return items

    platform = handles(lib.clGetPlatformIDs)[platform_index]
    device = handles(lib.clGetDeviceIDs, platform, 0xffffffff)[device_index]  # CL_DEVICE_TYPE_ALL
    length = size()
    check(lib.clGetDeviceInfo(device, 0x102B, 0, None, ct.byref(length)))  # CL_DEVICE_NAME
    name = ct.create_string_buffer(length.value)
    check(lib.clGetDeviceInfo(device, 0x102B, length, name, None))
    device_type = ulong()
    check(lib.clGetDeviceInfo(device, 0x1000, ct.sizeof(device_type), ct.byref(device_type), None))
    if name.value.decode().strip() != expected_name.strip() or not device_type.value & 4:  # CL_DEVICE_TYPE_GPU
        raise RuntimeError(f"Selected OpenCL device is not a verified GPU: {expected_name}")


def _check_dynamics(context, integrator) -> None:
    import numpy as np
    from openmm import unit

    def read():
        state = context.getState(positions=True, energy=True, forces=True)
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not (np.isfinite(xyz).all() and np.isfinite(forces).all()
                and np.isfinite([energy, kinetic]).all() and np.any(forces != 0)):
            raise RuntimeError("GPU smoke test produced nonfinite values or zero forces")
        return xyz

    start_xyz = read()
    start_step = context.getStepCount()
    integrator.step(10)
    if context.getStepCount() != start_step + 10 or np.array_equal(start_xyz, read()):
        raise RuntimeError("GPU smoke test did not advance particle coordinates and step count")


def gpu_smoke(platform_name: str = "CUDA") -> None:
    import openmm as mm

    if platform_name not in ("CUDA", "OpenCL"):
        raise ValueError(f"Unsupported GPU backend: {platform_name}")
    requested = mm.Platform.getPlatformByName(platform_name)
    system = mm.System()
    system.addParticle(1)
    system.addParticle(1)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.0, 100.0)
    system.addForce(bond)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, requested, {"Precision": "single"})
    actual = context.getPlatform()
    if actual.getName() != platform_name:
        raise RuntimeError(f"Expected {platform_name}; got {actual.getName()}")
    name = actual.getPropertyValue(context, "DeviceName")
    if platform_name == "OpenCL":
        platform_index = int(actual.getPropertyValue(context, "OpenCLPlatformIndex"))
        indices = actual.getPropertyValue(context, "DeviceIndex").split(",")
        names = name.split(",") if len(indices) > 1 else [name]
        if len(names) != len(indices):
            raise RuntimeError(f"Cannot verify OpenCL device list: {name}")
        for index, device_name in zip(indices, names):
            _check_opencl_gpu(platform_index, int(index), device_name)
    context.setPositions([[0, 0, 0], [1.25, 0, 0]])
    context.setVelocities([[0, 0, 0], [0, 0, 0]])
    _check_dynamics(context, integrator)
    print(f"PASS: {platform_name} GPU dynamics on {name} (10 steps)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fast", "complete"), nargs="?", default="fast")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Include the optional live ENCODE streaming test.",
    )
    parser.add_argument("--public-sources", action="store_true",
                        help="Also test the public NDB, ENCODE and NCBI file servers (implies --network).")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="gpu_platform", action="store_const", const="CUDA",
                     help="Require a CUDA GPU (alias for --gpu-platform CUDA).")
    gpu.add_argument("--gpu-platform", choices=("CUDA", "OpenCL"),
                     help="Require a short dynamics test on the specified GPU backend.")
    parser.add_argument(
        "--full-science",
        action="store_true",
        help="Use production tutorial step counts (requires --network and can take many hours).",
    )
    parser.add_argument("--notebook-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.public_sources:
        args.network = True
    if args.full_science and args.mode != "complete":
        parser.error("--full-science is available only in complete mode")
    if args.full_science and not args.network:
        parser.error("--full-science requires --network for external-data tutorial cells")

    # Modest CPU defaults for laptop validation; explicit user settings win.
    for name in ("OPENMM_CPU_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")
    run_stage("notebook static audit", [sys.executable, "scripts/audit_notebooks.py"])
    run_stage(
        "canonical/docs notebook synchronization",
        [sys.executable, "scripts/sync_tutorials.py", "--check", "--clear-outputs"],
    )

    test_command = [sys.executable, "-m", "pytest", "-q"]
    test_env = dict(os.environ)
    if args.public_sources:
        test_env["OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION"] = "1"
    else:
        test_env.pop("OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION", None)
    if args.network:
        test_env["OPENMICHROM_RUN_ENCODE_STREAM_TESTS"] = "1"
    else:
        test_env.pop("OPENMICHROM_RUN_ENCODE_STREAM_TESTS", None)
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

    if args.gpu_platform:
        gpu_smoke(args.gpu_platform)

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

    print(f"\nVALIDATION PASS: mode={args.mode}, network={args.network}, gpu={args.gpu_platform or False}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
