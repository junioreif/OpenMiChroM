"""Test validation routing and the dynamics check without requiring any GPU."""
import sys

import openmm as mm
import pytest

from scripts import validate


@pytest.mark.parametrize("argv, expected", [
    ([], []), (["--gpu"], ["CUDA"]), (["--gpu-platform", "OpenCL"], ["OpenCL"]),
    (["--gpu-platform", "CUDA"], ["CUDA"]),
])
def test_optional_gpu_backend(monkeypatch, argv, expected):
    seen = []
    monkeypatch.setattr(sys, "argv", ["validate.py", "fast", *argv])
    monkeypatch.setattr(validate, "run_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(validate, "gpu_smoke", seen.append)
    assert validate.main() == 0
    assert seen == expected


def test_missing_gpu_cannot_fall_back(monkeypatch):
    def missing(name):
        raise RuntimeError(f"{name} unavailable")

    monkeypatch.setattr(mm.Platform, "getPlatformByName", missing)
    with pytest.raises(RuntimeError, match="OpenCL unavailable"):
        validate.gpu_smoke("OpenCL")


@pytest.mark.parametrize("stiffness", [0.0, 100.0])
def test_dynamics_check_uses_forces_not_just_step_count(stiffness):
    system = mm.System()
    for _ in range(2):
        system.addParticle(1.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 1.0, stiffness)
    system.addForce(bond)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0, 0, 0], [1.25, 0, 0]])
    context.setVelocities([[0, 0, 0], [0, 0, 0]])
    if stiffness:
        validate._check_dynamics(context, integrator)
        assert context.getStepCount() == 10
    else:
        with pytest.raises(RuntimeError, match="zero forces"):
            validate._check_dynamics(context, integrator)
