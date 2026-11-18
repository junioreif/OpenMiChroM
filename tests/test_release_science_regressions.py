"""Numeric contracts for IC forces, training statistics and PDB conversion."""

from contextlib import nullcontext

import numpy as np
import openmm
from openmm import unit
import pytest

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.Optimization import AdamTraining, CustomMiChroMTraining


def test_compact_ic_table_includes_last_separation_in_openmm(tmp_path):
    params = tmp_path / "ic.dat"
    np.savetxt(params, [-0.2, -0.8])
    sim = MiChroM(verbose=False)
    sim.setup(platform="CPU")
    sim.N = 4
    sim.addCustomIC(dinit=2, dend=3, IClist=params)
    force = sim.forceDict["CustomIC"]
    np.testing.assert_array_equal(force.getTabulatedFunction(0).getFunctionParameters(), [0, 0, -0.2, -0.8])
    force.addInteractionGroup({0}, {3})
    system = openmm.System()
    for _ in range(4):
        system.addParticle(1)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0]])
    energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert energy == pytest.approx(-0.8 * (1 + np.tanh(3.22 * (1.78 - 1.5))) / 2)
    with pytest.raises(ValueError, match="3 are required"):
        sim.addCustomIC(dinit=1, dend=3, IClist=params)
    sim.chains = [(1, 3, 0)]
    sim.addCustomMultiChainIC(dinit=2, dend=200, chainIndex=0, IClist=params)
    chain_force = sim.forceDict["CustomIC_chain_0"]
    np.testing.assert_array_equal(chain_force.getTabulatedFunction(0).getFunctionParameters(), [0, 0, -0.2, 0])


@pytest.mark.parametrize("method", ["getHiCexp", "getPars", "get_HiC_exp"])
def test_contact_probabilities_reflect_upper_triangle_once(tmp_path, method):
    counts = np.array([[0, 8, 2, 16], [8, 0, 0, 3], [2, 0, 0, 2], [16, 3, 2, 0]])
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    trainer = (CustomMiChroMTraining.__new__(CustomMiChroMTraining) if method == "get_HiC_exp"
               else AdamTraining(update_storagePath=tmp_path / "adam"))
    with pytest.warns(DeprecationWarning) if method == "getPars" else nullcontext():
        getattr(trainer, method)(path, norm=True)
    np.testing.assert_array_equal(trainer.expHiC, [
        [0, 1, 0.25, 1], [1, 0, 0, 0], [0.25, 0, 0, 1], [1, 0, 1, 0],
    ])


@pytest.mark.parametrize("mode", ["IC", "types"])
def test_single_frame_training_uses_current_frame_and_inclusive_bounds(tmp_path, mode):
    seq = tmp_path / "seq.txt"
    seq.write_text("1 A1\n2 A1\n3 A1\n", encoding="utf-8")
    types = tmp_path / "types.csv"
    types.write_text("A1,B1\n-0.2,-0.3\n0,-0.4\n", encoding="utf-8")
    params = tmp_path / "ic.dat"
    np.savetxt(params, [-0.2, -0.1])
    trainer = CustomMiChroMTraining(seq, types, IClist=params, dinit=1, dend=200)
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    expected = (1 + np.tanh(3.22 * (1.78 - np.abs(np.subtract.outer([0, 1, 2], [0, 1, 2]))))) / 2
    getattr(trainer, f"prob_calculation_{mode}")(positions)
    np.testing.assert_allclose(trainer.get_HiC_sim(), expected)
    path = tmp_path / "counts.npy"
    np.save(path, expected)
    if mode == "IC":
        phi = [expected[0, 1], expected[0, 2]]
        np.testing.assert_allclose(trainer.calc_phi_sim_IC(), phi)
        np.testing.assert_allclose(trainer.get_PiPj_sim_IC(), np.outer(phi, phi))
        np.testing.assert_allclose(trainer.get_lambdas_IC(path, write_error=False), [-0.2, -0.1])
    else:
        # B1 is in the table but not present in this chromosome.
        updated = trainer.get_lambdas_types(path, write_error=False)
        np.testing.assert_allclose(updated["B1"], [-0.3, -0.4])
