"""Small CPU integration tests replacing the obsolete manual test runner."""

from pathlib import Path

import numpy as np
import pytest
from openmm import unit

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.CndbTools import CndbTools


@pytest.mark.parametrize("e_cut", [4.0, 2.0])
def test_softcore_energy_and_force_remain_finite_during_close_crossings(tmp_path, e_cut):
    sim = MiChroM(verbose=False)
    sim.setup(platform="CPU")
    sim.saveFolder(tmp_path)
    sim.loadStructure(np.array([[0., 0., 0.], [1., 0., 0.]]))
    sim.addRepulsiveSoftCore(eCut=e_cut)
    sim.createSimulation()

    # Compare both forces and energies with the uncapped analytic model on
    # the flat core, across its transition, on the LJ branch and past cutoff.
    # For e_cut=2 the join is exactly r0=1: test each side and equality.
    for r in (0.0001, 0.001, 0.5, 0.90, 0.96, 1 - 2**-14, 1.0, 1 + 2**-14, 1.05, 1.20):
        sim.context.setPositions([[0, 0, 0], [r, 0, 0]])
        state = sim.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        lj = 4 * (r**-12 - r**-6) + 1
        repulsion = 24 * (2 * r**-13 - r**-7)
        if r >= 2**(1/6):
            expected, repulsion = 0., 0.
        elif lj > e_cut / 2:
            t = np.tanh(2 * lj / e_cut - 1)
            expected, repulsion = e_cut / 2 * (1 + t), repulsion * (1 - t*t)
        else:
            expected = lj
        np.testing.assert_allclose(energy, expected, atol=1e-5)
        np.testing.assert_allclose(forces, [[-repulsion, 0, 0], [repulsion, 0, 0]], atol=2e-4)


@pytest.mark.parametrize("configure_before_load", [True, False])
def test_coordinates_define_a_full_chain_and_checked_runs_finish(tmp_path, configure_before_load):
    sim = MiChroM(verbose=False)
    sim.setup(platform="CPU")
    sim.saveFolder(str(tmp_path))
    if configure_before_load:
        sim.setChains()
    sim.loadStructure(np.array([[i, 0., 0.] for i in range(5)]))
    sim.addFENEBonds()
    assert sim.chains == [(0, 4, False)]
    assert sim.forceDict["FENEBond"].getNumBonds() == 4
    sim.createSimulation()
    for steps, block in [(3, 2), (1, 10)]:
        start = sim.simulation.currentStep
        sim.run(steps, checkSystem=True, blockSize=block, report=False)
        assert sim.simulation.currentStep == start + steps
    assert np.isfinite(sim.getPositions()).all()


@pytest.mark.parametrize("frames", [2, 12])
def test_chirality_follows_bead_axis_and_reverses_under_reflection(frames):
    t = np.arange(8, dtype=float)
    helix = np.column_stack((np.cos(t), np.sin(t), t))
    xyz = np.repeat(helix[None], frames, axis=0)
    tools = CndbTools()
    chi = tools.compute_Chirality(xyz)
    assert chi.shape == (frames, 3)
    assert np.all(np.abs(chi) > 0.1)
    np.testing.assert_allclose(tools.compute_Chirality(xyz * [-1, 1, 1]), -chi)


def test_cpu_simulation_reporting_and_structure_exports(tmp_path: Path):
    sequence = tmp_path / "sequence.txt"
    sequence.write_text(
        "".join(f"{index} {kind}\n" for index, kind in enumerate((["A1", "B1"] * 4))),
        encoding="utf-8",
    )

    simulation = MiChroM(name="smoke", temperature=1.0, timeStep=0.01)
    simulation.setup(platform="CPU")
    output = tmp_path / "output"
    simulation.saveFolder(str(output))
    positions = simulation.createSpringSpiral(ChromSeq=str(sequence))
    simulation.loadStructure(positions, center=True)
    simulation.addFENEBonds()
    simulation.addAngles()
    simulation.addRepulsiveSoftCore()
    simulation.createSimulation()
    simulation.createReporters(
        statistics=False,
        traj=True,
        outputName="trajectory",
        trajFormat="cndb",
        interval=1,
    )
    simulation.run(nsteps=2, report=False, blockSize=1)

    for mode in ("ndb", "pdb", "gro", "xyz"):
        simulation.saveStructure(fileName="snapshot", mode=mode)
        assert (output / f"snapshot.{mode}").is_file()

    for reporter in simulation.simulation.reporters:
        close = getattr(reporter, "close", None)
        if close is not None:
            close()

    with CndbTools().load(output / "trajectory_0.cndb") as trajectory:
        assert trajectory.Nframes == 2
        assert trajectory.Nbeads == 8
        assert trajectory.xyz().shape == (2, 8, 3)

    # Exercise the supported NDB auto-conversion against the actual current
    # simulation writer rather than only a hand-authored format fixture.
    with CndbTools().load(output / "snapshot.ndb") as snapshot:
        assert snapshot.Nframes == 1
        assert snapshot.Nbeads == 8
        assert snapshot.xyz().shape == (1, 8, 3)
