"""Small CPU integration tests replacing the obsolete manual test runner."""

from pathlib import Path

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.CndbTools import CndbTools


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
