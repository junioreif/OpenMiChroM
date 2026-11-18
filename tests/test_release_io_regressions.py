"""Regression tests for format limits and formerly inaccessible training paths."""

import numpy as np
import pytest
from openmm import unit

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.Converters import ConverterError, ndb_to_pdb
from OpenMiChroM.CustomReporter import SaveStructure
from OpenMiChroM.Optimization import CustomMiChroMTraining


def test_custom_training_balancing_supports_empty_bins(tmp_path):
    counts = np.array([[0., 2., 0.], [2., 0., 0.], [0., 0., 0.]])
    path = tmp_path / "counts.npy"
    np.save(path, counts)
    custom = CustomMiChroMTraining.__new__(CustomMiChroMTraining)
    custom.get_HiC_exp(path, norm=False, KR=True)
    np.testing.assert_allclose(custom.expHiC, [[0, 1, 0], [1, 0, 0], [0, 0, 0]], atol=1e-5)


def test_training_preserves_coefficients_for_absent_types(tmp_path):
    seq = tmp_path / "seq.txt"
    seq.write_text("1 A1\n2 A1\n3 A1\n", encoding="utf-8")
    table = tmp_path / "types.csv"
    table.write_text("A1,B1\n-0.2,-0.3\n0,-0.4\n", encoding="utf-8")
    trainer = CustomMiChroMTraining(seq, table)
    trainer.prob_calculation_types(np.zeros((3, 3)))
    path = tmp_path / "counts.npy"
    np.save(path, np.ones((3, 3)))
    updated = trainer.get_lambdas_types(path, write_error=False)
    np.testing.assert_array_equal(updated["B1"], [-0.3, -0.4])


@pytest.mark.parametrize(("model", "residue", "x", "field"), [
    (10000, 1, "0", "MODEL"), (1, 10000, "0", "residue"),
    (1, 1, "9999.9999", "coordinate"), (1, 1, "-1000.000", "coordinate"),
])
def test_pdb_overflow_rejected_before_creating_output(tmp_path, model, residue, x, field):
    path = tmp_path / "one.ndb"
    path.write_text(f"MODEL {model}\nCHROM 1 A1 C1 {residue} {x} 0 0 1 50000 0\nENDMDL\nEND\n", encoding="utf-8")
    output = tmp_path / "one.pdb"
    with pytest.raises(ConverterError, match=field):
        ndb_to_pdb(path, output)
    assert not output.exists()
    assert not output.with_suffix(".loops").exists()


def test_structure_snapshot_validates_all_chains_before_writing(tmp_path):
    simulation = MiChroM(verbose=False)
    simulation.chains = [(0, 0, 0), (1, 1, 0)]
    simulation.type_list_letter = ["A1", "B3"]
    simulation.folder = tmp_path
    simulation.getPositions = lambda: np.array([[0, 0, 0], [10000, 0, 0]])
    with pytest.raises(ValueError, match="PDB coordinate"):
        simulation.saveStructure("snapshot", mode="pdb")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(("step", "x", "field"), [(9999, 0, "MODEL"), (0, -1000, "coordinate")])
def test_periodic_reporter_rejects_pdb_overflow_before_writing(tmp_path, step, x, field):
    class State:
        def getPositions(self, asNumpy):
            return unit.Quantity(np.array([[0, 0, 0], [x, 0, 0]]), unit.nanometer)

    reporter = SaveStructure("trajectory", 1, mode="pdb", folder=tmp_path,
                             chains=[(0, 0, 0), (1, 1, 0)], typeListLetter=["A1", "B4"])
    reporter.step = step
    with pytest.raises(ValueError, match=field):
        reporter.report(None, State())
    assert not list(tmp_path.iterdir())
