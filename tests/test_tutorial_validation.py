"""Regression checks for portable, repeatable tutorial validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_notebooks, execute_notebooks, sync_tutorials


REPO_ROOT = Path(__file__).resolve().parents[1]


def cells(relative):
    return json.loads((REPO_ROOT / "Tutorials" / relative).read_text())["cells"]


def source(cell):
    return "".join(cell["source"])


def test_all_canonical_notebooks_are_documented_and_executed():
    paths = set(audit_notebooks.notebook_paths())
    documented = [tutorial.source_path for tutorial in sync_tutorials.TUTORIALS]
    executed = [run.path for run in execute_notebooks.RUNS]
    assert set(documented) == set(executed) == paths
    assert len(documented) == len(set(documented))
    assert len(executed) == len(set(executed))


def test_bundled_inputs_match_tutorial_generation():
    inputs = REPO_ROOT / "Tutorials/Chromosome_Pulling_Tutorial/input"
    np.testing.assert_array_equal(
        np.loadtxt(inputs / "DT40_chr7.txt", dtype=str),
        np.loadtxt(inputs / "DT40_chr7.eigen", dtype=str),
    )
    inputs = REPO_ROOT / "Tutorials/MiChroM_Optimization/input"
    sequence = np.loadtxt(inputs / "seq_chr10_100k.txt", dtype=str)
    np.testing.assert_array_equal(np.loadtxt(inputs / "lambda_IC_0"), np.zeros(len(sequence) - 2))


DT40 = Path("DT40_Annealing_50kb/Tutorial_DT40_Annealing_50kb.ipynb")


def test_dt40_inputs_and_annotation():
    from openmm import app, unit

    inputs = REPO_ROOT / "Tutorials" / DT40.parent / "input"
    sequence = np.loadtxt(inputs / "DT40_chr7.txt", dtype=str)
    np.testing.assert_array_equal(sequence[:, 0].astype(int), np.arange(1, 740))
    structures = []
    for filename in ("chr7_10m-Q.gro", "chr7_15m-Q.gro"):
        gro = app.GromacsGroFile(str(inputs / filename))
        xyz = gro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert xyz.shape == (739, 3) and np.isfinite(xyz).all()
        np.testing.assert_array_equal(sequence[:, 1], [
            {"ASP": "A1", "GLU": "B1"}[r] for r in gro.residueNames
        ])
        structures.append(xyz)
    assert not np.allclose(*structures)
    types = np.genfromtxt(inputs / "lambdas_types_15m", delimiter=",", names=True)
    assert types.dtype.names == ("A1", "B1")
    matrix = np.column_stack([types[t] for t in types.dtype.names])
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T)
    ic = np.loadtxt(inputs / "lambdas_ic_15m")
    assert ic.ndim == 1 and ic.size > 735 and np.isfinite(ic).all()


def test_dt40_archived_periods_and_input_discovery(monkeypatch):
    from openmm import app, unit

    monkeypatch.chdir(REPO_ROOT)
    body = {c["id"]: source(c) for c in cells(DT40)}
    namespace = {}
    exec(compile(body["inputs"], str(DT40), "exec"), namespace)
    inputs = namespace["data"]
    assert inputs.resolve() == REPO_ROOT / "Tutorials" / DT40.parent / "input"
    gro = app.GromacsGroFile(str(inputs / "chr7_15m-Q.gro"))
    xyz = gro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    namespace.update(reference=xyz, cold=xyz[None])
    exec(compile(body["spectrum"], str(DT40), "exec"), namespace)
    periods = 1 / namespace["freq"][namespace["prominent"]]
    np.testing.assert_allclose(periods, [3.670, 0.322], atol=0.001)


@pytest.mark.parametrize("path", audit_notebooks.notebook_paths())
def test_tutorials_do_not_require_shell_magics(path):
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not source(cell).lstrip().startswith(("%%bash", "%%script", "!"))


INPUT_NOTEBOOKS = [
    run.path.relative_to(REPO_ROOT / "Tutorials")
    for run in execute_notebooks.RUNS
    if any(cell.get("id") == "openmichrom-mode-code" and "tutorial_dir" in source(cell)
           for cell in json.loads(run.path.read_text())["cells"])
]


@pytest.mark.parametrize("relative", INPUT_NOTEBOOKS)
def test_bundled_inputs_resolve_from_repository_root(relative, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("OPENMICHROM_TUTORIAL_MODE", "fast")
    setup = next(cell for cell in cells(relative) if cell.get("id") == "openmichrom-mode-code")
    exec(compile(source(setup), str(relative), "exec"), {})
    assert Path.cwd() == REPO_ROOT / "Tutorials" / relative.parent


@pytest.mark.parametrize("relative,index,input_name", [
    ("Chromosome_Pulling_Tutorial/Tutorial_Pulling.ipynb", 5, "DT40_chr7.txt"),
    ("MiChroM_Optimization/Tutorial_MiChroM_Optimization.ipynb", 16, "seq_chr10_100k.txt"),
    ("Full_Inversion_Optimization/Tutorial_Full_Inversion_Optimization.ipynb", 14, "seq_chr10"),
    ("MiChroM_Optimization/Tutorial_MiChroM_Optimization.ipynb", 72, "lambda_IC_0"),
])
def test_input_generation_is_repeatable(relative, index, input_name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    eigen = np.array([-1.0, 1.0, 0.0, -0.5, np.nan])
    # The bundled pulling input is already annotated, not a raw eigenvector.
    annotations = ["B1", "A1", "A1", "B1", "A1"]
    (tmp_path / "input/DT40_chr7.eigen").write_text(
        "".join(f"{i} {kind}\n" for i, kind in enumerate(annotations, 1))
    )
    (tmp_path / "input/seq_chr10_100k.txt").write_text(
        "1 A1\n2 B1\n3 B1\n4 A1\n5 B1\n"
    )
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    namespace = {"np": np, "Path": Path, "eigen": eigen, "r": np.eye(5), "run_dir": output_dir}
    code = compile(source(cells(relative)[index]), str(relative), "exec")
    exec(code, namespace)
    target = (output_dir if input_name == "DT40_chr7.txt" else tmp_path / "input") / input_name
    first = target.read_bytes()
    exec(code, namespace)
    assert target.read_bytes() == first
    if input_name == "lambda_IC_0":
        np.testing.assert_array_equal(np.loadtxt(target), np.zeros(3))
    else:
        rows = target.read_text().splitlines()
        assert len(rows) == 5
        assert [int(row.split()[0]) for row in rows] == list(range(1, 6))
        if input_name == "DT40_chr7.txt":
            assert [row.split()[1] for row in rows] == ["B1", "A1", "A1", "B1", "A1"]
        elif input_name == "seq_chr10_100k.txt":
            assert [row.split()[1] for row in rows] == ["A1", "B1", "B1", "A1", "B1"]


@pytest.mark.parametrize("energy, bond, stable", [(3.0, 1.1, True), (np.nan, 1.1, False),
                                                  (3.0, 1.5, False), (3.0, np.inf, False)])
def test_pulling_trajectory_guard(energy, bond, stable):
    from types import SimpleNamespace
    from openmm import unit

    state = SimpleNamespace(
        getPositions=lambda asNumpy: np.array([[0., 0., 0.], [bond, 0., 0.]]) * unit.nanometer,
        getPotentialEnergy=lambda: energy * unit.kilojoule_per_mole,
        getKineticEnergy=lambda: 1.0 * unit.kilojoule_per_mole,
    )
    sim = SimpleNamespace(context=SimpleNamespace(getState=lambda **kwargs: state,
                                                  getParameter=lambda name: 1.5),
                          simulation=SimpleNamespace(currentStep=5))
    setup = next(c for c in cells("Chromosome_Pulling_Tutorial/Tutorial_Pulling.ipynb")
                 if source(c).startswith("def checked_positions("))
    namespace = {"np": np, "unit": unit}
    exec(source(setup), namespace)
    if stable:
        assert namespace["checked_positions"](sim).shape == (2, 3)
    else:
        with pytest.raises(RuntimeError, match="Unstable trajectory at step 5"):
            namespace["checked_positions"](sim)


@pytest.mark.parametrize("args", [
    ["--mode", "fast", "--full-science", "--allow-network"],
    ["--mode", "complete", "--full-science"],
])
def test_full_science_cannot_silently_skip_notebooks(args, monkeypatch):
    monkeypatch.setattr("sys.argv", ["execute_notebooks.py", *args])
    with pytest.raises(SystemExit) as exc:
        execute_notebooks.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("fast,positions,expected", [
    (True, [[50.0, 0.0, 0.0]], 52.0),
    (False, [[0.0, 0.0, 0.0]], None),
])
def test_spherical_wall_transition_respects_validation_mode(fast, positions, expected):
    class Simulation:
        N = 1000

        def getPositions(self):
            return np.asarray(positions)

        def removeFlatBottomHarmonic(self):
            self.removed = True

        def addSphericalConfinementLJ(self, radius):
            self.wall_radius = radius

        def addAdditionalForce(self, function, **kwargs):
            function(**kwargs)

    simulation = Simulation()
    code = source(cells("Chromosome_simulations/Tutorial_Multiple_Chromosomes.ipynb")[40])
    exec(code, {"np": np, "sim_nucleus": simulation, "TUTORIAL_FAST": fast})
    density_radius = (3 * simulation.N / (4 * np.pi * 0.1)) ** (1 / 3)
    assert simulation.wall_radius == pytest.approx(expected or density_radius)
    assert simulation.removed


def test_production_wall_rejects_insufficiently_collapsed_structure():
    class Simulation:
        N = 1000

        def getPositions(self):
            return np.array([[50.0, 0.0, 0.0]])

        def removeFlatBottomHarmonic(self):
            pytest.fail("Must reject unsafe coordinates before changing forces")

    code = source(cells("Chromosome_simulations/Tutorial_Multiple_Chromosomes.ipynb")[40])
    with pytest.raises(ValueError, match="Extend the harmonic-collapse"):
        exec(code, {"np": np, "sim_nucleus": Simulation(), "TUTORIAL_FAST": False})
