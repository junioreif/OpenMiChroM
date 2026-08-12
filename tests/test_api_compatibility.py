import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

import OpenMiChroM
from OpenMiChroM.CndbTools import CndbTools, cndbTools
from OpenMiChroM.Optimization import AdamTraining, CustomMiChroMTraining


@pytest.mark.parametrize(
    "module_name",
    [
        "OpenMiChroM",
        "OpenMiChroM.ChromDynamics",
        "OpenMiChroM.CndbTools",
        "OpenMiChroM.CustomReporter",
        "OpenMiChroM.Integrators",
        "OpenMiChroM.Optimization",
        "OpenMiChroM._cndb_stream",
    ],
)
def test_public_modules_import(module_name):
    assert importlib.import_module(module_name) is not None


def test_cndbtools_class_capitalization_alias_is_backward_compatible():
    assert CndbTools is cndbTools
    assert OpenMiChroM.CndbTools is cndbTools
    assert OpenMiChroM.cndbTools is cndbTools


def test_adam_getpars_deprecated_wrapper_matches_supported_method(tmp_path):
    matrix = np.array(
        [
            [1.0, 4.0, 2.0, 1.0],
            [4.0, 1.0, 3.0, 1.0],
            [2.0, 3.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 1.0],
        ]
    )
    hic_path = tmp_path / "hic.npy"
    np.save(hic_path, matrix)
    supported = AdamTraining(update_storagePath=tmp_path / "supported")
    deprecated = AdamTraining(update_storagePath=tmp_path / "deprecated")

    supported.getHiCexp(hic_path, neighbors=1)
    with pytest.warns(DeprecationWarning, match="getHiCexp"):
        deprecated.getPars(hic_path, neighbors=1)

    np.testing.assert_allclose(deprecated.expHiC, supported.expHiC, rtol=0, atol=0)
    np.testing.assert_allclose(deprecated.Pi, supported.Pi, rtol=0, atol=0)
    assert deprecated.NFrames == supported.NFrames == 0


def test_training_normalization_wrappers_preserve_identical_numerical_behavior(tmp_path):
    matrix = np.array(
        [
            [np.nan, 4.0, 2.0],
            [4.0, np.inf, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
        ]
    )
    adam = AdamTraining(update_storagePath=tmp_path / "adam")
    custom = CustomMiChroMTraining.__new__(CustomMiChroMTraining)

    np.testing.assert_allclose(adam.normalize_matrix(matrix.copy()), expected)
    np.testing.assert_allclose(custom.normalize_matrix(matrix.copy()), expected)


def test_no_duplicate_method_definitions_in_first_party_modules():
    package_dir = Path(OpenMiChroM.__file__).parent
    duplicates = []
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
            duplicates.extend(
                f"{path.name}:{node.name}.{name}" for name in names if names.count(name) > 1
            )
    assert sorted(set(duplicates)) == []
