import h5py
import numpy as np
import pytest

from OpenMiChroM.CndbTools import CndbTools
from OpenMiChroM._cndb_stream.exceptions import CNDBFormatError


@pytest.mark.parametrize("rows", [2, 4])
def test_local_frame_must_agree_with_bead_count_even_for_subselection(tmp_path, rows):
    path = tmp_path / "malformed.cndb"
    with h5py.File(path, "w") as h5:
        h5.attrs["format_version"] = "1"
        h5["types"] = [1, 2, 3]
        h5["1"] = np.zeros((rows, 3))
    with CndbTools.open(path) as reader, pytest.raises(CNDBFormatError, match="Frame '1'"):
        reader.xyz(frames=1, beadSelection=[0])


@pytest.mark.parametrize("options", [{}, {"normalization": "shell-volume"}], ids=["default", "explicit"])
def test_radial_shells_conserve_counts(options):
    tool = CndbTools()
    xyz = np.array([[[-3., 0, 0], [0., 0, 0], [3., 0, 0]]])
    density, mids = tool.compute_RadNumDens(xyz, dr=2, **options)
    np.testing.assert_array_equal(mids, [1, 3])
    volumes = 4 * np.pi / 3 * np.diff(np.array([0., 2., 4.]) ** 3)
    np.testing.assert_allclose(density * volumes, [1, 2])
    mids, rdp = tool.compute_RDP(xyz, np.array([0, 1, 2]), radius=3, bins=2, **options)
    np.testing.assert_array_equal(mids, [0.75, 2.25])
    volumes = 4 * np.pi / 3 * np.diff(np.array([0., 1.5, 3.]) ** 3)
    np.testing.assert_allclose(rdp * volumes, [1, 2])


def test_legacy_radial_normalization_reproduces_historical_radii_and_densities():
    tool = CndbTools()
    xyz = np.array([[[-3., 0, 0], [0., 0, 0], [3., 0, 0]]])
    density, mids = tool.compute_RadNumDens(xyz, dr=2, normalization="legacy")
    np.testing.assert_array_equal(mids, [1])
    np.testing.assert_allclose(density, [1 / (4 * np.pi / 3 * 2**3)])
    radii, rdp = tool.compute_RDP(xyz, np.array([0, 1, 2]), radius=4, bins=4,
                                  normalization="legacy")
    np.testing.assert_array_equal(radii, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(rdp, [0, 0, 2 / (4 * np.pi * 3**2), 0])
