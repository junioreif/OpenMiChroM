"""Integrity and offline selection of the shared GM12878 optimization input."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from Tutorials.optimization_data.make_sample import SAMPLE_NAME


TUTORIALS = Path(__file__).resolve().parents[1] / "Tutorials"
SAMPLE = TUTORIALS / "optimization_data" / SAMPLE_NAME


def test_chr10_sample_retains_original_tokens_and_bin_order():
    # Digest of the original dense file's upper left 512x512 text entries,
    # joined with tabs and LF. Includes every original NaN and zero.
    assert hashlib.sha256(gzip.decompress(SAMPLE.read_bytes())).hexdigest() == (
        "6c7ad0d61810194fe2fffaf1ccd784681aa8183ded111aab426c699b089328d4"
    )
    matrix = np.loadtxt(SAMPLE)
    assert matrix.shape == (512, 512)
    np.testing.assert_array_equal(matrix, matrix.T)


@pytest.mark.parametrize("tutorial", ["Full_Inversion_Optimization", "MiChroM_Optimization"])
def test_full_map_is_opt_in_and_hash_checked(tutorial, tmp_path, monkeypatch):
    import os

    path = TUTORIALS / tutorial / f"Tutorial_{tutorial}.ipynb"
    code = next("".join(c["source"]) for c in json.loads(path.read_text())["cells"]
                if c["cell_type"] == "code" and "FULL_CHROMOSOME =" in "".join(c["source"]))
    monkeypatch.chdir(path.parent)
    monkeypatch.delenv("OPENMICHROM_CHR10_FULL", raising=False)
    ns = {"os": os, "Path": Path}
    exec(code, ns)
    assert ns["dense_path"].resolve() == SAMPLE
    work = tmp_path / tutorial
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("OPENMICHROM_CHR10_FULL", "1")
    with pytest.raises(FileNotFoundError, match="chr10_100k.dense"):
        exec(code, ns)
    shared = tmp_path / "optimization_data"
    shared.mkdir()
    (shared / "chr10_100k.dense").write_text("not the original map")
    with pytest.raises(ValueError, match="does not match"):
        exec(code, ns)
