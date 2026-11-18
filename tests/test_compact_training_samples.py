"""Regression tests on the archived 24-bead windows in training/samples."""
from pathlib import Path
from importlib.resources import files

import numpy as np
import pytest

from OpenMiChroM.CndbTools import CndbTools
from OpenMiChroM.Optimization import AdamTraining, CustomMiChroMTraining
from OpenMiChroM._cndb_stream import LegacyCNDBVersionWarning


SAMPLES = Path(__file__).parent / "training" / "samples"


def _probabilities(counts):
    """Reference normalization: upper triangle, reflecting each observed pair once."""
    contacts = np.zeros_like(counts)
    for i in range(len(counts) - 1):
        nearest = counts[i, i + 1]
        if nearest > 0:
            for j in range(i + 1, len(counts)):
                contacts[i, j] = contacts[j, i] = np.clip(counts[i, j] / nearest, 0, 1)
    return contacts


def test_training_upper_triangle(tmp_path):
    path = SAMPLES / "chr10_upper_probability.dense"
    raw = np.loadtxt(path)
    assert np.count_nonzero(np.tril(raw, -1)) == 0 and np.any(raw)
    trainer = AdamTraining(update_storagePath=tmp_path)
    trainer.getHiCexp(path)
    np.testing.assert_allclose(trainer.phi_exp, _probabilities(raw))


def test_training_with_archived_chr18_frames():
    types = files("OpenMiChroM").joinpath("share", "MiChroM.ff")
    trainer = CustomMiChroMTraining(ChromSeq=SAMPLES / "chr18.seq", TypesTable=types,
                                   dinit=3, dend=12)
    with pytest.warns(LegacyCNDBVersionWarning):
        reader = CndbTools.open(SAMPLES / "chr18.cndb")
    with reader:
        frames = reader.xyz().astype(np.float64)
        np.testing.assert_array_equal(reader.ChromSeq, trainer.ChromSeq)
    assert frames.shape == (3, 24, 3)
    for frame in frames:
        trainer.prob_calculation_IC(frame)
    distances = np.linalg.norm(frames[:, :, None] - frames[:, None, :], axis=-1)
    contacts = ((1 + np.tanh(trainer.mu * (trainer.r_cut - distances))) / 2).mean(axis=0)
    np.testing.assert_allclose(trainer.get_HiC_sim(), contacts)
    np.testing.assert_allclose(trainer.calc_phi_sim_IC(), [contacts.diagonal(d).mean() for d in range(3, 13)])
    counts = SAMPLES / "chr18_counts.dense"
    trainer.get_HiC_exp(counts, norm=True)
    np.testing.assert_allclose(trainer.expHiC, _probabilities(np.loadtxt(counts)))


def test_archived_chr19_adam_step(tmp_path):
    trainer = AdamTraining(eta=0.01, update_storagePath=tmp_path)
    counts = SAMPLES / "chr19_counts.dense"
    trainer.getHiCexp(counts)
    exp = _probabilities(np.loadtxt(counts))
    with np.load(SAMPLES / "chr19_accumulator.npz", allow_pickle=False) as data:
        trainer.Pi += data["Pi"]
        trainer.NFrames = int(data["frames"])
        simulated = np.where(exp != 0, data["Pi"] / trainer.NFrames, 0)
    gradient = exp - simulated
    assert trainer.NFrames == 75 and np.any(gradient < 0) and np.any(gradient > 0)
    path = SAMPLES / "chr19_lambda.csv"
    initial = np.loadtxt(path, delimiter=",", skiprows=1)
    expected = initial - trainer.eta * gradient / (np.abs(gradient) + trainer.epsilon)
    actual = trainer.getLamb(path)
    assert list(actual.columns) == [f"Bead{i}" for i in range(1, 25)]
    np.testing.assert_allclose(actual.to_numpy(), expected, atol=1e-12, rtol=0)
