"""Small analysis helpers for coordinates loaded by the CNDB streaming backend."""

from __future__ import annotations

import numpy as np


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix for coordinates.

    This implementation uses NumPy broadcasting and is O(N^2) in memory and
    time. For very large systems, call it on bead subsets rather than full
    chromosome-scale frames.
    """

    coords = np.asarray(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def radius_of_gyration(coords: np.ndarray) -> float:
    """Return the radius of gyration for one coordinate frame."""

    coords = np.asarray(coords)
    center = coords.mean(axis=0)
    return float(np.sqrt(((coords - center) ** 2).sum(axis=1).mean()))
