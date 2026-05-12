"""Internal CNDB streaming backend for OpenMiChroM CNDBTools."""

from .analysis import distance_matrix, radius_of_gyration
from .exceptions import (
    CNDBIndexError,
    CNDBStreamError,
    FrameNotFoundError,
    RangeRequestUnsupportedError,
    UnsupportedLayoutError,
)
from .reader import IndexedCNDB

__all__ = [
    "CNDBIndexError",
    "CNDBStreamError",
    "FrameNotFoundError",
    "IndexedCNDB",
    "RangeRequestUnsupportedError",
    "UnsupportedLayoutError",
    "distance_matrix",
    "radius_of_gyration",
]
