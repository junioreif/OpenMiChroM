"""Internal CNDB streaming backend for OpenMiChroM CNDBTools."""

from .analysis import distance_matrix, radius_of_gyration
from .exceptions import (
    CNDBFormatError,
    CNDBIndexError,
    CNDBStreamError,
    FrameNotFoundError,
    LegacyCNDBVersionWarning,
    RangeRequestUnsupportedError,
    RemoteAccessError,
    UnsupportedCNDBVersionError,
    UnsupportedLayoutError,
)
from .reader import IndexedCNDB
from .version import CNDB_FORMAT_NAME, CNDB_FORMAT_VERSION

__all__ = [
    "CNDBFormatError",
    "CNDB_FORMAT_NAME",
    "CNDB_FORMAT_VERSION",
    "CNDBIndexError",
    "CNDBStreamError",
    "FrameNotFoundError",
    "IndexedCNDB",
    "LegacyCNDBVersionWarning",
    "RangeRequestUnsupportedError",
    "RemoteAccessError",
    "UnsupportedCNDBVersionError",
    "UnsupportedLayoutError",
    "distance_matrix",
    "radius_of_gyration",
]
