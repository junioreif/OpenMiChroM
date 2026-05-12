"""Custom exceptions for the OpenMiChroM CNDB streaming backend."""


class CNDBStreamError(Exception):
    """Base exception for CNDB streaming backend failures."""


class CNDBIndexError(CNDBStreamError):
    """Raised when an index is missing, invalid, or incompatible."""


class UnsupportedLayoutError(CNDBStreamError):
    """Raised when a dataset layout cannot be read by the MVP byte reader."""


class RangeRequestUnsupportedError(CNDBStreamError):
    """Raised when a remote server does not honor HTTP Range requests."""


class FrameNotFoundError(CNDBStreamError):
    """Raised when a requested frame is absent from the CNDB index."""
