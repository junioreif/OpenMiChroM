"""Custom exceptions for the OpenMiChroM CNDB streaming backend."""


class CNDBStreamError(Exception):
    """Base exception for CNDB streaming backend failures."""


class CNDBIndexError(CNDBStreamError):
    """Raised when an index is missing, invalid, or incompatible."""


class CNDBFormatError(CNDBStreamError):
    """Raised when a CNDB file or its format metadata is invalid."""


class UnsupportedCNDBVersionError(CNDBFormatError):
    """Raised when a CNDB file declares an unsupported format version."""


class LegacyCNDBVersionWarning(UserWarning):
    """Warn that a legacy CNDB file has no current version metadata."""


class UnsupportedLayoutError(CNDBStreamError):
    """Raised when a dataset layout cannot be read by the MVP byte reader."""


class RangeRequestUnsupportedError(CNDBStreamError):
    """Raised when a remote server does not honor HTTP Range requests."""


class RemoteAccessError(CNDBStreamError):
    """Raised when a remote CNDB resource cannot be accessed completely."""


class FrameNotFoundError(CNDBStreamError):
    """Raised when a requested frame is absent from the CNDB index."""
