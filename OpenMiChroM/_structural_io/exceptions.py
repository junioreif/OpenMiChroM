"""Exceptions for OpenMiChroM structural file I/O helpers."""


class StructuralIOError(Exception):
    """Base class for structural file I/O errors."""


class StructuralDetectionError(StructuralIOError):
    """Raised when a structural file cannot be safely inspected."""
