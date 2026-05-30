"""Internal structural file detection and routing helpers."""

from .detect import detect_structural_file
from .formats import StructuralFileInfo
from .selections import coalesce_indices

__all__ = [
    "StructuralFileInfo",
    "coalesce_indices",
    "detect_structural_file",
]
