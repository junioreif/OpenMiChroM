"""Internal structural file detection and routing helpers."""

from .detect import detect_structural_file
from .formats import StructuralFileInfo

__all__ = [
    "StructuralFileInfo",
    "detect_structural_file",
]
