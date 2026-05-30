"""Internal structural file detection and routing helpers."""

from .detect import detect_structural_file
from .converters import convert_structure_file
from .formats import StructuralFileInfo
from .selections import coalesce_indices

__all__ = [
    "StructuralFileInfo",
    "coalesce_indices",
    "convert_structure_file",
    "detect_structural_file",
]
