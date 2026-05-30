"""Data models for structural file detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StructuralFileInfo:
    """Summary of a local or remote structural trajectory file."""

    source: str
    is_remote: bool
    file_type: str = "unknown"
    layout: str = "unknown"
    has_embedded_index: bool = False
    range_supported: bool | None = None
    direct_streaming_supported: bool = False
    trajectories: list[str] = field(default_factory=list)
    frame_count: int | None = None
    bead_count: int | None = None
    coordinate_paths: list[str] = field(default_factory=list)
    dtype: str | None = None
    compression: str | None = None
    chunks: Any | None = None
    file_size: int | None = None
    detected_hdf5: bool = False
    detected_text_ndb: bool = False
    detected_spacewalk: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)
