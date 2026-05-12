"""Internal utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def coerce_frame_id(frame: int | str) -> str:
    """Normalize a public frame identifier to the JSON index key."""

    if isinstance(frame, np.integer):
        return str(int(frame))
    return str(frame)


def sort_frame_ids(frame_ids: list[str]) -> list[str]:
    """Sort frame identifiers numerically."""

    return sorted(frame_ids, key=lambda value: int(value))


def json_safe_value(value: Any) -> Any:
    """Convert small h5py/numpy values into JSON-safe Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe_value(item) for item in value.tolist()]
    return value
