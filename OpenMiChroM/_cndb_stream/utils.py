"""Internal utility helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def coerce_frame_id(frame: int | str) -> str:
    """Normalize a public frame identifier to the JSON index key."""

    if isinstance(frame, np.integer):
        return str(int(frame))
    return str(frame)


def sort_frame_ids(frame_ids: list[str]) -> list[str]:
    """Sort frame identifiers numerically."""

    return sorted(frame_ids, key=frame_id_sort_key)


def frame_id_sort_key(value: str) -> tuple[int, str]:
    """Return a stable numeric sort key for ``1`` and ``t_1`` style frames."""

    text = str(value)
    if text.startswith("t_") and text[2:].isdigit():
        return int(text[2:]), text
    if text.isdigit():
        return int(text), text
    return 0, text


def sort_trajectory_names(names: list[str]) -> list[str]:
    """Sort replica/chromosome trajectory names by embedded numbers."""

    def key(value: str):
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r"(\d+)", str(value))
            if part
        )

    return sorted(names, key=key)


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
