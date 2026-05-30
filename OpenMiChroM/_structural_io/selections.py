"""Selection helpers for structural coordinate reads."""

from __future__ import annotations

from collections.abc import Iterable


def coalesce_indices(indices: Iterable[int], max_gap: int = 0) -> list[tuple[int, int]]:
    """Convert integer indices into sorted half-open ranges.

    ``max_gap`` controls how many unrequested rows may be included between two
    requested index blocks before they are merged into one range.
    """

    values = sorted({int(index) for index in indices})
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        gap = value - previous - 1
        if gap <= max_gap:
            previous = value
            continue
        ranges.append((start, previous + 1))
        start = value
        previous = value
    ranges.append((start, previous + 1))
    return ranges
