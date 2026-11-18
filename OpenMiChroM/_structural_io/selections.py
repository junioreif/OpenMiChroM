"""Selection helpers for structural coordinate reads."""

from __future__ import annotations

from collections.abc import Iterable


def coalesce_indices(
    indices: Iterable[int],
    max_gap: int = 0,
    max_ranges: int | None = None,
) -> list[tuple[int, int]]:
    """Convert integer indices into sorted half-open ranges.

    ``max_gap`` controls how many unrequested rows may be included between two
    requested index blocks before they are merged into one range. If
    ``max_ranges`` is provided, the nearest neighboring ranges are merged until
    at most that many ranges remain.
    """

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")
    if max_ranges is not None and max_ranges < 1:
        raise ValueError("max_ranges must be at least 1 when provided.")

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
    if max_ranges is not None:
        ranges = _merge_to_max_ranges(ranges, max_ranges)
    return ranges


def _merge_to_max_ranges(
    ranges: list[tuple[int, int]],
    max_ranges: int,
) -> list[tuple[int, int]]:
    """Merge neighboring ranges with the smallest gaps until count is bounded."""

    merged = list(ranges)
    while len(merged) > max_ranges:
        gaps = [
            (merged[index + 1][0] - merged[index][1], index)
            for index in range(len(merged) - 1)
        ]
        _, merge_index = min(gaps, key=lambda item: (item[0], item[1]))
        start = merged[merge_index][0]
        stop = merged[merge_index + 1][1]
        merged[merge_index : merge_index + 2] = [(start, stop)]
    return merged
