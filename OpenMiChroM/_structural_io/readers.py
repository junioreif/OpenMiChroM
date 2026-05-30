"""Small structural file readers used by CNDBTools routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NDBTextReader:
    """In-memory reader for local text NDB files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._frames: dict[str, np.ndarray] = {}
        self._types: list[str] = []
        self._genomic_positions: list[tuple[int, int]] = []
        self._parse()

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def n_beads(self) -> int:
        return len(self._types)

    @property
    def frame_ids(self) -> list[str]:
        return sorted(self._frames, key=lambda value: int(value))

    @property
    def trajectories(self) -> list[str]:
        return []

    @property
    def current_trajectory(self) -> None:
        return None

    @property
    def types(self) -> list[str]:
        return list(self._types)

    @property
    def genomic_positions(self) -> np.ndarray | None:
        if not self._genomic_positions:
            return None
        return np.array(self._genomic_positions, dtype=np.int64)

    @property
    def data_bytes_read(self) -> int:
        return 0

    @property
    def index_bytes_read(self) -> int:
        return 0

    @property
    def metadata_bytes_read(self) -> int:
        return 0

    @property
    def bytes_read(self) -> int:
        return 0

    @property
    def index_cache_hit(self) -> bool:
        return False

    def get_coordinates(self, frame: int | str, start: int | None = None, stop: int | None = None):
        frame_id = str(int(frame))
        coords = self._frames[frame_id]
        start_row = 0 if start is None else int(start)
        stop_row = coords.shape[0] if stop is None else int(stop)
        return coords[start_row:stop_row].copy()

    def stats(self) -> dict[str, int | bool]:
        return {
            "index_bytes_read": 0,
            "metadata_bytes_read": 0,
            "data_bytes_read": 0,
            "bytes_read": 0,
            "index_cache_hit": False,
        }

    def _parse(self) -> None:
        current: list[list[float]] | None = None
        current_types: list[str] = []
        current_genomic: list[tuple[int, int]] = []
        current_frame: str | None = None

        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = line[:6].strip()
                if record == "MODEL":
                    if current is not None and current_frame is not None:
                        self._commit_frame(current_frame, current, current_types, current_genomic)
                    parts = line.split()
                    current_frame = parts[1] if len(parts) > 1 else str(len(self._frames) + 1)
                    current = []
                    current_types = []
                    current_genomic = []
                elif record == "CHROM":
                    if current is None:
                        current_frame = str(len(self._frames) + 1)
                        current = []
                    parsed = _parse_chrom_line(line)
                    current.append([parsed["x"], parsed["y"], parsed["z"]])
                    current_types.append(parsed["type"])
                    current_genomic.append((parsed["start"], parsed["end"]))
                elif record == "ENDMDL":
                    if current is not None and current_frame is not None:
                        self._commit_frame(current_frame, current, current_types, current_genomic)
                    current = None
                    current_frame = None
                    current_types = []
                    current_genomic = []

        if current is not None and current_frame is not None:
            self._commit_frame(current_frame, current, current_types, current_genomic)

    def _commit_frame(
        self,
        frame_id: str,
        coords: list[list[float]],
        types: list[str],
        genomic_positions: list[tuple[int, int]],
    ) -> None:
        self._frames[str(int(frame_id))] = np.array(coords, dtype=np.float32)
        if not self._types:
            self._types = list(types)
        if not self._genomic_positions:
            self._genomic_positions = list(genomic_positions)


def _parse_chrom_line(line: str) -> dict[str, Any]:
    try:
        return {
            "type": line[16:18].strip() or "UN",
            "x": float(line[40:48]),
            "y": float(line[49:57]),
            "z": float(line[58:66]),
            "start": int(line[67:77]),
            "end": int(line[78:88]),
        }
    except (ValueError, IndexError):
        parts = line.split()
        return {
            "type": parts[2] if len(parts) > 2 else "UN",
            "x": float(parts[5]),
            "y": float(parts[6]),
            "z": float(parts[7]),
            "start": int(float(parts[8])) if len(parts) > 8 else 0,
            "end": int(float(parts[9])) if len(parts) > 9 else 0,
        }
