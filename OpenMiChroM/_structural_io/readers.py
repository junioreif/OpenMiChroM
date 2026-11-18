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
        self.chain_ids: list[str] = []
        self.chain_indices: list[int] = []
        self.sigma: list[float] = []
        self.loops: list[tuple[int, int]] = []
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
        bead_metadata: list[tuple[str, int, float]] = []

        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = line[:6].strip()
                if record == "MODEL":
                    if current is not None and current_frame is not None:
                        self._commit_frame(current_frame, current, current_types, current_genomic, bead_metadata)
                    parts = line.split()
                    current_frame = parts[1] if len(parts) > 1 else str(len(self._frames) + 1)
                    current = []
                    current_types = []
                    current_genomic = []
                    bead_metadata = []
                elif record == "CHROM":
                    if current is None:
                        current_frame = str(len(self._frames) + 1)
                        current = []
                    parsed = _parse_chrom_line(line)
                    current.append([parsed["x"], parsed["y"], parsed["z"]])
                    current_types.append(parsed["type"])
                    current_genomic.append((parsed["start"], parsed["end"]))
                    fields = line.split()
                    split_chain = len(fields) >= 12 and len(fields[3]) == 1
                    chain = fields[3] + fields[4] if split_chain else fields[3]
                    index_pos = 5 if split_chain else 4
                    sigma_pos = index_pos + 6
                    bead_metadata.append((chain, int(fields[index_pos]), float(fields[sigma_pos])
                                          if len(fields) > sigma_pos else 0.0))
                elif record == "LOOPS":
                    first, second = line.split()[1:3]
                    self.loops.append((int(first), int(second)))
                elif record == "ENDMDL":
                    if current is not None and current_frame is not None:
                        self._commit_frame(current_frame, current, current_types, current_genomic, bead_metadata)
                    current = None
                    current_frame = None
                    current_types = []
                    current_genomic = []
                    bead_metadata = []

        if current is not None and current_frame is not None:
            self._commit_frame(current_frame, current, current_types, current_genomic, bead_metadata)
        if not self._frames:
            raise ValueError(f"No coordinate models found in {self.path}.")

    def _commit_frame(
        self,
        frame_id: str,
        coords: list[list[float]],
        types: list[str],
        genomic_positions: list[tuple[int, int]],
        bead_metadata: list[tuple[str, int, float]],
    ) -> None:
        key = str(int(frame_id))
        if key in self._frames or not coords:
            raise ValueError(f"Empty or duplicate NDB MODEL {key} in {self.path}.")
        if not self._frames:
            self._types = list(types)
            self._genomic_positions = list(genomic_positions)
            self.chain_ids, self.chain_indices, self.sigma = map(list, zip(*bead_metadata))
        elif (types != self._types or genomic_positions != self._genomic_positions
              or bead_metadata != list(zip(self.chain_ids, self.chain_indices, self.sigma))):
            raise ValueError(f"Inconsistent bead metadata in NDB MODEL {key} of {self.path}.")
        self._frames[key] = np.array(coords, dtype=np.float32)


def _parse_chrom_line(line: str) -> dict[str, Any]:
    parts = line.split()
    if len(parts) >= 10:
        # Two common NDB variants are used in OpenMiChroM examples:
        # CHROM bead type chain bead x y z start end score
        # CHROM bead type chain chain_id bead x y z start end score
        coord_start = 6 if len(parts) >= 12 and len(parts[3]) == 1 else 5
        try:
            return {
                "type": parts[2] if len(parts) > 2 else "UN",
                "x": float(parts[coord_start]),
                "y": float(parts[coord_start + 1]),
                "z": float(parts[coord_start + 2]),
                "start": int(float(parts[coord_start + 3])),
                "end": int(float(parts[coord_start + 4])),
            }
        except (ValueError, IndexError):
            pass

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
        return {
            "type": parts[2] if len(parts) > 2 else "UN",
            "x": float(parts[5]),
            "y": float(parts[6]),
            "z": float(parts[7]),
            "start": int(float(parts[8])) if len(parts) > 8 else 0,
            "end": int(float(parts[9])) if len(parts) > 9 else 0,
        }
