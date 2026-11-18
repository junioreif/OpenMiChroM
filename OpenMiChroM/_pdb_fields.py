"""Validate numeric fields shared by the fixed-width PDB writers."""

import numpy as np


def validate_pdb_fields(frame_ids, frames, residue_indices, max_serial):
    if max_serial > 99999:
        raise ValueError("PDB atom/TER serial numbers cannot exceed 99999.")
    if any(len(f"{int(index):4d}") > 4 for index in residue_indices):
        raise ValueError("PDB residue sequence numbers must fit four columns.")
    if any(len(f"{int(index):4d}") > 4 for index in frame_ids):
        raise ValueError("PDB MODEL identifiers must fit four columns.")
    for coordinates in frames:
        values = np.asarray(coordinates)
        # Check the actual rounding at the edges of the 8.3 coordinate fields.
        if not np.isfinite(values).all() or any(
            len(f"{extreme:8.3f}") > 8 for extreme in (values.min(), values.max())
        ):
            raise ValueError("PDB coordinates must fit an 8.3 field.")
