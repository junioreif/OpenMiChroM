# Copyright (c) 2020-2026 The Center for Theoretical Biological Physics (CTBP)
# Rice University
# This file is part of OpenMiChroM and is released under the MIT License.

"""Matrix and motif transformations for chromosome structural variants.

The public API in this module grew from the structural-variation workflow
contributed by Miles Gantcher in OpenMiChroM pull request 123.  The
implementation here separates reusable, tested transformations from the
tutorial interface and makes the half-open interval convention explicit.

The operations transform a symmetric contact or effective-potential matrix and,
optionally, paired forward/reverse motif tracks.  Coordinates use Python's
standard half-open convention: ``start`` is included and ``end`` is excluded.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


StructuralVariantKind = Literal["deletion", "inversion", "duplication"]
DuplicateContacts = Literal["copy", "ideal"]


@dataclass(frozen=True)
class StructuralVariantResult:
    """Result of a structural-variant matrix and motif transformation.

    Attributes
    ----------
    matrix
        Transformed symmetric matrix.
    index_map
        For every output row/column, the corresponding input index.  A tandem
        duplication therefore contains the duplicated indices twice.
    kind
        Canonical operation name: ``deletion``, ``inversion``, or
        ``duplication``.
    start, end
        Half-open input interval used for the operation.
    forward_motifs, reverse_motifs
        Transformed directional motif tracks, or ``None`` when motif tracks
        were not supplied.
    """

    matrix: NDArray[np.float64]
    index_map: NDArray[np.int64]
    kind: StructuralVariantKind
    start: int
    end: int
    forward_motifs: NDArray[np.float64] | None = None
    reverse_motifs: NDArray[np.float64] | None = None


_KIND_ALIASES: dict[str, StructuralVariantKind] = {
    "del": "deletion",
    "delete": "deletion",
    "deletion": "deletion",
    "inv": "inversion",
    "invert": "inversion",
    "inversion": "inversion",
    "dup": "duplication",
    "duplicate": "duplication",
    "duplication": "duplication",
}


def _as_symmetric_matrix(matrix: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if array.shape[0] == 0:
        raise ValueError("matrix must not be empty")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError("matrix must contain numeric values")

    result = np.array(array, dtype=float, copy=True)
    if np.isinf(result).any():
        raise ValueError("matrix must not contain infinite values")
    if not np.allclose(
        result, result.T, rtol=1e-10, atol=1e-12, equal_nan=True
    ):
        raise ValueError("matrix must be symmetric")
    # Normalize accepted round-off differences so every public result satisfies
    # the advertised symmetry invariant exactly.
    return 0.5 * (result + result.T)


def _as_motif_track(track: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(track)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain numeric values")
    result = np.array(array, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_interval(size: int, start: int, end: int, *, deletion: bool) -> None:
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise TypeError("start must be an integer")
    if isinstance(end, bool) or not isinstance(end, (int, np.integer)):
        raise TypeError("end must be an integer")
    if start < 0 or end > size:
        raise ValueError(f"interval [{start}, {end}) is outside a matrix of size {size}")
    if start >= end:
        raise ValueError("start must be smaller than end")
    if deletion and end - start == size:
        raise ValueError("deletion cannot remove the entire matrix")


def _canonical_kind(kind: str) -> StructuralVariantKind:
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    try:
        return _KIND_ALIASES[kind.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_KIND_ALIASES.values())))
        raise ValueError(f"unsupported structural variant {kind!r}; choose {choices}") from exc


def ideal_chromosome_profile(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return the mean value at each genomic separation.

    NaN entries are ignored.  Every diagonal must contain at least one finite
    value; infinite values and asymmetric matrices are rejected.
    """

    array = _as_symmetric_matrix(matrix)
    profile = np.empty(array.shape[0], dtype=float)
    for separation in range(array.shape[0]):
        diagonal = np.diag(array, k=separation)
        finite = diagonal[np.isfinite(diagonal)]
        if finite.size == 0:
            raise ValueError(
                f"matrix diagonal at separation {separation} has no finite values"
            )
        profile[separation] = float(finite.mean())
    return profile


def add_ideal_chromosome(
    matrix: ArrayLike,
    profile: ArrayLike,
    *,
    out_of_range: float = 0.0,
) -> NDArray[np.float64]:
    """Add a genomic-separation profile to a symmetric matrix.

    ``out_of_range`` is used when a duplication creates separations longer than
    the original profile.  The default of zero is explicit and matches the
    absence of an inferred long-range contribution.
    """

    array = _as_symmetric_matrix(matrix)
    raw_curve = np.asarray(profile)
    if np.issubdtype(raw_curve.dtype, np.complexfloating):
        raise TypeError("profile must contain real numeric values")
    curve = np.asarray(raw_curve, dtype=float)
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError("profile must be a non-empty one-dimensional array")
    if not np.isfinite(curve).all():
        raise ValueError("profile must contain only finite values")
    if not np.isfinite(out_of_range):
        raise ValueError("out_of_range must be finite")

    indices = np.arange(array.shape[0])
    separations = np.abs(indices[:, None] - indices[None, :])
    background = np.full(array.shape, float(out_of_range), dtype=float)
    available = separations < curve.size
    background[available] = curve[separations[available]]
    return array + background


def remove_ideal_chromosome(
    matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Subtract the mean genomic-separation profile from ``matrix``."""

    array = _as_symmetric_matrix(matrix)
    profile = ideal_chromosome_profile(array)
    background = add_ideal_chromosome(np.zeros_like(array), profile)
    return array - background, profile


def _index_map(
    size: int,
    kind: StructuralVariantKind,
    start: int,
    end: int,
) -> NDArray[np.int64]:
    if kind == "deletion":
        return np.concatenate((np.arange(start), np.arange(end, size))).astype(np.int64)
    if kind == "inversion":
        return np.concatenate(
            (np.arange(start), np.arange(end - 1, start - 1, -1), np.arange(end, size))
        ).astype(np.int64)
    return np.concatenate(
        (np.arange(end), np.arange(start, end), np.arange(end, size))
    ).astype(np.int64)


def _transform_matrix(
    matrix: NDArray[np.float64],
    index_map: NDArray[np.int64],
    *,
    adjust_ideal_chromosome: bool,
    duplicate_interval: tuple[int, int] | None,
    duplicate_contacts: DuplicateContacts,
) -> NDArray[np.float64]:
    if not isinstance(adjust_ideal_chromosome, (bool, np.bool_)):
        raise TypeError("adjust_ideal_chromosome must be a boolean")
    if duplicate_contacts not in {"copy", "ideal"}:
        raise ValueError("duplicate_contacts must be 'copy' or 'ideal'")

    if adjust_ideal_chromosome:
        working, profile = remove_ideal_chromosome(matrix)
    else:
        working = matrix.copy()
        profile = None

    transformed = working[np.ix_(index_map, index_map)]
    if duplicate_interval is not None and duplicate_contacts == "ideal":
        inserted_start, inserted_end = duplicate_interval
        transformed[inserted_start:inserted_end, :] = 0.0
        transformed[:, inserted_start:inserted_end] = 0.0

    if profile is not None:
        transformed = add_ideal_chromosome(transformed, profile)
    return transformed


def delete_region(
    matrix: ArrayLike,
    start: int,
    end: int,
    *,
    adjust_ideal_chromosome: bool = False,
) -> NDArray[np.float64]:
    """Delete matrix rows and columns in the half-open interval ``[start, end)``."""

    array = _as_symmetric_matrix(matrix)
    _validate_interval(array.shape[0], start, end, deletion=True)
    mapping = _index_map(array.shape[0], "deletion", start, end)
    return _transform_matrix(
        array,
        mapping,
        adjust_ideal_chromosome=adjust_ideal_chromosome,
        duplicate_interval=None,
        duplicate_contacts="copy",
    )


def invert_region(
    matrix: ArrayLike,
    start: int,
    end: int,
    *,
    adjust_ideal_chromosome: bool = False,
) -> NDArray[np.float64]:
    """Reverse matrix rows and columns in ``[start, end)``."""

    array = _as_symmetric_matrix(matrix)
    _validate_interval(array.shape[0], start, end, deletion=False)
    mapping = _index_map(array.shape[0], "inversion", start, end)
    return _transform_matrix(
        array,
        mapping,
        adjust_ideal_chromosome=adjust_ideal_chromosome,
        duplicate_interval=None,
        duplicate_contacts="copy",
    )


def duplicate_region(
    matrix: ArrayLike,
    start: int,
    end: int,
    *,
    adjust_ideal_chromosome: bool = False,
    duplicate_contacts: DuplicateContacts = "copy",
) -> NDArray[np.float64]:
    """Insert a tandem copy of ``[start, end)`` immediately after the interval.

    With ``duplicate_contacts='copy'``, every contact is copied from its source
    index.  With ``'ideal'``, contacts involving the inserted copy contain only
    the genomic-separation background when ``adjust_ideal_chromosome=True`` (or
    zero when it is false).  The latter is useful when those contacts are meant
    to be generated by a subsequent optimization rather than copied.
    """

    array = _as_symmetric_matrix(matrix)
    _validate_interval(array.shape[0], start, end, deletion=False)
    mapping = _index_map(array.shape[0], "duplication", start, end)
    width = end - start
    return _transform_matrix(
        array,
        mapping,
        adjust_ideal_chromosome=adjust_ideal_chromosome,
        duplicate_interval=(end, end + width),
        duplicate_contacts=duplicate_contacts,
    )


def _transform_motifs(
    forward: NDArray[np.float64],
    reverse: NDArray[np.float64],
    kind: StructuralVariantKind,
    start: int,
    end: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if kind == "deletion":
        return (
            np.concatenate((forward[:start], forward[end:])),
            np.concatenate((reverse[:start], reverse[end:])),
        )
    if kind == "inversion":
        return (
            np.concatenate((forward[:start], reverse[start:end][::-1], forward[end:])),
            np.concatenate((reverse[:start], forward[start:end][::-1], reverse[end:])),
        )
    return (
        np.concatenate((forward[:end], forward[start:end], forward[end:])),
        np.concatenate((reverse[:end], reverse[start:end], reverse[end:])),
    )


def apply_structural_variant(
    matrix: ArrayLike,
    kind: str,
    start: int,
    end: int,
    *,
    forward_motifs: ArrayLike | None = None,
    reverse_motifs: ArrayLike | None = None,
    adjust_ideal_chromosome: bool = False,
    duplicate_contacts: DuplicateContacts = "copy",
) -> StructuralVariantResult:
    """Apply one structural variant to a matrix and optional motif tracks.

    Directional motif tracks must be provided as a pair.  During an inversion,
    the interval is reversed and the two directions are exchanged.
    """

    array = _as_symmetric_matrix(matrix)
    canonical = _canonical_kind(kind)
    _validate_interval(
        array.shape[0], start, end, deletion=canonical == "deletion"
    )
    mapping = _index_map(array.shape[0], canonical, start, end)

    duplicate_interval = None
    if canonical == "duplication":
        duplicate_interval = (end, end + end - start)
    transformed = _transform_matrix(
        array,
        mapping,
        adjust_ideal_chromosome=adjust_ideal_chromosome,
        duplicate_interval=duplicate_interval,
        duplicate_contacts=duplicate_contacts,
    )

    if (forward_motifs is None) != (reverse_motifs is None):
        raise ValueError("forward_motifs and reverse_motifs must be provided together")

    transformed_forward = transformed_reverse = None
    if forward_motifs is not None and reverse_motifs is not None:
        forward = _as_motif_track(forward_motifs, name="forward_motifs")
        reverse = _as_motif_track(reverse_motifs, name="reverse_motifs")
        if forward.shape != reverse.shape:
            raise ValueError("forward_motifs and reverse_motifs must have the same length")
        if forward.size != array.shape[0]:
            raise ValueError("motif tracks must have the same length as the matrix")
        transformed_forward, transformed_reverse = _transform_motifs(
            forward, reverse, canonical, start, end
        )

    return StructuralVariantResult(
        matrix=transformed,
        index_map=mapping,
        kind=canonical,
        start=int(start),
        end=int(end),
        forward_motifs=transformed_forward,
        reverse_motifs=transformed_reverse,
    )


def locus_labels(size: int, *, prefix: str = "Locus") -> tuple[str, ...]:
    """Return unique one-based labels for a locus-specific potential matrix."""

    count = _validate_label_count(size)
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    if not prefix.isprintable() or any(character.isspace() for character in prefix):
        raise ValueError("prefix must not contain whitespace or control characters")
    return tuple(f"{prefix}{index}" for index in range(1, count + 1))


def _validate_label_count(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise TypeError("size must be an integer")
    count = int(size)
    if count < 1:
        raise ValueError("size must be at least 1")
    return count


def _validated_labels(labels: ArrayLike, size: int) -> tuple[str, ...]:
    array = np.asarray(labels, dtype=object)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"labels must contain exactly {size} entries")
    if size == 0:
        raise ValueError("labels must not contain empty input")
    result = tuple(str(value) for value in array)
    if any(not value for value in result):
        raise ValueError("labels must not contain empty values")
    if any(
        not value.isprintable()
        or any(character.isspace() for character in value)
        for value in result
    ):
        raise ValueError("labels must not contain whitespace or control characters")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return result


def _atomic_text_write(path: Path, text: str, *, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise
    return path


def read_locus_matrix(
    path: str | os.PathLike[str],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Read the header-plus-dense-rows format consumed by ``addCustomTypes``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"matrix file does not exist: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError("locus matrix must contain a label header and numeric rows")
    labels = tuple(rows[0])
    if any(len(row) != len(labels) for row in rows[1:]):
        raise ValueError("every numeric row must have the same width as the label header")
    if len(rows) - 1 != len(labels):
        raise ValueError("locus matrix must have one numeric row per label")
    try:
        matrix = np.asarray(rows[1:], dtype=float)
    except ValueError as exc:
        raise ValueError("locus matrix contains a non-numeric value") from exc
    matrix = _as_symmetric_matrix(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("locus matrix must contain only finite values")
    return matrix, _validated_labels(labels, matrix.shape[0])


def write_locus_matrix(
    path: str | os.PathLike[str],
    matrix: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a transformed matrix in the format consumed by ``addCustomTypes``."""

    array = _as_symmetric_matrix(matrix)
    if not np.isfinite(array).all():
        raise ValueError("locus matrix must contain only finite values")
    output_labels = (
        locus_labels(array.shape[0])
        if labels is None
        else _validated_labels(labels, array.shape[0])
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(output_labels)
    writer.writerows(
        [[format(float(value), ".17g") for value in row] for row in array]
    )
    return _atomic_text_write(Path(path), stream.getvalue(), overwrite=overwrite)


def write_locus_sequence(
    path: str | os.PathLike[str],
    labels: ArrayLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a two-column sequence whose types match a locus-matrix header."""

    array = np.asarray(labels, dtype=object)
    if array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    output_labels = _validated_labels(array, array.size)
    text = "".join(
        f"{index} {label}\n" for index, label in enumerate(output_labels, start=1)
    )
    return _atomic_text_write(Path(path), text, overwrite=overwrite)


__all__ = [
    "StructuralVariantResult",
    "add_ideal_chromosome",
    "apply_structural_variant",
    "delete_region",
    "duplicate_region",
    "ideal_chromosome_profile",
    "invert_region",
    "locus_labels",
    "read_locus_matrix",
    "remove_ideal_chromosome",
    "write_locus_matrix",
    "write_locus_sequence",
]
