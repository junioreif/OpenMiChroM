import numpy as np
import pytest

import OpenMiChroM
from OpenMiChroM.StructuralVariants import (
    StructuralVariantResult,
    add_ideal_chromosome,
    apply_structural_variant,
    delete_region,
    duplicate_region,
    ideal_chromosome_profile,
    invert_region,
    locus_labels,
    read_locus_matrix,
    remove_ideal_chromosome,
    write_locus_matrix,
    write_locus_sequence,
)


def _unique_symmetric_matrix(size=5):
    indices = np.arange(size)
    return (
        100 * np.minimum.outer(indices, indices)
        + np.maximum.outer(indices, indices)
    ).astype(float)


def _distance_matrix(profile, size=None, *, out_of_range=0.0):
    profile = np.asarray(profile, dtype=float)
    if size is None:
        size = profile.size
    indices = np.arange(size)
    separations = np.abs(indices[:, None] - indices[None, :])
    result = np.full((size, size), float(out_of_range))
    available = separations < profile.size
    result[available] = profile[separations[available]]
    return result


@pytest.mark.parametrize(
    ("kind", "start", "end", "expected_map", "wrapper"),
    [
        ("deletion", 1, 3, [0, 3, 4], delete_region),
        ("inversion", 1, 4, [0, 3, 2, 1, 4], invert_region),
        ("duplication", 1, 3, [0, 1, 2, 1, 2, 3, 4], duplicate_region),
    ],
)
def test_exact_matrix_transformations_and_public_index_maps(
    kind, start, end, expected_map, wrapper
):
    matrix = _unique_symmetric_matrix()
    expected_map = np.asarray(expected_map, dtype=np.int64)
    expected = matrix[np.ix_(expected_map, expected_map)]

    result = apply_structural_variant(matrix, kind, start, end)

    assert isinstance(result, StructuralVariantResult)
    assert result.kind == kind
    assert result.start == start
    assert result.end == end
    assert result.index_map.dtype == np.int64
    np.testing.assert_array_equal(result.index_map, expected_map)
    np.testing.assert_array_equal(result.matrix, expected)
    np.testing.assert_array_equal(wrapper(matrix, start, end), expected)
    assert result.forward_motifs is None
    assert result.reverse_motifs is None


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("del", "deletion"),
        ("DELETE", "deletion"),
        ("inv", "inversion"),
        (" invert ", "inversion"),
        ("dup", "duplication"),
        ("DUPLICATE", "duplication"),
    ],
)
def test_structural_variant_kind_aliases_are_canonicalized(alias, canonical):
    result = apply_structural_variant(_unique_symmetric_matrix(), alias, 1, 3)

    assert result.kind == canonical


@pytest.mark.parametrize(
    ("kind", "expected_forward", "expected_reverse"),
    [
        ("deletion", [10, 13, 14], [20, 23, 24]),
        ("inversion", [10, 23, 22, 21, 14], [20, 13, 12, 11, 24]),
        ("duplication", [10, 11, 12, 11, 12, 13, 14], [20, 21, 22, 21, 22, 23, 24]),
    ],
)
def test_directional_motif_transformations_are_exact(
    kind, expected_forward, expected_reverse
):
    matrix = _unique_symmetric_matrix()
    forward = np.arange(10, 15, dtype=float)
    reverse = np.arange(20, 25, dtype=float)
    end = 4 if kind == "inversion" else 3

    result = apply_structural_variant(
        matrix,
        kind,
        1,
        end,
        forward_motifs=forward,
        reverse_motifs=reverse,
    )

    np.testing.assert_array_equal(result.forward_motifs, expected_forward)
    np.testing.assert_array_equal(result.reverse_motifs, expected_reverse)


def test_transformations_do_not_mutate_matrix_or_motif_inputs():
    matrix = _unique_symmetric_matrix()
    forward = np.arange(5, dtype=float)
    reverse = np.arange(10, 15, dtype=float)
    original_matrix = matrix.copy()
    original_forward = forward.copy()
    original_reverse = reverse.copy()

    result = apply_structural_variant(
        matrix,
        "duplication",
        1,
        3,
        forward_motifs=forward,
        reverse_motifs=reverse,
    )
    result.matrix[0, 0] = -999
    result.forward_motifs[0] = -999

    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(forward, original_forward)
    np.testing.assert_array_equal(reverse, original_reverse)


def test_ideal_chromosome_profile_remove_and_add_are_exact():
    expected_profile = np.array([10.0, 3.0, -2.0, 1.0, 0.5])
    matrix = _distance_matrix(expected_profile)

    profile = ideal_chromosome_profile(matrix)
    residual, removed_profile = remove_ideal_chromosome(matrix)

    np.testing.assert_allclose(profile, expected_profile, rtol=0, atol=0)
    np.testing.assert_allclose(removed_profile, expected_profile, rtol=0, atol=0)
    np.testing.assert_allclose(residual, np.zeros_like(matrix), rtol=0, atol=0)
    np.testing.assert_allclose(
        add_ideal_chromosome(residual, removed_profile), matrix, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("kind", "start", "end", "expected_map"),
    [
        ("deletion", 1, 3, [0, 3, 4]),
        ("inversion", 1, 4, [0, 3, 2, 1, 4]),
        ("duplication", 1, 3, [0, 1, 2, 1, 2, 3, 4]),
    ],
)
def test_ideal_chromosome_adjustment_rebuilds_output_separation_profile(
    kind, start, end, expected_map
):
    profile = np.array([10.0, 3.0, -2.0, 1.0, 0.5])
    matrix = _distance_matrix(profile)
    expected_map = np.asarray(expected_map)

    unadjusted = apply_structural_variant(matrix, kind, start, end)
    adjusted = apply_structural_variant(
        matrix, kind, start, end, adjust_ideal_chromosome=True
    )

    np.testing.assert_array_equal(
        unadjusted.matrix, matrix[np.ix_(expected_map, expected_map)]
    )
    np.testing.assert_array_equal(
        adjusted.matrix, _distance_matrix(profile, expected_map.size)
    )


def test_add_ideal_chromosome_uses_explicit_out_of_range_value():
    profile = np.array([5.0, 2.0])
    matrix = np.zeros((4, 4))

    result = add_ideal_chromosome(matrix, profile, out_of_range=-7.0)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [5, 2, -7, -7],
                [2, 5, 2, -7],
                [-7, 2, 5, 2],
                [-7, -7, 2, 5],
            ],
            dtype=float,
        ),
    )


def test_ideal_duplication_zeros_every_contact_of_inserted_copy():
    matrix = _unique_symmetric_matrix()
    result = duplicate_region(
        matrix, 1, 3, adjust_ideal_chromosome=False, duplicate_contacts="ideal"
    )
    expected_map = np.array([0, 1, 2, 1, 2, 3, 4])
    expected = matrix[np.ix_(expected_map, expected_map)]
    expected[3:5, :] = 0.0
    expected[:, 3:5] = 0.0

    np.testing.assert_array_equal(result, expected)


def test_symmetric_nan_entries_are_ignored_by_profile_and_preserved():
    expected_profile = np.array([10.0, 3.0, -2.0, 1.0, 0.5])
    matrix = _distance_matrix(expected_profile)
    matrix[0, 1] = np.nan
    matrix[1, 0] = np.nan

    profile = ideal_chromosome_profile(matrix)
    residual, removed_profile = remove_ideal_chromosome(matrix)
    round_trip = add_ideal_chromosome(residual, removed_profile)
    transformed = apply_structural_variant(matrix, "inversion", 1, 4)
    index_map = np.array([0, 3, 2, 1, 4])

    np.testing.assert_allclose(profile, expected_profile, rtol=0, atol=0)
    np.testing.assert_allclose(round_trip, matrix, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        transformed.matrix,
        matrix[np.ix_(index_map, index_map)],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert np.isnan(transformed.matrix).sum() == 2


def test_profile_rejects_a_diagonal_with_no_finite_observation():
    matrix = _distance_matrix([10.0, 3.0, -2.0, 1.0, 0.5])
    matrix[0, -1] = np.nan
    matrix[-1, 0] = np.nan

    with pytest.raises(ValueError, match="separation 4"):
        ideal_chromosome_profile(matrix)


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (np.arange(4), ValueError, "two-dimensional"),
        (np.zeros((2, 3)), ValueError, "square"),
        (np.empty((0, 0)), ValueError, "must not be empty"),
        (np.array([["x"]]), TypeError, "numeric"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), ValueError, "symmetric"),
        (np.array([[0.0, np.inf], [np.inf, 0.0]]), ValueError, "infinite"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), ValueError, "symmetric"),
    ],
)
def test_invalid_matrices_are_rejected(matrix, error, message):
    with pytest.raises(error, match=message):
        apply_structural_variant(matrix, "inversion", 0, 1)


def test_complex_matrix_motif_and_profile_inputs_are_rejected():
    complex_matrix = np.array(
        [[1.0 + 2.0j, 0.0], [0.0, 1.0 + 2.0j]], dtype=complex
    )
    real_matrix = np.eye(2)

    with pytest.raises(TypeError):
        invert_region(complex_matrix, 0, 1)
    with pytest.raises(TypeError):
        apply_structural_variant(
            real_matrix,
            "inversion",
            0,
            1,
            forward_motifs=np.array([1.0 + 2.0j, 0.0]),
            reverse_motifs=np.zeros(2),
        )
    with pytest.raises(TypeError):
        add_ideal_chromosome(real_matrix, np.array([1.0 + 2.0j, 0.0]))


@pytest.mark.parametrize(
    ("start", "end", "error", "message"),
    [
        (True, 2, TypeError, "start must be an integer"),
        (1, 2.5, TypeError, "end must be an integer"),
        (-1, 2, ValueError, "outside"),
        (1, 6, ValueError, "outside"),
        (2, 2, ValueError, "smaller"),
        (3, 2, ValueError, "smaller"),
        (0, 5, ValueError, "entire matrix"),
    ],
)
def test_invalid_deletion_intervals_are_rejected(start, end, error, message):
    with pytest.raises(error, match=message):
        delete_region(_unique_symmetric_matrix(), start, end)


def test_numpy_integer_interval_is_accepted_and_normalized_to_python_int():
    result = apply_structural_variant(
        _unique_symmetric_matrix(), "deletion", np.int64(1), np.int32(3)
    )

    assert type(result.start) is int
    assert type(result.end) is int


def test_invalid_operation_options_are_rejected():
    matrix = _unique_symmetric_matrix()

    with pytest.raises(TypeError, match="kind must be a string"):
        apply_structural_variant(matrix, None, 1, 3)
    with pytest.raises(ValueError, match="unsupported structural variant"):
        apply_structural_variant(matrix, "translocation", 1, 3)
    with pytest.raises(TypeError, match="adjust_ideal_chromosome must be a boolean"):
        delete_region(matrix, 1, 3, adjust_ideal_chromosome="yes")
    with pytest.raises(ValueError, match="duplicate_contacts"):
        duplicate_region(matrix, 1, 3, duplicate_contacts="unknown")
    with pytest.raises(ValueError, match="duplicate_contacts"):
        apply_structural_variant(
            matrix, "deletion", 1, 3, duplicate_contacts="unknown"
        )


def test_invalid_ideal_chromosome_options_are_rejected():
    matrix = np.zeros((3, 3))

    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        add_ideal_chromosome(matrix, [])
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        add_ideal_chromosome(matrix, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="only finite"):
        add_ideal_chromosome(matrix, [1.0, np.nan])
    with pytest.raises(ValueError, match="out_of_range must be finite"):
        add_ideal_chromosome(matrix, [1.0], out_of_range=np.inf)


def test_motif_tracks_must_be_a_finite_equal_length_pair():
    matrix = _unique_symmetric_matrix()
    valid = np.arange(5, dtype=float)

    with pytest.raises(ValueError, match="provided together"):
        apply_structural_variant(
            matrix, "deletion", 1, 3, forward_motifs=valid
        )
    with pytest.raises(ValueError, match="same length"):
        apply_structural_variant(
            matrix,
            "deletion",
            1,
            3,
            forward_motifs=valid,
            reverse_motifs=np.arange(4),
        )
    with pytest.raises(ValueError, match="same length as the matrix"):
        apply_structural_variant(
            matrix,
            "deletion",
            1,
            3,
            forward_motifs=np.arange(4),
            reverse_motifs=np.arange(4),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_structural_variant(
            matrix,
            "deletion",
            1,
            3,
            forward_motifs=np.zeros((1, 5)),
            reverse_motifs=valid,
        )
    with pytest.raises(TypeError, match="numeric"):
        apply_structural_variant(
            matrix,
            "deletion",
            1,
            3,
            forward_motifs=["A"] * 5,
            reverse_motifs=valid,
        )
    with pytest.raises(ValueError, match="only finite"):
        apply_structural_variant(
            matrix,
            "deletion",
            1,
            3,
            forward_motifs=[0, 1, np.nan, 3, 4],
            reverse_motifs=valid,
        )


def test_locus_matrix_round_trip_is_exact_and_creates_parent_directories(tmp_path):
    matrix = _unique_symmetric_matrix(3)
    destination = tmp_path / "nested" / "lambdas.csv"

    result = write_locus_matrix(destination, matrix)
    observed, labels = read_locus_matrix(destination)

    assert result == destination.resolve()
    assert labels == ("Locus1", "Locus2", "Locus3")
    assert destination.read_text(encoding="utf-8").splitlines()[0] == (
        "Locus1,Locus2,Locus3"
    )
    np.testing.assert_allclose(observed, matrix, rtol=0, atol=0)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_locus_matrix_writer_rejects_nonfinite_epm_values(tmp_path, nonfinite):
    matrix = np.eye(2)
    matrix[0, 0] = nonfinite
    destination = tmp_path / "unsafe.csv"

    with pytest.raises(ValueError):
        write_locus_matrix(destination, matrix)

    assert not destination.exists()


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_locus_matrix_reader_rejects_nonfinite_epm_values(tmp_path, token):
    source = tmp_path / "unsafe.csv"
    source.write_text(
        f"Locus1,Locus2\n{token},0\n0,1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        read_locus_matrix(source)


def test_locus_matrix_and_sequence_use_the_same_explicit_labels(tmp_path):
    labels = ("Enhancer", "Boundary", "Promoter")
    matrix_path = tmp_path / "lambdas.csv"
    sequence_path = tmp_path / "loci.seq"

    write_locus_matrix(matrix_path, np.eye(3), labels=labels)
    write_locus_sequence(sequence_path, labels)

    observed, observed_labels = read_locus_matrix(matrix_path)
    np.testing.assert_array_equal(observed, np.eye(3))
    assert observed_labels == labels
    assert sequence_path.read_text(encoding="utf-8") == (
        "1 Enhancer\n2 Boundary\n3 Promoter\n"
    )


def test_locus_writers_require_explicit_overwrite_permission(tmp_path):
    matrix_path = tmp_path / "lambdas.csv"
    sequence_path = tmp_path / "loci.seq"
    write_locus_matrix(matrix_path, np.eye(2))
    write_locus_sequence(sequence_path, ("Locus1", "Locus2"))

    with pytest.raises(FileExistsError, match="already exists"):
        write_locus_matrix(matrix_path, np.zeros((2, 2)))
    with pytest.raises(FileExistsError, match="already exists"):
        write_locus_sequence(sequence_path, ("A", "B"))

    write_locus_matrix(matrix_path, np.zeros((2, 2)), overwrite=True)
    write_locus_sequence(sequence_path, ("A", "B"), overwrite=True)
    observed, labels = read_locus_matrix(matrix_path)
    np.testing.assert_array_equal(observed, np.zeros((2, 2)))
    assert labels == ("Locus1", "Locus2")
    assert sequence_path.read_text(encoding="utf-8") == "1 A\n2 B\n"


@pytest.mark.parametrize(
    ("size", "prefix", "error", "message"),
    [
        (True, "Locus", TypeError, "size must be an integer"),
        (2.5, "Locus", TypeError, "size must be an integer"),
        (0, "Locus", ValueError, "size must be at least 1"),
        (2, "", ValueError, "prefix must be a non-empty string"),
        (2, "   ", ValueError, "prefix must be a non-empty string"),
        (2, None, ValueError, "prefix must be a non-empty string"),
        (2, "Bad Prefix", ValueError, "whitespace or control"),
        (2, "Bad\tPrefix", ValueError, "whitespace or control"),
        (2, "Bad\nPrefix", ValueError, "whitespace or control"),
        (2, " Leading", ValueError, "whitespace or control"),
        (2, "Trailing ", ValueError, "whitespace or control"),
    ],
)
def test_locus_label_options_are_validated(size, prefix, error, message):
    with pytest.raises(error, match=message):
        locus_labels(size, prefix=prefix)


def test_locus_labels_are_one_based_and_accept_numpy_integer_size():
    assert locus_labels(np.int64(3), prefix="Bin") == ("Bin1", "Bin2", "Bin3")


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (("A", "B"), "exactly 3 entries"),
        (("A", "", "C"), "must not contain empty"),
        (("A", "A", "C"), "must be unique"),
        (np.array([["A", "B", "C"]]), "exactly 3 entries"),
    ],
)
def test_locus_matrix_labels_are_validated(tmp_path, labels, message):
    with pytest.raises(ValueError, match=message):
        write_locus_matrix(tmp_path / "matrix.csv", np.eye(3), labels=labels)


def test_locus_sequence_labels_must_be_one_dimensional_and_nonempty(tmp_path):
    with pytest.raises(ValueError, match="one-dimensional"):
        write_locus_sequence(tmp_path / "matrix.seq", [["A", "B"]])
    with pytest.raises(ValueError, match="must not contain empty"):
        write_locus_sequence(tmp_path / "matrix.seq", [])


@pytest.mark.parametrize(
    "bad_label",
    ["Bad Label", "Bad\tLabel", "Bad\nLabel", " Leading", "Trailing "],
)
def test_locus_matrix_and_sequence_writers_reject_unsafe_labels(
    tmp_path, bad_label
):
    labels = ("Safe", bad_label)

    with pytest.raises(ValueError, match="whitespace or control"):
        write_locus_matrix(tmp_path / "matrix.csv", np.eye(2), labels=labels)
    with pytest.raises(ValueError, match="whitespace or control"):
        write_locus_sequence(tmp_path / "matrix.seq", labels)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("Locus1,Locus2\n", "label header and numeric rows"),
        ("Locus1,Locus2\n1,0\n0\n", "same width"),
        ("Locus1,Locus2\n1,0\n", "one numeric row per label"),
        ("Locus1,Locus2\n1,x\n0,1\n", "non-numeric"),
        ("Locus1,Locus2\n1,2\n3,1\n", "symmetric"),
        ("Locus1,Locus1\n1,0\n0,1\n", "labels must be unique"),
        ("Locus1,\n1,0\n0,1\n", "labels must not contain empty"),
    ],
)
def test_malformed_locus_matrix_files_are_rejected(tmp_path, content, message):
    source = tmp_path / "bad.csv"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_locus_matrix(source)


def test_missing_locus_matrix_is_reported_as_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="matrix file does not exist"):
        read_locus_matrix(tmp_path / "missing.csv")


def test_structural_variant_api_is_exported_from_package_root():
    assert OpenMiChroM.StructuralVariantResult is StructuralVariantResult
    assert OpenMiChroM.apply_structural_variant is apply_structural_variant
    assert OpenMiChroM.delete_region is delete_region
    assert OpenMiChroM.invert_region is invert_region
    assert OpenMiChroM.duplicate_region is duplicate_region
    assert OpenMiChroM.ideal_chromosome_profile is ideal_chromosome_profile
    assert OpenMiChroM.remove_ideal_chromosome is remove_ideal_chromosome
    assert OpenMiChroM.add_ideal_chromosome is add_ideal_chromosome
    assert OpenMiChroM.locus_labels is locus_labels
    assert OpenMiChroM.read_locus_matrix is read_locus_matrix
    assert OpenMiChroM.write_locus_matrix is write_locus_matrix
    assert OpenMiChroM.write_locus_sequence is write_locus_sequence
