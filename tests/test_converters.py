import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from openmm import unit

from OpenMiChroM.Converters import (
    ConverterWarning,
    cndb_to_ndb,
    convert,
    csv_to_ndb,
    gro_to_ndb,
    ndb_to_cndb,
    ndb_to_pdb,
    ndb_to_spw,
    pdb_to_ndb,
    spw_to_ndb,
)
from OpenMiChroM.CndbTools import CndbTools
from OpenMiChroM.CustomReporter import SaveStructure


FRAME_IDS = (0, 7)
TYPES = ("A1", "A2", "B4", "NA")
CHAINS = ("C1", "C1", "C2", "C2")
CHAIN_INDEXES = (1, 2, 1, 2)
GENOMIC_INTERVALS = ((1, 50_000), (50_001, 100_000), (1, 50_000), (50_001, 100_000))
COORDINATES = (
    np.array(
        [
            [1.125, 2.250, 3.375],
            [-4.500, 5.625, 6.750],
            [7.875, -8.000, 9.125],
            [10.250, 11.375, -12.500],
        ],
        dtype=float,
    ),
    np.array(
        [
            [1.625, 2.750, 3.875],
            [-4.000, 6.125, 7.250],
            [8.375, -7.500, 9.625],
            [10.750, 11.875, -12.000],
        ],
        dtype=float,
    ),
)
LOOPS = ((1, 4), (2, 3))
NUMERIC_TYPE_MAP = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "B3", 5: "B4", 6: "NA"}
ALL_TYPES = tuple(NUMERIC_TYPE_MAP.values())


def _write_ndb(path, *, frame_ids=FRAME_IDS, types=TYPES, coordinates=COORDINATES):
    lines = [
        "HEADER    deterministic converter test fixture",
        "ASMBLY    hg38",
        "SEQCHR   1 C1     2  A1 A2",
        "SEQCHR   1 C2     2  B4 NA",
    ]
    for frame_id, frame in zip(frame_ids, coordinates):
        lines.append(f"MODEL     {frame_id:4d}")
        previous_chain = None
        for serial, (chrom_type, chain, chain_index, interval, xyz) in enumerate(
            zip(types, CHAINS, CHAIN_INDEXES, GENOMIC_INTERVALS, frame), start=1
        ):
            if previous_chain is not None and chain != previous_chain:
                lines.append(f"TER    {serial:8d} {types[serial - 2]:2s}        {previous_chain:4s}")
            start, end = interval
            x, y, z = xyz
            lines.append(
                f"CHROM  {serial:8d} {chrom_type:2s}        {chain:4s} {chain_index:8d} "
                f"{x:8.3f} {y:8.3f} {z:8.3f} {start:10d} {end:10d} {0.0:8.3f}"
            )
            previous_chain = chain
        lines.append(f"TER    {len(types) + 1:8d} {types[-1]:2s}        {CHAINS[-1]:4s}")
        lines.append("ENDMDL")
    lines.extend(
        [
            *(f"LOOPS  {left:8d} {right:8d}" for left, right in LOOPS),
            f"MASTER {len(types):8d} {len(set(CHAINS)):6d} {len(LOOPS):6d} {50_000:10d}",
            "END",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_openmichrom_end_ndb(path):
    """Write the single-model END form emitted by MiChroM.saveStructure()."""

    lines = ["HEADER    OpenMiChroM END fixture", "MODEL 1"]
    for serial, (chrom_type, xyz) in enumerate(zip(("A1", "UN", "NA", "B4"), COORDINATES[0]), 1):
        x, y, z = xyz
        lines.append(
            f"CHROM  {serial:8d} {chrom_type:2s}        C1   {serial:8d} "
            f"{x:8.3f} {y:8.3f} {z:8.3f} {(serial - 1) * 50_000 + 1:10d} "
            f"{serial * 50_000:10d} {0.0:8.3f}"
        )
    # Existing OpenMiChroM files may place loop records after END.
    lines.extend(["END", "LOOPS         1        4"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _decoded_types(dataset):
    result = []
    for value in dataset[()]:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, np.integer)):
            value = NUMERIC_TYPE_MAP[int(value)]
        value = str(value).strip().upper()
        result.append("NA" if value == "UN" else value)
    return result


def _numeric_frame_ids(h5_file):
    return sorted((int(key) for key in h5_file.keys() if str(key).lstrip("-").isdigit()))


def _assert_cndb(
    path,
    *,
    expected_ids=FRAME_IDS,
    expected_types=TYPES,
    expected_coordinates=COORDINATES,
    expected_loops=LOOPS,
    atol=1e-3,
):
    with h5py.File(path, "r") as handle:
        assert _numeric_frame_ids(handle) == list(expected_ids)
        assert _decoded_types(handle["types"]) == list(expected_types)
        for frame_id, expected in zip(expected_ids, expected_coordinates):
            np.testing.assert_allclose(handle[str(frame_id)][()], expected, rtol=0, atol=atol)
        if expected_loops:
            np.testing.assert_array_equal(handle["loops"][()], np.asarray(expected_loops))


@pytest.mark.parametrize("unknown_type", ["NA", "UN"])
def test_ndb_to_cndb_preserves_all_endmdl_frames_types_and_loops(tmp_path, unknown_type):
    source = _write_ndb(tmp_path / "trajectory.ndb", types=("A1", "A2", "B4", unknown_type))
    output = tmp_path / "trajectory.cndb"

    result = ndb_to_cndb(source, output)

    assert Path(result) == output
    _assert_cndb(output)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["format"] == "cndb"
        assert "format_version" in handle.attrs
        assert handle.attrs["genome"] == "hg38"
    assembly_roundtrip = cndb_to_ndb(output, tmp_path / "assembly-roundtrip.ndb")
    assert "ASMBLY    hg38" in assembly_roundtrip.read_text(encoding="utf-8")


def test_ndb_to_cndb_accepts_openmichrom_end_terminator_and_post_end_loops(tmp_path):
    source = _write_openmichrom_end_ndb(tmp_path / "single.ndb")
    output = ndb_to_cndb(source)

    _assert_cndb(
        output,
        expected_ids=(1,),
        expected_types=("A1", "NA", "NA", "B4"),
        expected_coordinates=(COORDINATES[0],),
        expected_loops=((1, 4),),
    )


def test_ndb_to_cndb_reads_the_current_savestructure_reporter_layout(tmp_path):
    """Lock the whitespace fallback needed by the reporter's historical chain field."""

    class _State:
        def getPositions(self, asNumpy=False):
            assert asNumpy is True
            return unit.Quantity(COORDINATES[0], unit.nanometer)

    reporter = SaveStructure(
        filePrefix="reporter",
        reportInterval=1,
        mode="ndb",
        folder=tmp_path,
        chains=[(0, 3, False)],
        typeListLetter=list(TYPES),
    )
    reporter.report(None, _State())
    source = tmp_path / "reporter_0_state0.ndb"

    output = ndb_to_cndb(source)

    _assert_cndb(
        output,
        expected_ids=(1,),
        expected_types=TYPES,
        expected_coordinates=(COORDINATES[0],),
        expected_loops=(),
    )
    with h5py.File(output, "r") as handle:
        chain_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["_ndb_chain_ids"][()]
        ]
        assert chain_ids == ["A1"] * 4


def test_ndb_pdb_roundtrip_preserves_every_standard_chromatin_type(tmp_path):
    source = tmp_path / "all-types.ndb"
    lines = ["MODEL 1"]
    coordinates = np.arange(21, dtype=float).reshape(7, 3) / 8
    for serial, (kind, coordinate) in enumerate(zip(ALL_TYPES, coordinates), 1):
        lines.append(
            f"CHROM {serial} {kind} C1 {serial} "
            f"{coordinate[0]} {coordinate[1]} {coordinate[2]} "
            f"{(serial - 1) * 50000 + 1} {serial * 50000} 0.0"
        )
    lines.extend(["ENDMDL", "END"])
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdb_path = ndb_to_pdb(source)
    ndb_path = pdb_to_ndb(pdb_path, tmp_path / "all-types-roundtrip.ndb")
    cndb_path = ndb_to_cndb(ndb_path, tmp_path / "all-types-roundtrip.cndb")

    _assert_cndb(
        cndb_path,
        expected_ids=(1,),
        expected_types=ALL_TYPES,
        expected_coordinates=(coordinates,),
        expected_loops=(),
    )


def test_pdb_ambiguous_residue_mapping_warns_and_type_override_restores_b4(tmp_path):
    source = tmp_path / "ambiguous.pdb"
    source.write_text(
        "MODEL        1\n"
        "ATOM      1  CA  ARG A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        "ENDMDL\nEND\n",
        encoding="utf-8",
    )

    with pytest.warns(ConverterWarning, match="ambiguous"):
        inferred = pdb_to_ndb(source, tmp_path / "inferred.ndb")
    inferred_cndb = ndb_to_cndb(inferred, tmp_path / "inferred.cndb")
    _assert_cndb(
        inferred_cndb,
        expected_ids=(1,),
        expected_types=("B3",),
        expected_coordinates=(np.asarray([[1.0, 2.0, 3.0]]),),
        expected_loops=(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConverterWarning)
        overridden = pdb_to_ndb(
            source,
            tmp_path / "overridden.ndb",
            types=["B4"],
        )
    overridden_cndb = ndb_to_cndb(overridden, tmp_path / "overridden.cndb")
    with h5py.File(overridden_cndb, "r") as handle:
        assert _decoded_types(handle["types"]) == ["B4"]


@pytest.mark.parametrize("storage", ["numeric", "strings"])
def test_cndb_to_ndb_accepts_numeric_and_string_types_and_preserves_frame_ids(tmp_path, storage):
    source = tmp_path / f"{storage}.cndb"
    coordinates_by_frame = (
        np.arange(21, dtype=float).reshape(7, 3) / 10,
        10 + np.arange(21, dtype=float).reshape(7, 3) / 10,
    )
    with h5py.File(source, "w") as handle:
        handle.attrs["format"] = "cndb"
        handle.attrs["format_version"] = "1.0.0"
        if storage == "numeric":
            handle.create_dataset("types", data=np.arange(7, dtype=np.int8))
        else:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "types",
                data=np.asarray([*ALL_TYPES[:-1], "UN"], dtype=object),
                dtype=string_dtype,
            )
        for frame_id, coordinates in zip(FRAME_IDS, coordinates_by_frame):
            handle.create_dataset(str(frame_id), data=coordinates)
        handle.create_dataset("loops", data=np.asarray(LOOPS, dtype=int))

    ndb_path = cndb_to_ndb(source, tmp_path / f"{storage}.ndb")
    roundtrip = ndb_to_cndb(ndb_path, tmp_path / f"{storage}-roundtrip.cndb")

    _assert_cndb(
        roundtrip,
        expected_types=ALL_TYPES,
        expected_coordinates=coordinates_by_frame,
    )


def test_cndb_to_ndb_reads_format_and_genome_metadata_from_legacy_header_group(tmp_path):
    source = tmp_path / "header-group.cndb"
    with h5py.File(source, "w") as handle:
        header = handle.create_group("Header")
        header.attrs["format"] = "cndb"
        header.attrs["format_version"] = "1.0.0"
        header.attrs["genome"] = "mm10"
        handle.create_dataset("types", data=np.asarray([b"A1"]))
        handle.create_dataset("0", data=np.asarray([[1.0, 2.0, 3.0]]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ndb_path = cndb_to_ndb(source)

    assert "ASMBLY    mm10" in ndb_path.read_text(encoding="utf-8")


def test_ndb_pdb_roundtrip_preserves_frames_coordinates_types_chains_and_loop_sidecar(tmp_path):
    # PDB MODEL numbering normally starts at one; non-contiguous IDs are still valid.
    frame_ids = (1, 8)
    source = _write_ndb(tmp_path / "source.ndb", frame_ids=frame_ids)
    pdb_path = ndb_to_pdb(source, tmp_path / "trajectory.pdb")
    loop_sidecar = Path(pdb_path).with_suffix(".loops")

    assert loop_sidecar.read_text(encoding="utf-8").splitlines() == ["1 4", "2 3"]
    pdb_text = Path(pdb_path).read_text(encoding="utf-8")
    assert "MODEL" in pdb_text
    assert "ENDMDL" in pdb_text

    roundtrip_ndb = pdb_to_ndb(
        pdb_path,
        tmp_path / "from-pdb.ndb",
        loops=loop_sidecar,
        resolution=50_000,
    )
    roundtrip_cndb = ndb_to_cndb(roundtrip_ndb, tmp_path / "from-pdb.cndb")
    _assert_cndb(roundtrip_cndb, expected_ids=frame_ids)
    ndb_text = Path(roundtrip_ndb).read_text(encoding="utf-8")
    assert " C1 " in ndb_text
    assert " C2 " in ndb_text


def test_pdb_to_ndb_uses_ter_boundaries_for_historical_blank_chain_fields(tmp_path):
    def atom_line(serial, atom_name, residue, chain_index, coordinate):
        return (
            f"ATOM  {serial:5d} {atom_name:^4s} {residue:>3s}  {chain_index:4d}    "
            f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
            "  1.00  0.00           C"
        )

    source = tmp_path / "historical-blank-chain.pdb"
    source.write_text(
        "\n".join(
            [
                "MODEL        1",
                atom_line(1, "ZA", "ASP", 1, (1.0, 2.0, 3.0)),
                "TER       2",
                atom_line(2, "LB", "LEU", 1, (4.0, 5.0, 6.0)),
                "TER       3",
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ndb_path = pdb_to_ndb(source)
    cndb_path = ndb_to_cndb(ndb_path)

    _assert_cndb(
        cndb_path,
        expected_ids=(1,),
        expected_types=("A1", "B4"),
        expected_coordinates=(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),),
        expected_loops=(),
    )
    with h5py.File(cndb_path, "r") as handle:
        chain_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["_ndb_chain_ids"][()]
        ]
    assert chain_ids == ["C1", "C2"]


def test_ndb_spacewalk_roundtrip_preserves_frames_coordinates_loci_and_loops(tmp_path):
    frame_ids = (1, 8)
    source = _write_ndb(tmp_path / "source.ndb", frame_ids=frame_ids)
    with pytest.warns(UserWarning, match="(?i)type|sigma|represent"):
        spw_path = ndb_to_spw(
            source,
            tmp_path / "trajectory.spw",
            name="converter-fixture",
        )
    loop_sidecar = Path(spw_path).with_suffix(".loops")

    spw_text = Path(spw_path).read_text(encoding="utf-8")
    assert spw_text.startswith("##format=sw1")
    assert "genome=hg38" in spw_text
    assert "trace 0" in spw_text
    assert "trace 7" in spw_text
    assert loop_sidecar.read_text(encoding="utf-8").splitlines() == ["1 4", "2 3"]

    roundtrip_ndb = spw_to_ndb(
        spw_path,
        tmp_path / "from-spw.ndb",
        loops=loop_sidecar,
        sigma=0.25,
    )
    roundtrip_cndb = ndb_to_cndb(roundtrip_ndb, tmp_path / "from-spw.cndb")
    _assert_cndb(
        roundtrip_cndb,
        expected_ids=frame_ids,
        expected_types=("NA",) * 4,
    )
    ndb_text = Path(roundtrip_ndb).read_text(encoding="utf-8")
    for start, end in GENOMIC_INTERVALS:
        assert str(start) in ndb_text
        assert str(end) in ndb_text


def test_spw_to_ndb_accepts_the_historical_uncommented_column_header(tmp_path):
    source = tmp_path / "historical.spw"
    source.write_text(
        "##format=sw1 name=historical genome=hg38\n"
        "chromosome start end x y z\n"
        "trace 0\n"
        "chrX 1 50000 -1.0 2.0 3.0\n",
        encoding="utf-8",
    )

    ndb_path = spw_to_ndb(source)
    cndb_path = ndb_to_cndb(ndb_path)

    _assert_cndb(
        cndb_path,
        expected_ids=(1,),
        expected_types=("NA",),
        expected_coordinates=(np.asarray([[-1.0, 2.0, 3.0]]),),
        expected_loops=(),
    )
    ndb_text = ndb_path.read_text(encoding="utf-8")
    assert " CX " in ndb_text
    assert "ASMBLY    hg38" in ndb_text


@pytest.mark.parametrize(
    ("exporter", "suffix"),
    [(ndb_to_pdb, ".pdb"), (ndb_to_spw, ".spw")],
)
def test_ndb_text_exports_replace_stale_loop_sidecars_with_empty_files(
    tmp_path, exporter, suffix
):
    with_loops = _write_ndb(tmp_path / "with-loops.ndb", frame_ids=(1,), coordinates=(COORDINATES[0],))
    without_loops = tmp_path / "without-loops.ndb"
    without_loops.write_text(
        "\n".join(
            line
            for line in with_loops.read_text(encoding="utf-8").splitlines()
            if not line.startswith("LOOPS")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / f"reused{suffix}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConverterWarning)
        exporter(with_loops, output)
        assert output.with_suffix(".loops").read_text(encoding="utf-8").strip()
        exporter(without_loops, output, overwrite=True)

    assert output.with_suffix(".loops").is_file()
    assert output.with_suffix(".loops").read_text(encoding="utf-8") == ""


def _gro_atom(residue_number, residue_name, atom_name, atom_number, xyz):
    x, y, z = xyz
    return (
        f"{residue_number:5d}{residue_name:<5s}{atom_name:>5s}{atom_number:5d}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def _write_gro(path):
    legacy_types = ("ZA", "OA", "LB", "UN")
    lines = []
    for frame_number, coordinates in enumerate(COORDINATES, 1):
        lines.extend([f"deterministic frame {frame_number}", str(len(TYPES))])
        lines.extend(
            _gro_atom(index, "ChrA", chrom_type, index, xyz)
            for index, (chrom_type, xyz) in enumerate(zip(legacy_types, coordinates), 1)
        )
        lines.append("   0.00000   0.00000   0.00000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("legacy_code", "expected_type"),
    [
        ("ZA", "A1"),
        ("OA", "A2"),
        ("FB", "B1"),
        ("SB", "B2"),
        ("TB", "B3"),
        ("LB", "B4"),
        ("UN", "NA"),
    ],
)
def test_gro_to_ndb_accepts_every_historical_atom_type_code(
    tmp_path, legacy_code, expected_type
):
    source = tmp_path / f"legacy-{legacy_code}.gro"
    source.write_text(
        "legacy type\n1\n"
        + _gro_atom(1, "ChrA", legacy_code, 1, (1.0, 2.0, 3.0))
        + "\n   0.00000   0.00000   0.00000\n",
        encoding="utf-8",
    )

    ndb_path = gro_to_ndb(source)
    cndb_path = ndb_to_cndb(ndb_path)

    _assert_cndb(
        cndb_path,
        expected_ids=(1,),
        expected_types=(expected_type,),
        expected_coordinates=(np.asarray([[1.0, 2.0, 3.0]]),),
        expected_loops=(),
    )


def test_gro_to_ndb_reads_every_frame_legacy_types_scale_and_loop_file(tmp_path):
    source = _write_gro(tmp_path / "trajectory.gro")
    loop_file = tmp_path / "input.loops"
    loop_file.write_text("1 4\n2 3\n", encoding="utf-8")

    ndb_path = gro_to_ndb(
        source,
        tmp_path / "from-gro.ndb",
        resolution=25_000,
        scale=2.0,
        loops=loop_file,
    )
    cndb_path = ndb_to_cndb(ndb_path, tmp_path / "from-gro.cndb")

    _assert_cndb(
        cndb_path,
        expected_ids=(1, 2),
        expected_coordinates=tuple(frame * 2.0 for frame in COORDINATES),
    )


def _write_bintu_csv(path):
    lines = [
        "Bintu et al. deterministic converter fixture",
        "Chromosome 21 imaging coordinates",
    ]
    for model, coordinates in enumerate(COORDINATES, 1):
        for index, (x, y, z) in enumerate(coordinates, 1):
            lines.append(f"{model},{index},{z},{x},{y}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_bintu_csv_to_ndb_maps_zxy_columns_frames_loci_and_loops(tmp_path):
    source = _write_bintu_csv(tmp_path / "bintu.csv")
    loop_file = tmp_path / "bintu.loops"
    loop_file.write_text("1 4\n2 3\n", encoding="utf-8")

    ndb_path = csv_to_ndb(
        source,
        tmp_path / "from-csv.ndb",
        chromosome=21,
        resolution=30_000,
        genomic_start=18_000_000,
        loops=loop_file,
    )
    cndb_path = ndb_to_cndb(ndb_path, tmp_path / "from-csv.cndb")

    _assert_cndb(
        cndb_path,
        expected_ids=(1, 2),
        expected_types=("NA",) * 4,
    )
    text = Path(ndb_path).read_text(encoding="utf-8")
    assert "C21" in text
    assert "18000000" in text
    assert "18119999" in text


def test_bintu_csv_rejects_a_malformed_first_numeric_row_without_publishing(tmp_path):
    source = tmp_path / "malformed.csv"
    source.write_text("model,index,z,x,y\n1,1,not-a-coordinate,2,3\n", encoding="utf-8")
    output = tmp_path / "malformed.ndb"

    with pytest.raises(ValueError, match="Malformed Bintu CSV row"):
        csv_to_ndb(source, output, chromosome=21)

    assert not output.exists()


def test_generic_convert_infers_formats_and_cndbtools_exposes_static_api(tmp_path):
    source = _write_ndb(tmp_path / "trajectory with ünicode.v1.NDB")
    generic_output = tmp_path / "generic output.CNDB"

    # Inference is case-insensitive and both str and PathLike inputs are public API.
    assert Path(convert(str(source), str(generic_output))) == generic_output
    _assert_cndb(generic_output)

    class_output = tmp_path / "class-api.cndb"
    assert Path(CndbTools.convert(source, class_output)) == class_output
    _assert_cndb(class_output)
    for name in (
        "convert",
        "ndb_to_cndb",
        "cndb_to_ndb",
        "ndb_to_pdb",
        "pdb_to_ndb",
        "ndb_to_spw",
        "spw_to_ndb",
        "gro_to_ndb",
        "csv_to_ndb",
    ):
        assert callable(getattr(CndbTools, name))


def test_generic_convert_accepts_from_format_to_format_aliases(tmp_path):
    source = _write_ndb(tmp_path / "alias-source.ndb")
    output = tmp_path / "alias-output.cndb"

    result = convert(source, output, from_format="ndb", to_format="cndb")

    assert Path(result) == output
    _assert_cndb(output)


def test_generic_convert_can_derive_output_path_from_explicit_format(tmp_path):
    source = _write_ndb(tmp_path / "implicit-output.ndb")

    result = convert(source, output_format="cndb")

    assert Path(result) == source.with_suffix(".cndb")
    _assert_cndb(result)


def test_legacy_ndb2cndb_extensionless_call_remains_compatible(tmp_path):
    base = tmp_path / "legacy"
    _write_openmichrom_end_ndb(base.with_suffix(".ndb"))

    result = CndbTools().ndb2cndb(str(base))

    assert Path(result) == base.with_suffix(".cndb")
    _assert_cndb(
        result,
        expected_ids=(1,),
        expected_types=("A1", "NA", "NA", "B4"),
        expected_coordinates=(COORDINATES[0],),
        expected_loops=((1, 4),),
    )


def test_existing_output_requires_explicit_overwrite_and_is_preserved(tmp_path):
    source = _write_ndb(tmp_path / "source.ndb")
    output = tmp_path / "existing.cndb"
    sentinel = b"do-not-replace"
    output.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        ndb_to_cndb(source, output)
    assert output.read_bytes() == sentinel

    ndb_to_cndb(source, output, overwrite=True)
    _assert_cndb(output)


def test_failed_conversion_is_atomic_for_new_and_existing_outputs(tmp_path):
    malformed = tmp_path / "malformed.ndb"
    malformed.write_text(
        "MODEL 1\nCHROM 1 A1 C1 1 not-a-number 0 0 1 50000 0\nEND\n",
        encoding="utf-8",
    )
    new_output = tmp_path / "new.cndb"

    with pytest.raises((ValueError, TypeError)):
        ndb_to_cndb(malformed, new_output)
    assert not new_output.exists()

    existing_output = tmp_path / "existing.cndb"
    sentinel = b"keep-on-failure"
    existing_output.write_bytes(sentinel)
    with pytest.raises((ValueError, TypeError)):
        ndb_to_cndb(malformed, existing_output, overwrite=True)
    assert existing_output.read_bytes() == sentinel


def test_converter_validation_rejects_missing_source_unsupported_route_and_same_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path / "missing.ndb", tmp_path / "missing.cndb")

    source = _write_bintu_csv(tmp_path / "source.csv")
    unsupported_output = tmp_path / "unsupported.pdb"
    with pytest.raises(ValueError, match="(?i)supported|conversion|route"):
        convert(source, unsupported_output)
    assert not unsupported_output.exists()

    ndb_source = _write_ndb(tmp_path / "same.ndb")
    with pytest.raises(ValueError, match="(?i)same|source|output"):
        convert(ndb_source, ndb_source, input_format="ndb", output_format="cndb")

    gro_source = _write_gro(tmp_path / "invalid-scale.gro")
    with pytest.raises(ValueError, match="positive"):
        gro_to_ndb(gro_source, tmp_path / "invalid-scale.ndb", scale=-1)
