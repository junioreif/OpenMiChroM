import os

import pytest

from OpenMiChroM._structural_io import detect_structural_file


PUBLIC_STRUCTURAL_URLS = [
    (
        "harris_cndb",
        "https://ndb.rice.edu/d/Harris_etal_NatComm_2023-LCL_chr7_39.5-42.5/LCL_chr7_39.5-42.5.cndb",
        "cndb",
    ),
    (
        "mello_cndb",
        "https://ndb.rice.edu/d/Mello_etal_2026_GM12878_500-frames/GM12878-chr01-500-frames.cndb",
        "cndb",
    ),
    (
        "oliveira_cndb",
        "https://ndb.rice.edu/d/Oliveira_Jr-C1-multichain_2020/GM12878-C01-multichain.cndb",
        "cndb",
    ),
    (
        "encode_cndb",
        "https://www.encodeproject.org/files/ENCFF764BAH/@@download/ENCFF764BAH.cndb",
        "cndb",
    ),
    (
        "mammoth_direct_sw",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE268nnn/GSE268050/suppl/GSE268050%5FWoolly%5FMammoth%5FDirect%5FInv%2Esw",
        "sw",
    ),
    (
        "mammoth_michrom_sw",
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE268050&format=file&file=GSE268050%5FWoolly%5FMammoth%5FMiChroM%2Esw",
        "sw",
    ),
    (
        "bintu_cndb",
        "https://ndb.rice.edu/d/Bintu_etal_Science_2018/A549_chr21-28-30Mb.cndb",
        "cndb",
    ),
]


@pytest.fixture(scope="module")
def system_tls():
    # Optional live tests use native certificate validation, never an unverified
    # context. Restore urllib's normal SSL context when these probes finish.
    import truststore

    truststore.inject_into_ssl()
    try:
        yield
    finally:
        truststore.extract_from_ssl()


@pytest.mark.skipif(
    os.environ.get("OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION") != "1",
    reason="Set OPENMICHROM_RUN_STRUCTURAL_IO_INTEGRATION=1 to inspect public structural URLs.",
)
@pytest.mark.parametrize(("name", "url", "expected_type"), PUBLIC_STRUCTURAL_URLS)
def test_public_structural_url_detection(name, url, expected_type, system_tls):
    info = detect_structural_file(url, timeout=30.0)

    assert info.is_remote is True
    assert info.file_type == expected_type
    assert info.file_size is None or info.file_size > 0
    if not (info.detected_hdf5 or info.detected_text_ndb or info.detected_spacewalk):
        notes = " ".join(info.notes).lower()
        if "timed out" in notes or "timeout" in notes:
            pytest.skip(f"{name} did not respond to public URL probes: {info.notes}")
    assert info.detected_hdf5 or info.detected_text_ndb or info.detected_spacewalk

    assert info.range_supported is True
    if name != "mammoth_michrom_sw":
        assert info.has_embedded_index is True
        assert info.direct_streaming_supported is True
    else:
        assert info.detected_hdf5
