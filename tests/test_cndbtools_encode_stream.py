import os

import pytest

from OpenMiChroM.CndbTools import CndbTools


ENCODE_URL = (
    "https://encode-public.s3.amazonaws.com/2023/02/02/"
    "7f75d816-342a-4b49-adbd-aaa499dc5201/ENCFF161DID.cndb"
)


@pytest.mark.skipif(
    os.environ.get("OPENMICHROM_RUN_ENCODE_STREAM_TESTS") != "1",
    reason="Set OPENMICHROM_RUN_ENCODE_STREAM_TESTS=1 to run the ENCODE streaming test.",
)
def test_encode_remote_stream_xyz_smoke(tmp_path):
    tools = CndbTools.from_remote(
        h5_url=ENCODE_URL,
        trajectory="replica1_chr1",
        index_cache_path=tmp_path / "ENCFF161DID.embedded-index.json.gz",
    )

    coords = tools.xyz(frames=[1], beadSelection=range(0, 10))

    assert coords.shape == (1, 10, 3)
    assert tools.stream_stats()["data_bytes_read"] == 120
