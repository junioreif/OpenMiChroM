"""Reproduce the first 512 bins from the archived GM12878 chr10 map (100 kb/bin)."""

import gzip
import hashlib
from pathlib import Path


FULL_SHA256 = "5109a7f4a6692ccea90163cdf8878dfa72710a51ee9aeadfff51948b416b7227"
FULL_URL = (
    "https://raw.githubusercontent.com/junioreif/OpenMiChroM/"
    "04e53e731a75248ad13a1abb45eb632dbbf7cd5d/"
    "Tutorials/MiChroM_Optimization/input/chr10_100k.dense"
)
SAMPLE_NAME = "chr10_first512_100k.dense.gz"


def extract_sample(full_map):
    """Return the original tokens (including missing data), without numeric conversion."""
    content = Path(full_map).read_bytes()
    if hashlib.sha256(content).hexdigest() != FULL_SHA256:
        raise ValueError("Source is not the archived GM12878 chr10 contact map")
    return b"".join(b"\t".join(line.split()[:512]) + b"\n"
                    for line in content.splitlines()[:512])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_map", type=Path, help=f"Original dense text file from {FULL_URL}")
    args = parser.parse_args()
    data = extract_sample(args.full_map)
    target = Path(__file__).with_name(SAMPLE_NAME)
    with target.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as archive:
        archive.write(data)
