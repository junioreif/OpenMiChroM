"""Reproduce the compact training fixtures from the archived full-size files.

The source is the OpenMiChroM tree at 04e53e731a75248ad13a1abb45eb632dbbf7cd5d.
Export tests/{training,chr10,AdamTraining} from that revision to a separate
directory, then run: python make_samples.py <export-root> <new-output-dir>
The shipped samples cover beads 1–24 (inclusive); they are regression inputs,
not whole-chromosome reference maps or new simulations.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def extract(root, output):
    output.mkdir(parents=True, exist_ok=False)
    root = root / "tests"
    n = 24

    def window(source):
        return np.loadtxt(root / source, max_rows=n, usecols=range(n))

    for source, name in [
        ("chr10/chr10_Eprob.txt", "chr10_upper_probability.dense"),
        ("training/c18_10.dense", "chr18_counts.dense"),
        ("AdamTraining/input/chr19_50k.dense", "chr19_counts.dense"),
    ]:
        np.savetxt(output / name, window(source), fmt="%.17g")
    (output / "chr18.seq").write_text(
        "\n".join((root / "training/seq_c18_10").read_text().splitlines()[:n]) + "\n"
    )
    with h5py.File(root / "training/test_0.cndb", "r") as src:
        with h5py.File(output / "chr18.cndb", "x") as dst:
            for name in ("types", "1", "2", "3"):
                dst.create_dataset(name, data=src[name][:n], track_times=False)

    adam = root / "AdamTraining/input"
    with h5py.File(adam / "Pi_0.h5", "r") as f:
        np.savez_compressed(output / "chr19_accumulator.npz", Pi=f["Pi"][:n, :n],
                            frames=int(np.loadtxt(adam / "Nframes_0")))
    with (adam / "lambda_0").open() as src, (output / "chr19_lambda.csv").open("x") as dst:
        for _ in range(n + 1):  # header followed by rows 1–24
            dst.write(",".join(next(src).rstrip().split(",")[:n]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.archive_root, args.output)
