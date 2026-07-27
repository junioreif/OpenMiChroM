# OpenMiChroM benchmarks

This directory contains recorded benchmark measurements for OpenMiChroM
structural-file I/O. The measurements are data-transfer and latency
observations, not a promise of identical timings on other networks or machines.

## ENCODE remote CNDB streaming

Run the benchmark from the repository root:

```bash
python scripts/benchmark_remote_cndb.py \
  --refresh-cache \
  --repeat-reads 3
```

The benchmark uses the public 139 GB ENCODE CNDB file, reads its embedded HDF5
index through HTTP byte ranges, and compares:

- exact contiguous bead-range reads; and
- complete frame reads followed by local NumPy subsetting.

The cache is written under `/tmp` by default and contains index metadata only.
Coordinate data are never cached, and the full CNDB file is not downloaded.

Raw samples and summaries are written to `benchmarks/results/`. PNG and SVG
figures are written to `docs/source/_static/benchmarks/`.
