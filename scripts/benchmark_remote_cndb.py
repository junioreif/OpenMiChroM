#!/usr/bin/env python
"""Benchmark OpenMiChroM CNDBTools remote indexed coordinate reads.

The benchmark compares exact contiguous bead-range reads with the older
full-frame-read-then-subset pattern. It never downloads the full remote HDF5
file: all HDF5 access goes through the range-enforcing CNDBTools backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from OpenMiChroM.CndbTools import CndbTools


ENCODE_URL = (
    "https://encode-public.s3.amazonaws.com/2023/02/02/"
    "7f75d816-342a-4b49-adbd-aaa499dc5201/ENCFF161DID.cndb"
)
ENCODE_FILE_SIZE = 139_009_629_470
DEFAULT_TRAJECTORY = "replica1_chr1"
DEFAULT_FRAMES = ("1", "10", "100", "1000")
CSV_FIELDS = (
    "phase",
    "operation",
    "method",
    "repeat",
    "trajectory",
    "frames",
    "requested_beads",
    "transferred_beads",
    "shape",
    "elapsed_seconds",
    "index_cache_hit",
    "index_bytes",
    "metadata_bytes",
    "data_bytes",
    "total_bytes",
    "expected_coordinate_bytes",
    "byte_match",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-url", default=ENCODE_URL)
    parser.add_argument("--trajectory", default=DEFAULT_TRAJECTORY)
    parser.add_argument("--frame", default="1")
    parser.add_argument(
        "--multi-frames",
        default=",".join(DEFAULT_FRAMES),
        help="Comma-separated frames for the multi-frame benchmark.",
    )
    parser.add_argument(
        "--bead-counts",
        default="10,100,1000,full",
        help="Comma-separated bead counts; use 'full' for the complete frame.",
    )
    parser.add_argument("--repeat-reads", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--index-cache-path",
        default="/tmp/ENCFF161DID.openmichrom-index.json.gz",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Remove the selected cache before the cold-open measurement.",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results",
        help="Directory for CSV and JSON benchmark data.",
    )
    parser.add_argument(
        "--figures-dir",
        default="docs/source/_static/benchmarks",
        help="Directory for PNG and SVG figures.",
    )
    parser.add_argument(
        "--output-prefix",
        default=f"encode_remote_streaming_{datetime.now():%Y-%m-%d}",
    )
    return parser


def parse_csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def package_version() -> str:
    try:
        return version("OpenMiChroM")
    except PackageNotFoundError:
        return "unknown"


def stats_snapshot(tools: CndbTools) -> dict[str, Any]:
    stats = tools.stream_stats()
    return {
        "index_bytes": int(stats["index_bytes_read"]),
        "metadata_bytes": int(stats["metadata_bytes_read"]),
        "data_bytes": int(stats["data_bytes_read"]),
        "total_bytes": int(stats["bytes_read"]),
        "index_cache_hit": bool(stats["index_cache_hit"]),
    }


def stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(after[key]) - int(before[key])
        for key in ("index_bytes", "metadata_bytes", "data_bytes", "total_bytes")
    }


def open_remote(
    *,
    h5_url: str,
    trajectory: str,
    cache_path: Path,
    timeout: float,
) -> tuple[CndbTools, float, dict[str, Any]]:
    started = perf_counter()
    tools = CndbTools.from_remote(
        h5_url=h5_url,
        trajectory=trajectory,
        index_cache_path=str(cache_path),
        timeout=timeout,
    )
    elapsed = perf_counter() - started
    return tools, elapsed, stats_snapshot(tools)


def prefetch_metadata(tools: CndbTools, frames: Iterable[str]) -> tuple[float, dict[str, int]]:
    """Resolve frame metadata once so timed reads isolate coordinate requests."""

    reader = tools._stream_backend.traj
    before = stats_snapshot(tools)
    started = perf_counter()
    reader.prefetch_frame_metadata(frames)
    elapsed = perf_counter() - started
    after = stats_snapshot(tools)
    return elapsed, stats_delta(before, after)


def run_coordinate_read(
    tools: CndbTools,
    *,
    frames: list[str],
    requested_beads: int,
    full_frame_then_subset: bool,
) -> tuple[np.ndarray, float, dict[str, int]]:
    before = stats_snapshot(tools)
    started = perf_counter()
    if full_frame_then_subset:
        full = tools.xyz(frames=frames, beadSelection=None)
        result = full[:, :requested_beads, :]
    else:
        result = tools.xyz(
            frames=frames,
            beadSelection=range(0, requested_beads),
        )
    elapsed = perf_counter() - started
    after = stats_snapshot(tools)
    return result, elapsed, stats_delta(before, after)


def startup_row(
    *,
    operation: str,
    trajectory: str,
    elapsed: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "startup",
        "operation": operation,
        "method": "embedded_index",
        "repeat": 1,
        "trajectory": trajectory,
        "frames": "",
        "requested_beads": "",
        "transferred_beads": "",
        "shape": "",
        "elapsed_seconds": elapsed,
        "index_cache_hit": stats["index_cache_hit"],
        "index_bytes": stats["index_bytes"],
        "metadata_bytes": stats["metadata_bytes"],
        "data_bytes": stats["data_bytes"],
        "total_bytes": stats["total_bytes"],
        "expected_coordinate_bytes": 0,
        "byte_match": stats["data_bytes"] == 0,
    }


def prefetch_row(
    *,
    trajectory: str,
    frames: list[str],
    elapsed: float,
    delta: dict[str, int],
    cache_hit: bool,
) -> dict[str, Any]:
    return {
        "phase": "prefetch",
        "operation": "frame_metadata_prefetch",
        "method": "embedded_index_metadata",
        "repeat": 1,
        "trajectory": trajectory,
        "frames": ",".join(frames),
        "requested_beads": 0,
        "transferred_beads": 0,
        "shape": "",
        "elapsed_seconds": elapsed,
        "index_cache_hit": cache_hit,
        "index_bytes": delta["index_bytes"],
        "metadata_bytes": delta["metadata_bytes"],
        "data_bytes": delta["data_bytes"],
        "total_bytes": delta["total_bytes"],
        "expected_coordinate_bytes": 0,
        "byte_match": delta["data_bytes"] == 0,
    }


def read_row(
    *,
    operation: str,
    method: str,
    repeat: int,
    trajectory: str,
    frames: list[str],
    requested_beads: int,
    transferred_beads: int,
    result: np.ndarray,
    elapsed: float,
    delta: dict[str, int],
    cache_hit: bool,
) -> dict[str, Any]:
    expected = (
        len(frames)
        * requested_beads
        * int(result.shape[-1])
        * int(result.dtype.itemsize)
    )
    if method == "exact_range":
        byte_match = delta["data_bytes"] == expected
    else:
        full_frame_bytes = (
            len(frames)
            * transferred_beads
            * int(result.shape[-1])
            * int(result.dtype.itemsize)
        )
        byte_match = delta["data_bytes"] == full_frame_bytes
    return {
        "phase": "read",
        "operation": operation,
        "method": method,
        "repeat": repeat,
        "trajectory": trajectory,
        "frames": ",".join(frames),
        "requested_beads": requested_beads,
        "transferred_beads": transferred_beads,
        "shape": "x".join(str(value) for value in result.shape),
        "elapsed_seconds": elapsed,
        "index_cache_hit": cache_hit,
        "index_bytes": delta["index_bytes"],
        "metadata_bytes": delta["metadata_bytes"],
        "data_bytes": delta["data_bytes"],
        "total_bytes": delta["total_bytes"],
        "expected_coordinate_bytes": expected,
        "byte_match": byte_match,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase"] != "read":
            continue
        key = (
            str(row["operation"]),
            str(row["method"]),
            int(row["requested_beads"]),
            str(row["frames"]),
        )
        grouped[key].append(row)

    summaries = []
    for (operation, method, requested_beads, frames), samples in sorted(grouped.items()):
        elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
        data_bytes = [int(sample["data_bytes"]) for sample in samples]
        total_bytes = [int(sample["total_bytes"]) for sample in samples]
        summaries.append(
            {
                "operation": operation,
                "method": method,
                "frames": frames,
                "requested_beads": requested_beads,
                "samples": len(samples),
                "median_seconds": statistics.median(elapsed),
                "mean_seconds": statistics.fmean(elapsed),
                "min_seconds": min(elapsed),
                "max_seconds": max(elapsed),
                "data_bytes": int(statistics.median(data_bytes)),
                "total_bytes": int(statistics.median(total_bytes)),
                "expected_coordinate_bytes": int(samples[0]["expected_coordinate_bytes"]),
                "all_byte_checks_passed": all(bool(sample["byte_match"]) for sample in samples),
            }
        )
    return summaries


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_figure(fig: Any, base_path: Path) -> list[str]:
    paths = []
    for suffix in (".png", ".svg"):
        path = base_path.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    return paths


def make_figures(summary: dict[str, Any], figures_dir: Path, prefix: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Figure generation requires matplotlib. Install the documentation "
            "requirements or matplotlib directly."
        ) from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#1F2937",
            "text.color": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 10,
        }
    )
    colors = {"exact_range": "#007C91", "full_frame_then_subset": "#D97706"}
    created: list[str] = []

    startup = summary["startup"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Cold open", "Warm cached open"]
    values = [startup["cold_open_seconds"], startup["warm_open_seconds"]]
    bars = ax.bar(labels, values, color=["#B91C1C", "#007C91"], width=0.58)
    ax.set_ylabel("Initialization time (seconds)")
    ax.set_title("Remote CNDB initialization time")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    created.extend(save_figure(fig, figures_dir / f"{prefix}_startup_latency"))
    plt.close(fig)

    single_frame = [
        row for row in summary["read_summary"] if row["operation"] == "single_frame"
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, label in (
        ("exact_range", "Exact byte range"),
        ("full_frame_then_subset", "Full frame, then local subset"),
    ):
        rows = sorted(
            (row for row in single_frame if row["method"] == method),
            key=lambda row: row["requested_beads"],
        )
        x = [row["requested_beads"] for row in rows]
        y = [row["median_seconds"] for row in rows]
        low = [row["median_seconds"] - row["min_seconds"] for row in rows]
        high = [row["max_seconds"] - row["median_seconds"] for row in rows]
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors[method],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Requested beads")
    ax.set_ylabel("Read time (seconds)")
    ax.set_title("Remote coordinate read latency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    created.extend(save_figure(fig, figures_dir / f"{prefix}_read_latency"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, label in (
        ("exact_range", "Exact byte range"),
        ("full_frame_then_subset", "Full frame, then local subset"),
    ):
        rows = sorted(
            (row for row in single_frame if row["method"] == method),
            key=lambda row: row["requested_beads"],
        )
        ax.plot(
            [row["requested_beads"] for row in rows],
            [row["data_bytes"] for row in rows],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Requested beads")
    ax.set_ylabel("Coordinate payload bytes transferred")
    ax.set_title("Remote coordinate payload by request size")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    created.extend(save_figure(fig, figures_dir / f"{prefix}_coordinate_bytes"))
    plt.close(fig)

    multi_frame = [
        row for row in summary["read_summary"] if row["operation"] == "multi_frame"
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Exact ranges", "Full frames,\nthen subset"]
    ordered = [
        next(row for row in multi_frame if row["method"] == "exact_range"),
        next(
            row
            for row in multi_frame
            if row["method"] == "full_frame_then_subset"
        ),
    ]
    values = [row["data_bytes"] for row in ordered]
    bars = ax.bar(labels, values, color=[colors["exact_range"], colors["full_frame_then_subset"]])
    ax.set_ylabel("Coordinate payload bytes transferred")
    ax.set_title("Four frames with 100 requested beads per frame")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,} B",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    created.extend(save_figure(fig, figures_dir / f"{prefix}_four_frame_bytes"))
    plt.close(fig)

    return created


def print_summary(summary: dict[str, Any], csv_path: Path, json_path: Path) -> None:
    startup = summary["startup"]
    print(f"Cold open:        {startup['cold_open_seconds']:.3f} s")
    print(f"Warm cached open: {startup['warm_open_seconds']:.3f} s")
    print(
        f"Index bytes: cold={startup['cold_index_bytes']:,}, "
        f"warm={startup['warm_index_bytes']:,}"
    )
    print("")
    print(
        f"{'operation':14} {'method':25} {'beads':>7} "
        f"{'median (s)':>11} {'data bytes':>12} {'byte check':>11}"
    )
    print("-" * 88)
    for row in summary["read_summary"]:
        print(
            f"{row['operation']:14} {row['method']:25} "
            f"{row['requested_beads']:7d} {row['median_seconds']:11.4f} "
            f"{row['data_bytes']:12,d} "
            f"{'yes' if row['all_byte_checks_passed'] else 'NO':>11}"
        )
    print("")
    print(f"Raw results: {csv_path}")
    print(f"Summary:     {json_path}")
    for figure in summary["figures"]:
        print(f"Figure:      {figure}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat_reads < 1:
        raise ValueError("--repeat-reads must be at least 1.")

    cache_path = Path(args.index_cache_path).expanduser()
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cold_tools, cold_seconds, cold_stats = open_remote(
        h5_url=args.h5_url,
        trajectory=args.trajectory,
        cache_path=cache_path,
        timeout=args.timeout,
    )
    rows.append(
        startup_row(
            operation="cold_open",
            trajectory=args.trajectory,
            elapsed=cold_seconds,
            stats=cold_stats,
        )
    )
    del cold_tools

    tools, warm_seconds, warm_stats = open_remote(
        h5_url=args.h5_url,
        trajectory=args.trajectory,
        cache_path=cache_path,
        timeout=args.timeout,
    )
    rows.append(
        startup_row(
            operation="warm_cached_open",
            trajectory=args.trajectory,
            elapsed=warm_seconds,
            stats=warm_stats,
        )
    )

    frame = str(args.frame)
    multi_frames = parse_csv_values(args.multi_frames)
    all_frames = list(dict.fromkeys([frame, *multi_frames]))
    prefetch_seconds, prefetch_delta = prefetch_metadata(tools, all_frames)
    rows.append(
        prefetch_row(
            trajectory=args.trajectory,
            frames=all_frames,
            elapsed=prefetch_seconds,
            delta=prefetch_delta,
            cache_hit=warm_stats["index_cache_hit"],
        )
    )

    counts = []
    for raw_count in parse_csv_values(args.bead_counts):
        if raw_count.lower() == "full":
            count = int(tools.Nbeads)
        else:
            count = int(raw_count)
        if count < 1 or count > tools.Nbeads:
            raise ValueError(f"Bead count {count} is outside 1:{tools.Nbeads}.")
        counts.append(count)
    counts = list(dict.fromkeys(counts))

    for requested_beads in counts:
        for method in ("exact_range", "full_frame_then_subset"):
            for repeat in range(1, args.repeat_reads + 1):
                result, elapsed, delta = run_coordinate_read(
                    tools,
                    frames=[frame],
                    requested_beads=requested_beads,
                    full_frame_then_subset=method == "full_frame_then_subset",
                )
                rows.append(
                    read_row(
                        operation="single_frame",
                        method=method,
                        repeat=repeat,
                        trajectory=args.trajectory,
                        frames=[frame],
                        requested_beads=requested_beads,
                        transferred_beads=(
                            tools.Nbeads
                            if method == "full_frame_then_subset"
                            else requested_beads
                        ),
                        result=result,
                        elapsed=elapsed,
                        delta=delta,
                        cache_hit=warm_stats["index_cache_hit"],
                    )
                )

    multi_requested_beads = min(100, tools.Nbeads)
    for method in ("exact_range", "full_frame_then_subset"):
        for repeat in range(1, args.repeat_reads + 1):
            result, elapsed, delta = run_coordinate_read(
                tools,
                frames=multi_frames,
                requested_beads=multi_requested_beads,
                full_frame_then_subset=method == "full_frame_then_subset",
            )
            rows.append(
                read_row(
                    operation="multi_frame",
                    method=method,
                    repeat=repeat,
                    trajectory=args.trajectory,
                    frames=multi_frames,
                    requested_beads=multi_requested_beads,
                    transferred_beads=(
                        tools.Nbeads
                        if method == "full_frame_then_subset"
                        else multi_requested_beads
                    ),
                    result=result,
                    elapsed=elapsed,
                    delta=delta,
                    cache_hit=warm_stats["index_cache_hit"],
                )
            )

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "openmichrom": package_version(),
        },
        "source": {
            "url": args.h5_url,
            "file_size_bytes": ENCODE_FILE_SIZE if args.h5_url == ENCODE_URL else None,
            "trajectory": args.trajectory,
            "frame": frame,
            "multi_frames": multi_frames,
            "n_beads": int(tools.Nbeads),
            "coordinate_dtype": "float32",
        },
        "methodology": {
            "repeat_reads": args.repeat_reads,
            "metadata_prefetched": True,
            "exact_range": "Read only requested contiguous bead rows with HTTP Range.",
            "full_frame_then_subset": (
                "Read the complete frame through the same backend, then subset in NumPy."
            ),
            "cache_path": str(cache_path),
            "cache_size_bytes": cache_path.stat().st_size,
        },
        "startup": {
            "cold_open_seconds": cold_seconds,
            "cold_index_cache_hit": cold_stats["index_cache_hit"],
            "cold_index_bytes": cold_stats["index_bytes"],
            "cold_metadata_bytes": cold_stats["metadata_bytes"],
            "cold_total_bytes": cold_stats["total_bytes"],
            "warm_open_seconds": warm_seconds,
            "warm_index_cache_hit": warm_stats["index_cache_hit"],
            "warm_index_bytes": warm_stats["index_bytes"],
            "warm_metadata_bytes": warm_stats["metadata_bytes"],
            "warm_total_bytes": warm_stats["total_bytes"],
        },
        "prefetch": {
            "seconds": prefetch_seconds,
            **prefetch_delta,
        },
        "read_summary": aggregate_rows(rows),
        "figures": [],
    }

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    csv_path = results_dir / f"{args.output_prefix}.csv"
    json_path = results_dir / f"{args.output_prefix}.json"
    write_csv(rows, csv_path)
    summary["figures"] = make_figures(summary, figures_dir, args.output_prefix)
    write_json(summary, json_path)
    print_summary(summary, csv_path, json_path)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
