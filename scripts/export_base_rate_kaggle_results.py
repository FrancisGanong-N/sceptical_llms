#!/usr/bin/env python3
"""Download and export base-rate merged results from Kaggle Benchmarks run JSON files.

Typical workflow::

    kaggle auth login
    python scripts/export_base_rate_kaggle_results.py --download
    python scripts/export_base_rate_kaggle_results.py \\
        --runs-dir data/kaggle_runs/base-rate-normative-accuracy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.base_rate import BENCHMARK_CSV, MERGE_RESULT_COLUMNS  # noqa: E402
from benchmarks.kaggle_runs import (  # noqa: E402
    DEFAULT_BASE_RATE_TASK_SLUG,
    download_task_runs,
    load_base_rate_run_rows_from_tree,
    merged_results_from_kaggle_runs,
)

DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "kaggle_runs" / DEFAULT_BASE_RATE_TASK_SLUG
DEFAULT_MERGED_CSV = ROOT / "data" / "base_rate" / "base_rate_merged_results.csv"


def _benchmark_fieldnames(benchmark_path: Path) -> list[str]:
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download run outputs via Kaggle CLI (task: {DEFAULT_BASE_RATE_TASK_SLUG})",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_BASE_RATE_TASK_SLUG,
        help="Benchmark task slug",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory containing downloaded *.run.json files",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        help="Limit download to model slug(s); repeat for multiple models",
    )
    parser.add_argument(
        "-f",
        "--force-download",
        action="store_true",
        help="Pass -f to kaggle benchmarks tasks download",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=DEFAULT_MERGED_CSV,
        help="Write merged results CSV for base-rate-results.ipynb",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=BENCHMARK_CSV,
        help="Current benchmark definition CSV",
    )
    args = parser.parse_args()

    if args.download:
        download_task_runs(
            args.task,
            args.runs_dir,
            models=args.models,
            force=args.force_download,
        )

    if not args.runs_dir.is_dir():
        parser.error(
            f"Runs directory not found: {args.runs_dir}\n"
            "Use --download after `kaggle auth login`, or pass --runs-dir."
        )

    run_rows = load_base_rate_run_rows_from_tree(args.runs_dir)
    if not run_rows:
        print(
            f"No per-prompt records found under {args.runs_dir}.\n"
            f"Expected aggregate *.run.json files from task {args.task}."
        )
        return 1

    merged = merged_results_from_kaggle_runs(
        args.runs_dir,
        benchmark_path=args.benchmark_csv,
    )
    fieldnames = _benchmark_fieldnames(args.benchmark_csv) + list(MERGE_RESULT_COLUMNS)
    args.merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.merged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    models = sorted({row["model"] for row in merged})
    print(f"Wrote {len(merged)} rows ({len(models)} models) to {args.merged_csv}")
    print("Models:", ", ".join(models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
