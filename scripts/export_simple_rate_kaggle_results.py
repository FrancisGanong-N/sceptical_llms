#!/usr/bin/env python3
"""Download and export simple-rate merged results from Kaggle Benchmarks run JSON files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.kaggle_runs import (  # noqa: E402
    DEFAULT_SIMPLE_RATE_TASK_SLUG,
    download_task_runs,
    load_base_rate_run_rows_from_tree,
    merged_simple_results_from_kaggle_runs,
)
from benchmarks.simple_rate import BENCHMARK_CSV  # noqa: E402

DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "kaggle_runs" / DEFAULT_SIMPLE_RATE_TASK_SLUG
DEFAULT_MERGED_CSV = ROOT / "data" / "simple" / "simple_merged_results.csv"


def _benchmark_fieldnames(benchmark_path: Path) -> list[str]:
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        benchmark_fields = list(csv.DictReader(handle).fieldnames or [])
    extra = ["path_c_confusion"]
    return benchmark_fields + [
        "model",
        "llm_response",
        "reasoning",
        "answer_line",
        "confidence_line",
        "parsed_answer_type",
        "parsed_percent",
        "parsed_choice",
        "parsed_confidence",
        "scoring_type",
        "parseable",
        "score",
        *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--task", default=DEFAULT_SIMPLE_RATE_TASK_SLUG)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("-m", "--model", action="append", dest="models")
    parser.add_argument("-f", "--force-download", action="store_true")
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_CSV)
    args = parser.parse_args()

    if args.download:
        download_task_runs(
            args.task,
            args.runs_dir,
            models=args.models,
            force=args.force_download,
        )

    if not args.runs_dir.is_dir():
        parser.error(f"Runs directory not found: {args.runs_dir}")

    if not load_base_rate_run_rows_from_tree(args.runs_dir):
        print(f"No per-prompt records found under {args.runs_dir}")
        return 1

    merged = merged_simple_results_from_kaggle_runs(
        args.runs_dir,
        benchmark_path=args.benchmark_csv,
    )
    fieldnames = _benchmark_fieldnames(args.benchmark_csv)
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
