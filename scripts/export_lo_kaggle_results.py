#!/usr/bin/env python3
"""Download and export LO benchmark merged results from Kaggle run JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.kaggle_runs import (  # noqa: E402
    download_task_runs,
    load_base_rate_run_rows_from_tree,
)
from benchmarks.lp_rate import (  # noqa: E402
    BENCHMARK_CSV,
    write_merged_results_csv,
)

DEFAULT_LO_TASK_SLUG = "lo-normative-accuracy-5"
DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "kaggle_runs" / DEFAULT_LO_TASK_SLUG
DEFAULT_MERGED_CSV = ROOT / "data" / "lp" / "lo_merged_results.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--task", default=DEFAULT_LO_TASK_SLUG)
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

    run_rows = load_base_rate_run_rows_from_tree(args.runs_dir)
    if not run_rows:
        print(f"No per-prompt records found under {args.runs_dir}")
        return 1

    merged_path, pivot_path = write_merged_results_csv(
        run_rows,
        args.merged_csv,
        benchmark_path=args.benchmark_csv,
    )
    models = sorted(
        {
            str(row.get("model") or "").strip()
            for row in run_rows
            if row.get("model")
        }
    )
    print(f"Wrote {merged_path}")
    print(f"Wrote {pivot_path}")
    print("Models:", ", ".join(models) if models else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
