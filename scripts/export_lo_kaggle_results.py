#!/usr/bin/env python3
"""Download and export LO benchmark merged results from Kaggle run JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.kaggle_runs import (  # noqa: E402
    DEFAULT_LO_TASK_SLUGS,
    download_task_runs,
    load_base_rate_run_rows_from_tree,
)
from benchmarks.lp_rate import (  # noqa: E402
    BENCHMARK_CSV,
    write_merged_results_csv,
)

DEFAULT_RUNS_ROOT = ROOT / "data" / "kaggle_runs"
DEFAULT_MERGED_CSV = ROOT / "data" / "lp" / "lo_merged_results.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Task slug to download (repeatable). Default: all four LO versions.",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("-m", "--model", action="append", dest="models")
    parser.add_argument("-f", "--force-download", action="store_true")
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_CSV)
    args = parser.parse_args()

    tasks = args.tasks or list(DEFAULT_LO_TASK_SLUGS)
    run_dirs = [args.runs_root / task for task in tasks]

    if args.download:
        for task, run_dir in zip(tasks, run_dirs, strict=True):
            print(f"Downloading {task} -> {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            download_task_runs(
                task,
                run_dir,
                models=args.models,
                force=args.force_download,
            )

    existing = [d for d in run_dirs if d.is_dir()]
    if not existing:
        parser.error(f"No runs directories found under {args.runs_root} for {tasks}")

    run_rows: list[dict[str, object]] = []
    for run_dir in existing:
        run_rows.extend(load_base_rate_run_rows_from_tree(run_dir))
    if not run_rows:
        print(f"No per-prompt records found under {existing}")
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
    print("Tasks:", ", ".join(tasks))
    print("Models:", ", ".join(models) if models else "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
