"""Parse Kaggle Benchmarks ``*.run.json`` files into base-rate merged result rows."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BASE_RATE_PROMPT_DICT_KEYS = frozenset({"example_id", "response"})

DEFAULT_BASE_RATE_TASK_SLUG = "base-rate-normative-accuracy"
DEFAULT_SIMPLE_RATE_TASK_SLUG = "simple-rate-normative-accuracy"
DEFAULT_LO_TASK_SLUG = "lo-normative-accuracy-6"


@dataclass(frozen=True)
class BaseRatePromptRecord:
    """One model answer for a base-rate benchmark prompt."""

    model: str
    example_id: str
    response: str
    reasoning: str | None
    run_file: Path
    task_name: str | None = None


def _get_task_name(run_data: dict[str, Any]) -> str | None:
    task_version = run_data.get("taskVersion") or run_data.get("task_version") or {}
    name = task_version.get("name")
    return str(name) if name else None


def _get_model_slug(run_data: dict[str, Any]) -> str | None:
    model_version = run_data.get("modelVersion") or run_data.get("model_version") or {}
    slug = model_version.get("slug")
    return str(slug) if slug else None


def _extract_prompt_dict(run_data: dict[str, Any]) -> dict[str, Any] | None:
    results = run_data.get("results") or []
    if not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    payload = first.get("dictResult") or first.get("dict_result")
    if not isinstance(payload, dict):
        return None
    if not BASE_RATE_PROMPT_DICT_KEYS.issubset(payload.keys()):
        return None
    return payload


def _normalize_reasoning(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def iter_base_rate_prompt_records(
    run_data: dict[str, Any],
    *,
    run_file: Path,
    model: str | None = None,
) -> Iterable[BaseRatePromptRecord]:
    """Yield per-prompt records from one run JSON object (including subruns)."""
    model_slug = model or _get_model_slug(run_data)
    if not model_slug:
        return

    payload = _extract_prompt_dict(run_data)
    if payload is not None:
        yield BaseRatePromptRecord(
            model=model_slug,
            example_id=str(payload["example_id"]),
            response=str(payload.get("response") or ""),
            reasoning=_normalize_reasoning(payload.get("reasoning")) or None,
            run_file=run_file,
            task_name=_get_task_name(run_data),
        )

    for subrun in run_data.get("subruns") or []:
        if isinstance(subrun, dict):
            yield from iter_base_rate_prompt_records(
                subrun, run_file=run_file, model=model_slug
            )


def load_base_rate_prompt_records_from_run_file(path: Path | str) -> list[BaseRatePromptRecord]:
    run_file = Path(path)
    with run_file.open(encoding="utf-8") as handle:
        run_data = json.load(handle)
    return list(iter_base_rate_prompt_records(run_data, run_file=run_file))


def find_run_json_files(root: Path | str) -> list[Path]:
    root_path = Path(root)
    if root_path.is_file() and root_path.name.endswith(".run.json"):
        return [root_path]
    return sorted(root_path.rglob("*.run.json"))


def load_base_rate_prompt_records_from_tree(root: Path | str) -> list[BaseRatePromptRecord]:
    records: list[BaseRatePromptRecord] = []
    for run_file in find_run_json_files(root):
        records.extend(load_base_rate_prompt_records_from_run_file(run_file))
    return records


def dedupe_base_rate_prompt_records(
    records: Iterable[BaseRatePromptRecord],
) -> list[BaseRatePromptRecord]:
    """Keep the last record per (model, example_id), e.g. after multiple downloads."""
    latest: dict[tuple[str, str], BaseRatePromptRecord] = {}
    for record in records:
        latest[(record.model, record.example_id)] = record
    return [
        latest[key]
        for key in sorted(latest, key=lambda pair: (pair[0], pair[1]))
    ]


def base_rate_prompt_records_to_run_rows(
    records: Iterable[BaseRatePromptRecord],
) -> list[dict[str, object]]:
    return [
        {
            "example_id": record.example_id,
            "response": record.response,
            "reasoning": record.reasoning or "",
            "model": record.model,
        }
        for record in records
    ]


def filter_run_rows_to_benchmark(
    run_rows: list[dict[str, object]],
    *,
    benchmark_example_ids: set[str] | frozenset[str],
) -> list[dict[str, object]]:
    """Drop run rows whose ``example_id`` is not in the current benchmark CSV."""
    allowed = frozenset(benchmark_example_ids)
    return [
        row
        for row in run_rows
        if str(row.get("example_id", "")).strip() in allowed
    ]


def load_base_rate_run_rows_from_tree(root: Path | str) -> list[dict[str, object]]:
    records = dedupe_base_rate_prompt_records(load_base_rate_prompt_records_from_tree(root))
    return base_rate_prompt_records_to_run_rows(records)


def _run_rows_for_benchmark_merge(
    root: Path | str,
    *,
    benchmark_path: Path | str,
    task_slug: str,
    download_hint: str,
) -> list[dict[str, object]]:
    from benchmarks.base_rate import load_benchmark_rows

    benchmark_path = Path(benchmark_path)
    run_rows = load_base_rate_run_rows_from_tree(root)
    if not run_rows:
        raise FileNotFoundError(
            f"No per-prompt records found under {root}. "
            "Download runs with:\n"
            f"  {download_hint.format(slug=task_slug)}"
        )
    _, benchmark_rows = load_benchmark_rows(benchmark_path)
    benchmark_example_ids = {row["example_id"] for row in benchmark_rows}
    n_before = len(run_rows)
    run_rows = filter_run_rows_to_benchmark(
        run_rows,
        benchmark_example_ids=benchmark_example_ids,
    )
    n_skipped = n_before - len(run_rows)
    if n_skipped:
        print(
            f"Skipped {n_skipped} stale run row(s) with example_ids not in "
            f"{benchmark_path.name}.",
            flush=True,
        )
    if not run_rows:
        raise FileNotFoundError(
            f"No run rows left after filtering to {benchmark_path.name} "
            f"({n_skipped} stale row(s) skipped under {root})."
        )
    return run_rows


def merged_results_from_kaggle_runs(
    root: Path | str,
    *,
    benchmark_path: Path | str,
    fill_missing: bool = False,
):
    """Build merged benchmark rows (list[dict]) from downloaded Kaggle run JSON files."""
    from benchmarks.base_rate import BENCHMARK_CSV, merge_run_results

    benchmark_path = Path(benchmark_path or BENCHMARK_CSV)
    run_rows = _run_rows_for_benchmark_merge(
        root,
        benchmark_path=benchmark_path,
        task_slug=DEFAULT_BASE_RATE_TASK_SLUG,
        download_hint=(
            "kaggle benchmarks tasks download {slug} "
            f"-o data/kaggle_runs/{DEFAULT_BASE_RATE_TASK_SLUG}"
        ),
    )
    return merge_run_results(
        run_rows,
        benchmark_path=benchmark_path,
        fill_missing=fill_missing,
    )


def merged_simple_results_from_kaggle_runs(
    root: Path | str,
    *,
    benchmark_path: Path | str,
    fill_missing: bool = False,
):
    """Build merged simple-benchmark rows from downloaded Kaggle run JSON files."""
    from benchmarks.simple_rate import BENCHMARK_CSV, merge_run_results

    benchmark_path = Path(benchmark_path or BENCHMARK_CSV)
    run_rows = _run_rows_for_benchmark_merge(
        root,
        benchmark_path=benchmark_path,
        task_slug=DEFAULT_SIMPLE_RATE_TASK_SLUG,
        download_hint=(
            "python -m kaggle benchmarks tasks download {slug} "
            f"-o data/kaggle_runs/{DEFAULT_SIMPLE_RATE_TASK_SLUG}"
        ),
    )
    return merge_run_results(
        run_rows,
        benchmark_path=benchmark_path,
        fill_missing=fill_missing,
    )


def merged_lo_results_from_kaggle_runs(
    root: Path | str,
    *,
    benchmark_path: Path | str | None = None,
    fill_missing: bool = False,
):
    """Build merged LO-benchmark rows from downloaded Kaggle run JSON files."""
    from benchmarks.lp_rate import BENCHMARK_CSV, merge_run_results

    benchmark_path = Path(benchmark_path or BENCHMARK_CSV)
    run_rows = _run_rows_for_benchmark_merge(
        root,
        benchmark_path=benchmark_path,
        task_slug=DEFAULT_LO_TASK_SLUG,
        download_hint=(
            "python -m kaggle benchmarks tasks download {slug} "
            f"-o data/kaggle_runs/{DEFAULT_LO_TASK_SLUG}"
        ),
    )
    return merge_run_results(
        run_rows,
        benchmark_path=benchmark_path,
        fill_missing=fill_missing,
    )


def kaggle_cmd(*args: str) -> list[str]:
    """Build a Kaggle CLI argv list (standalone ``kaggle`` or ``python -m kaggle``)."""
    exe = shutil.which("kaggle")
    if exe:
        return [exe, *args]
    scripts_dir = Path(sys.executable).resolve().parent
    candidate = scripts_dir / "kaggle.exe"
    if candidate.is_file():
        return [str(candidate), *args]
    try:
        import kaggle  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then authenticate: python -m kaggle auth login"
        ) from exc
    return [sys.executable, "-m", "kaggle", *args]


def kaggle_exe() -> str:
    """First token of :func:`kaggle_cmd` (for callers that only need the executable)."""
    return kaggle_cmd()[0]


def download_task_runs(
    task_slug: str,
    output_dir: Path | str,
    *,
    models: list[str] | None = None,
    force: bool = False,
    include_source: bool = False,
) -> None:
    cmd = kaggle_cmd(
        "benchmarks",
        "tasks",
        "download",
        task_slug,
        "-o",
        str(output_dir),
    )
    for model in models or []:
        cmd.extend(["-m", model])
    if force:
        cmd.append("-f")
    if include_source:
        cmd.append("-s")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        hint = ""
        if "Authentication required" in detail or "auth login" in detail.lower():
            hint = (
                "\nAuthenticate once in this environment, then re-run the cell:\n"
                f"  {sys.executable} -m kaggle auth login"
            )
        raise RuntimeError(
            f"Kaggle download failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"{detail}{hint}"
        )
