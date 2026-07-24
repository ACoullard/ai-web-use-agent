from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.grading import CriterionResult


class RunRecord(BaseModel):
    """One fixture's outcome within one `evals run` invocation - one JSON line in the
    append-only history log.
    """

    schema_version: int = 1
    run_id: str
    run_at: datetime
    fixture_id: str
    grading: str
    expected_status: str
    status: str
    passed: bool
    rationale: str | None = None
    criteria_results: list[CriterionResult] | None = None
    pass_fraction: float | None = None
    steps_taken: int
    duration_seconds: float
    model: str
    judge_model: str | None = None
    answer: Any = None
    error: str | None = None


def append_history(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def load_history(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(RunRecord.model_validate_json(line))
    return records


def pass_rate_by_run(records: list[RunRecord]) -> list[tuple[str, datetime, float]]:
    """Chronological (run_id, run_at, pass_rate) per distinct run_id - a simple
    read-and-aggregate over the JSONL history, not a database query.
    """
    groups: dict[str, list[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.run_id, []).append(record)

    results = [
        (run_id, min(r.run_at for r in group), sum(r.passed for r in group) / len(group))
        for run_id, group in groups.items()
    ]
    results.sort(key=lambda item: item[1])
    return results
