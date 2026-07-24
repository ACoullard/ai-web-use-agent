from __future__ import annotations

import typer

from evals.history import RunRecord


def format_progress_mark(record: RunRecord) -> str:
    return typer.style(".", fg="green") if record.passed else typer.style("F", fg="red")


def _criterion_lines(record: RunRecord) -> list[str]:
    lines = []
    for criterion in record.criteria_results or []:
        prefix = "[PASS]" if criterion.passes else "[FAIL]"
        line = f"  {prefix} {criterion.criterion}"
        if not criterion.passes:
            line += f"\n         reason: {criterion.reason}"
        lines.append(line)
    return lines


def format_failure_section(records: list[RunRecord]) -> str:
    """A red pytest-style 'FAILURES' block: full rationale for exact_match/status
    failures, or a per-criterion [PASS]/[FAIL] breakdown plus the pass fraction for
    rubric failures - the fraction stays visible even though the overall grade is
    strict all-or-nothing. Returns "" when there are no failures.
    """
    failures = [record for record in records if not record.passed]
    if not failures:
        return ""

    blocks = []
    for record in failures:
        header = typer.style(f"FAIL {record.fixture_id}", fg="red", bold=True)
        body_lines: list[str] = []
        if record.criteria_results:
            body_lines.extend(_criterion_lines(record))
            passed_count = sum(1 for c in record.criteria_results if c.passes)
            total = len(record.criteria_results)
            body_lines.append(typer.style(f"  criteria: {passed_count}/{total} passed", fg="red"))
        elif record.rationale:
            body_lines.append(typer.style(f"  {record.rationale}", fg="red"))
        blocks.append("\n".join([header, *body_lines]))

    return "\n\n".join(blocks)


def format_summary_line(records: list[RunRecord]) -> str:
    total = len(records)
    passed = sum(1 for record in records if record.passed)
    pct = round(100 * passed / total) if total else 0
    return f"{passed}/{total} passed ({pct}%)"
