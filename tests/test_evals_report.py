from datetime import datetime, timezone

from evals.grading import CriterionResult
from evals.history import RunRecord
from evals.report import format_failure_section, format_progress_mark, format_summary_line


def _record(**overrides):
    body = dict(
        run_id="run-1",
        run_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
        fixture_id="f1",
        grading="exact_match",
        expected_status="success",
        status="success",
        passed=True,
        steps_taken=1,
        duration_seconds=1.0,
        model="anthropic:claude-sonnet-5",
    )
    body.update(overrides)
    return RunRecord(**body)


def test_format_progress_mark_pass_and_fail():
    assert "." in format_progress_mark(_record(passed=True))
    assert "F" in format_progress_mark(_record(passed=False))


def test_format_failure_section_empty_when_all_pass():
    assert format_failure_section([_record(passed=True)]) == ""


def test_format_failure_section_shows_rationale_for_exact_match_failure():
    record = _record(fixture_id="pricing-page-link", passed=False, rationale="expected (subset) {'x': 1}, got {}")

    section = format_failure_section([record])

    assert "FAIL pricing-page-link" in section
    assert "expected (subset)" in section


def test_format_failure_section_shows_criterion_breakdown_and_fraction():
    criteria = [
        CriterionResult(criterion="mentions watering", passes=True, reason="ok"),
        CriterionResult(criterion="mentions light", passes=False, reason="never mentioned"),
    ]
    record = _record(
        fixture_id="open-ended-summary-judge",
        grading="llm_judge",
        passed=False,
        criteria_results=criteria,
        pass_fraction=0.5,
    )

    section = format_failure_section([record])

    assert "FAIL open-ended-summary-judge" in section
    assert "[PASS] mentions watering" in section
    assert "[FAIL] mentions light" in section
    assert "never mentioned" in section
    assert "criteria: 1/2 passed" in section


def test_format_summary_line_counts_and_percentage():
    records = [_record(passed=True), _record(passed=True), _record(passed=False)]

    assert format_summary_line(records) == "2/3 passed (67%)"


def test_format_summary_line_empty_records():
    assert format_summary_line([]) == "0/0 passed (0%)"
