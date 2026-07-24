from datetime import datetime, timezone

from evals.grading import CriterionResult
from evals.history import RunRecord, append_history, load_history, pass_rate_by_run


def _record(**overrides):
    body = dict(
        run_id="run-1",
        run_at=datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc),
        fixture_id="f1",
        grading="exact_match",
        expected_status="success",
        status="success",
        passed=True,
        steps_taken=3,
        duration_seconds=1.5,
        model="anthropic:claude-sonnet-5",
    )
    body.update(overrides)
    return RunRecord(**body)


def test_append_and_load_round_trip(tmp_path):
    history_path = tmp_path / "history.jsonl"
    records = [_record(fixture_id="f1"), _record(fixture_id="f2", passed=False, rationale="nope")]

    append_history(history_path, records)
    loaded = load_history(history_path)

    assert [r.fixture_id for r in loaded] == ["f1", "f2"]
    assert loaded[1].rationale == "nope"


def test_append_history_extends_rather_than_overwrites(tmp_path):
    history_path = tmp_path / "history.jsonl"

    append_history(history_path, [_record(fixture_id="f1")])
    append_history(history_path, [_record(fixture_id="f2")])

    loaded = load_history(history_path)
    assert [r.fixture_id for r in loaded] == ["f1", "f2"]


def test_round_trips_criteria_results_and_pass_fraction(tmp_path):
    history_path = tmp_path / "history.jsonl"
    criteria = [CriterionResult(criterion="a", passes=True, reason="ok")]
    record = _record(grading="llm_judge", criteria_results=criteria, pass_fraction=1.0)

    append_history(history_path, [record])
    loaded = load_history(history_path)

    assert loaded[0].criteria_results[0].criterion == "a"
    assert loaded[0].pass_fraction == 1.0


def test_load_history_missing_file_returns_empty_list(tmp_path):
    assert load_history(tmp_path / "does-not-exist.jsonl") == []


def test_pass_rate_by_run_aggregates_chronologically():
    early = datetime(2026, 7, 1, tzinfo=timezone.utc)
    later = datetime(2026, 7, 15, tzinfo=timezone.utc)
    records = [
        _record(run_id="run-a", run_at=early, passed=True),
        _record(run_id="run-a", run_at=early, passed=False),
        _record(run_id="run-b", run_at=later, passed=True),
    ]

    rates = pass_rate_by_run(records)

    assert rates == [("run-a", early, 0.5), ("run-b", later, 1.0)]
