import json
import os
from datetime import datetime, timezone

from typer.testing import CliRunner

import evals.cli as evals_cli
import webagent.cli as cli
from evals.history import RunRecord

runner = CliRunner()


def _write_fixture(path, fixture_id, *, url="https://example.com/page"):
    path.write_text(
        f"""
id: {fixture_id}
task: "do something"
url: "{url}"
output_schema:
  type: object
  properties:
    x: {{type: string}}
grading: exact_match
expected:
  x: "y"
""",
        encoding="utf-8",
    )


def _fake_record(fixture_id, *, passed=True, rationale=None):
    return RunRecord(
        run_id="run-1",
        run_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
        fixture_id=fixture_id,
        grading="exact_match",
        expected_status="success",
        status="success" if passed else "success",
        passed=passed,
        rationale=rationale,
        steps_taken=1,
        duration_seconds=0.1,
        model="anthropic:claude-sonnet-5",
    )


def _mock_run_suite(records_by_id=None, all_passed=True):
    async def fake_run_suite(fixtures, *, model, judge_model, concurrency, headless, on_complete=None):
        fake_run_suite.received = dict(
            fixtures=fixtures, model=model, judge_model=judge_model, concurrency=concurrency, headless=headless
        )
        records = [
            (records_by_id or {}).get(f.id) or _fake_record(f.id, passed=all_passed)
            for f in fixtures
        ]
        for record in records:
            if on_complete is not None:
                on_complete(record)
        return records

    fake_run_suite.received = None
    return fake_run_suite


def test_no_fixtures_matched_exits_one(tmp_path):
    result = runner.invoke(cli.app, ["evals", "run", "--fixtures", str(tmp_path), "--history", str(tmp_path / "h.jsonl")])

    assert result.exit_code == 1
    assert "No fixtures matched" in result.output


def test_all_pass_exits_zero_and_writes_history(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")

    monkeypatch.setattr(evals_cli, "run_suite", _mock_run_suite(all_passed=True))
    history_path = tmp_path / "history.jsonl"

    result = runner.invoke(
        cli.app, ["evals", "run", "--fixtures", str(tmp_path), "--history", str(history_path)]
    )

    assert result.exit_code == 0
    assert "1/1 passed (100%)" in result.output
    assert history_path.exists()
    assert "fixture-a" in history_path.read_text(encoding="utf-8")


def test_running_count_and_source_path_announced_before_running(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")
    _write_fixture(local_dir / "b.yaml", "fixture-b")

    monkeypatch.setattr(evals_cli, "run_suite", _mock_run_suite(all_passed=True))

    result_both = runner.invoke(
        cli.app, ["evals", "run", "--fixtures", str(tmp_path), "--history", str(tmp_path / "h1.jsonl")]
    )
    assert f"Running 2 fixtures from {os.path.relpath(tmp_path)}" in result_both.output

    result_one = runner.invoke(
        cli.app,
        ["evals", "run", "local/a.yaml", "--fixtures-root", str(tmp_path), "--history", str(tmp_path / "h2.jsonl")],
    )
    assert f"Running 1 fixture from {os.path.relpath(local_dir / 'a.yaml')}\n" in result_one.output


def test_running_message_shows_source_path_relative_to_cwd(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")

    monkeypatch.setattr(evals_cli, "run_suite", _mock_run_suite(all_passed=True))

    result = runner.invoke(
        cli.app,
        [
            "evals", "run", "local/a.yaml",
            "--fixtures-root", str(tmp_path),
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    expected_path = os.path.relpath(local_dir / "a.yaml")
    assert f"Running 1 fixture from {expected_path}" in result.output


def test_failure_exits_one_and_prints_failure_section(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")

    records_by_id = {"fixture-a": _fake_record("fixture-a", passed=False, rationale="expected x=y, got x=z")}
    monkeypatch.setattr(evals_cli, "run_suite", _mock_run_suite(records_by_id))

    result = runner.invoke(
        cli.app,
        ["evals", "run", "--fixtures", str(tmp_path), "--history", str(tmp_path / "history.jsonl")],
    )

    assert result.exit_code == 1
    assert "FAIL fixture-a" in result.output
    assert "expected x=y, got x=z" in result.output


def test_live_fixtures_excluded_by_default_and_included_with_flag(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    live_dir = tmp_path / "live"
    local_dir.mkdir()
    live_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-local")
    _write_fixture(live_dir / "b.yaml", "fixture-live")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    result = runner.invoke(
        cli.app,
        ["evals", "run", "--fixtures", str(tmp_path), "--history", str(tmp_path / "history.jsonl")],
    )
    assert result.exit_code == 0
    assert [f.id for f in mock.received["fixtures"]] == ["fixture-local"]

    result_live = runner.invoke(
        cli.app,
        ["evals", "run", "--fixtures", str(tmp_path), "--live", "--history", str(tmp_path / "history2.jsonl")],
    )
    assert result_live.exit_code == 0
    assert sorted(f.id for f in mock.received["fixtures"]) == ["fixture-live", "fixture-local"]


def test_judge_model_defaults_to_model(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    runner.invoke(
        cli.app,
        [
            "evals", "run",
            "--fixtures", str(tmp_path),
            "--model", "openai:gpt-4o",
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    assert mock.received["model"] == "openai:gpt-4o"
    assert mock.received["judge_model"] == "openai:gpt-4o"


def test_report_option_writes_json_file(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")

    monkeypatch.setattr(evals_cli, "run_suite", _mock_run_suite(all_passed=True))
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        cli.app,
        [
            "evals", "run",
            "--fixtures", str(tmp_path),
            "--report", str(report_path),
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload[0]["fixture_id"] == "fixture-a"
    assert payload[0]["passed"] is True


def test_positional_single_file_path_relative_to_fixtures_root(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")
    _write_fixture(local_dir / "b.yaml", "fixture-b")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    result = runner.invoke(
        cli.app,
        [
            "evals", "run", "local/a.yaml",
            "--fixtures-root", str(tmp_path),
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert [f.id for f in mock.received["fixtures"]] == ["fixture-a"]


def test_positional_folder_path_runs_everything_under_it(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    other_dir = tmp_path / "other"
    local_dir.mkdir()
    other_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")
    _write_fixture(local_dir / "b.yaml", "fixture-b")
    _write_fixture(other_dir / "c.yaml", "fixture-c")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    result = runner.invoke(
        cli.app,
        ["evals", "run", "local", "--fixtures-root", str(tmp_path), "--history", str(tmp_path / "history.jsonl")],
    )

    assert result.exit_code == 0
    assert sorted(f.id for f in mock.received["fixtures"]) == ["fixture-a", "fixture-b"]


def test_positional_absolute_path_bypasses_fixtures_root(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    fixture_path = local_dir / "a.yaml"
    _write_fixture(fixture_path, "fixture-a")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    result = runner.invoke(
        cli.app,
        ["evals", "run", str(fixture_path), "--history", str(tmp_path / "history.jsonl")],
    )

    assert result.exit_code == 0
    assert [f.id for f in mock.received["fixtures"]] == ["fixture-a"]


def test_multiple_positional_paths_are_merged(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", "fixture-a")
    _write_fixture(local_dir / "b.yaml", "fixture-b")
    _write_fixture(local_dir / "c.yaml", "fixture-c")

    mock = _mock_run_suite()
    monkeypatch.setattr(evals_cli, "run_suite", mock)

    result = runner.invoke(
        cli.app,
        [
            "evals", "run", "local/a.yaml", "local/c.yaml",
            "--fixtures-root", str(tmp_path),
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert sorted(f.id for f in mock.received["fixtures"]) == ["fixture-a", "fixture-c"]


def test_nonexistent_positional_path_exits_one_with_message(tmp_path):
    result = runner.invoke(
        cli.app,
        [
            "evals", "run", "local/does-not-exist.yaml",
            "--fixtures-root", str(tmp_path),
            "--history", str(tmp_path / "history.jsonl"),
        ],
    )

    assert result.exit_code == 1
    assert "not found" in result.output


def test_history_command_prints_pass_rate(tmp_path):
    history_path = tmp_path / "history.jsonl"
    from evals.history import append_history

    append_history(history_path, [_fake_record("fixture-a", passed=True)])

    result = runner.invoke(cli.app, ["evals", "history", "--history", str(history_path)])

    assert result.exit_code == 0
    assert "100%" in result.output


def test_history_command_with_no_history_reports_none(tmp_path):
    result = runner.invoke(cli.app, ["evals", "history", "--history", str(tmp_path / "missing.jsonl")])

    assert result.exit_code == 0
    assert "No history recorded" in result.output
