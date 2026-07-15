import json

from typer.testing import CliRunner

import webagent.cli as cli
from webagent.result import AgentResult

runner = CliRunner()


def _mock_run_task(status="success", **kwargs):
    async def fake_run_task(**call_kwargs):
        fake_run_task.received = call_kwargs
        return AgentResult(status=status, **kwargs)

    fake_run_task.received = None
    return fake_run_task


def test_missing_required_args_exits_nonzero():
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code != 0


def test_schema_and_description_mutually_exclusive(monkeypatch, tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object", "properties": {}}))
    monkeypatch.setattr(cli, "run_task", _mock_run_task())

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--task", "do something",
            "--url", "https://example.com",
            "--schema", str(schema_path),
            "--description", "a description",
        ],
    )

    assert result.exit_code == 2
    assert "at most one" in result.output.lower()


def test_success_exits_zero_and_prints_json(monkeypatch):
    mock = _mock_run_task(status="success", answer={"foo": "bar"})
    monkeypatch.setattr(cli, "run_task", mock)

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    assert payload["answer"] == {"foo": "bar"}


def test_validation_failed_exits_one(monkeypatch):
    monkeypatch.setattr(cli, "run_task", _mock_run_task(status="validation_failed", error="nope"))

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com"],
    )

    assert result.exit_code == 1


def test_max_steps_exceeded_exits_two(monkeypatch):
    monkeypatch.setattr(cli, "run_task", _mock_run_task(status="max_steps_exceeded"))

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com"],
    )

    assert result.exit_code == 2


def test_dry_run_passes_flag_through_and_exits_zero(monkeypatch):
    mock = _mock_run_task(
        status="dry_run",
        answer={"system_prompt": "you are an agent", "observation_prompt": "page stuff"},
    )
    monkeypatch.setattr(cli, "run_task", mock)

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com", "--dry-run"],
    )

    assert result.exit_code == 0
    assert mock.received["dry_run"] is True
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert "system_prompt" in payload["answer"]


def test_schema_reads_json_file_into_output_schema(monkeypatch, tmp_path):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    mock = _mock_run_task(status="success", answer={"x": "y"})
    monkeypatch.setattr(cli, "run_task", mock)

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com", "--schema", str(schema_path)],
    )

    assert result.exit_code == 0
    assert mock.received["output_schema"] == schema
    assert mock.received["output_description"] is None


def test_schema_path_must_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "run_task", _mock_run_task())
    missing = tmp_path / "does-not-exist.json"

    result = runner.invoke(
        cli.app,
        ["run", "--task", "do something", "--url", "https://example.com", "--schema", str(missing)],
    )

    assert result.exit_code != 0
