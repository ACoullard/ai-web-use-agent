import asyncio

import pytest

import evals.runner as runner_mod
from evals.models import Fixture
from evals.runner import run_suite
from webagent.result import AgentResult
from webagent.trace import FileTracer, load_traces


def _fixture(fixture_id: str) -> Fixture:
    return Fixture(
        id=fixture_id,
        task="do something",
        url="https://example.com",
        output_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        grading="exact_match",
        expected={"x": "y"},
    )


@pytest.fixture
def stub_run_task(monkeypatch):
    """Replace run_task with a stub that drives the injected tracer like the real loop."""

    async def fake_run_task(*, tracer, task, url, model, **kwargs):
        recording = tracer.start(
            task=task,
            url=url,
            model=model,
            thinking="medium",
            output_mode="schema",
            system_prompt="you are an agent",
        )
        recording.finish(status="success", steps_taken=1, duration=0.1)
        return AgentResult(status="success", answer={"x": "y"}, steps_taken=1, trace_id=recording.trace_id)

    monkeypatch.setattr(runner_mod, "run_task", fake_run_task)
    monkeypatch.setattr(runner_mod, "check_model_config", lambda model: None)


def test_run_suite_stamps_run_id_at_suite_level_and_fixture_id_per_fixture(stub_run_task, tmp_path):
    records = asyncio.run(
        run_suite(
            [_fixture("fixture-a"), _fixture("fixture-b")],
            model="test:model",
            judge_model="test:model",
            concurrency=2,
            tracer=FileTracer(tmp_path),
        )
    )

    assert all(record.passed for record in records)
    traces = load_traces(tmp_path)
    assert sorted(t.fixture_id for t in traces) == ["fixture-a", "fixture-b"]
    # one run_id, stamped once at suite level, shared by every fixture's trace
    run_ids = {t.run_id for t in traces}
    assert run_ids == {records[0].run_id} != {None}


def test_run_suite_without_tracer_writes_nothing(stub_run_task, tmp_path):
    records = asyncio.run(
        run_suite([_fixture("fixture-a")], model="test:model", judge_model="test:model")
    )

    assert records[0].passed
    assert load_traces(tmp_path) == []
