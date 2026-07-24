from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from evals.grading import grade_fixture
from evals.history import RunRecord
from evals.models import Fixture
from webagent.agent import run_task
from webagent.providers import check_model_config


async def run_fixture(
    fixture: Fixture,
    *,
    model: str,
    judge_model: str,
    thinking: str | bool = "medium",
    headless: bool,
    run_id: str,
) -> RunRecord:
    kwargs: dict[str, Any] = dict(
        task=fixture.task,
        url=fixture.url,
        output_schema=fixture.output_schema,
        output_description=fixture.output_description,
        model=model,
        thinking=thinking,
        headless=headless,
    )
    if fixture.max_steps is not None:
        kwargs["max_steps"] = fixture.max_steps
    if fixture.max_reask_attempts is not None:
        kwargs["max_reask_attempts"] = fixture.max_reask_attempts

    started = time.monotonic()
    try:
        result = await run_task(**kwargs)
    except Exception as exc:  # a crashed browser/model call fails the fixture, not the suite
        return RunRecord(
            run_id=run_id,
            run_at=datetime.now(timezone.utc),
            fixture_id=fixture.id,
            grading=fixture.grading,
            expected_status=fixture.expected_status,
            status="error",
            passed=False,
            rationale=str(exc),
            steps_taken=0,
            duration_seconds=time.monotonic() - started,
            model=model,
            judge_model=judge_model,
            error=str(exc),
        )

    grade = await grade_fixture(fixture, result, judge_model)
    return RunRecord(
        run_id=run_id,
        run_at=datetime.now(timezone.utc),
        fixture_id=fixture.id,
        grading=fixture.grading,
        expected_status=fixture.expected_status,
        status=result.status,
        passed=grade.passed,
        rationale=grade.rationale,
        criteria_results=grade.criteria_results,
        pass_fraction=grade.pass_fraction,
        steps_taken=result.steps_taken,
        duration_seconds=time.monotonic() - started,
        model=model,
        judge_model=judge_model,
        answer=result.answer,
        error=result.error,
    )


async def run_suite(
    fixtures: list[Fixture],
    *,
    model: str,
    judge_model: str,
    thinking: str | bool = "medium",
    concurrency: int = 1,
    headless: bool = True,
    on_complete: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    """Run every fixture, at most `concurrency` at a time.

    `on_complete`, if given, fires synchronously with each RunRecord the moment its
    fixture finishes - this is how the CLI streams pytest-style progress marks in
    real time regardless of concurrency. The returned list stays in original
    fixture order; `on_complete` fires in completion order (equal to fixture order
    when concurrency=1).
    """
    check_model_config(model)
    check_model_config(judge_model)

    run_id = uuid.uuid4().hex
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(fixture: Fixture) -> RunRecord:
        async with semaphore:
            record = await run_fixture(
                fixture,
                model=model,
                judge_model=judge_model,
                thinking=thinking,
                headless=headless,
                run_id=run_id,
            )
            if on_complete is not None:
                on_complete(record)
            return record

    return list(await asyncio.gather(*(_bounded(fixture) for fixture in fixtures)))
