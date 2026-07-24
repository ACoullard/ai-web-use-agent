from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from evals.models import Fixture
from webagent.result import AgentResult

_JUDGE_SYSTEM_PROMPT = (
    "You are grading a web-browsing agent's final answer against a numbered checklist "
    "of criteria. Judge each criterion independently and strictly: a criterion passes "
    "only if the answer clearly satisfies it. Do not let a failure on one criterion "
    "bias your judgment of another. Return exactly one verdict per criterion, in the "
    "same order they were given, each with a concise `reason`."
)


class CriterionResult(BaseModel):
    criterion: str
    passes: bool
    reason: str


class RubricVerdict(BaseModel):
    criteria: list[CriterionResult]


class GradeResult(BaseModel):
    passed: bool
    rationale: str | None = None
    criteria_results: list[CriterionResult] | None = None

    @property
    def pass_fraction(self) -> float | None:
        if not self.criteria_results:
            return None
        return sum(1 for c in self.criteria_results if c.passes) / len(self.criteria_results)


def _partial_match(expected: Any, actual: Any) -> bool:
    """Recursive dict-subset match: every key in `expected` must be present and match
    in `actual`; extra keys in `actual` are ignored. Falls back to `==` for non-dict
    values (including lists - no fuzzy list matching).

    This is deliberately not full equality: json_schema_to_model gives every
    non-required schema property a `None` default, so a successful answer always
    dumps every schema property, including ones a fixture author never asserted on.
    """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _partial_match(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def _describe_contract(fixture: Fixture) -> str:
    if fixture.output_schema is not None:
        return f"a JSON object matching this schema: {json.dumps(fixture.output_schema)}"
    if fixture.output_description is not None:
        return f'a JSON object of the form {{"result": ...}}, where "result" satisfies: {fixture.output_description}'
    return "a freeform text answer"


def grade_exact_match(fixture: Fixture, answer: Any) -> GradeResult:
    passed = _partial_match(fixture.expected, answer)
    rationale = None if passed else f"expected (subset) {fixture.expected!r}, got {answer!r}"
    return GradeResult(passed=passed, rationale=rationale)


async def grade_llm_judge(fixture: Fixture, answer: Any, judge_model: str) -> GradeResult:
    judge: Agent[None, RubricVerdict] = Agent(
        judge_model,
        output_type=RubricVerdict,
        system_prompt=_JUDGE_SYSTEM_PROMPT,
    )
    checklist = "\n".join(f"{i + 1}. {criterion}" for i, criterion in enumerate(fixture.rubric or []))
    prompt = (
        f"Task: {fixture.task}\n"
        f"Output contract: {_describe_contract(fixture)}\n"
        f"Agent's answer: {json.dumps(answer, default=str)}\n\n"
        f"Criteria checklist:\n{checklist}"
    )
    verdict = await judge.run(prompt)
    criteria_results = verdict.output.criteria
    passed = all(c.passes for c in criteria_results)
    rationale = None
    if not passed:
        rationale = "; ".join(f"{c.criterion}: {c.reason}" for c in criteria_results if not c.passes)
    return GradeResult(passed=passed, rationale=rationale, criteria_results=criteria_results)


async def grade_fixture(fixture: Fixture, result: AgentResult, judge_model: str) -> GradeResult:
    if result.status != fixture.expected_status:
        return GradeResult(
            passed=False,
            rationale=f"expected status {fixture.expected_status!r}, got {result.status!r} (error={result.error!r})",
        )
    if fixture.expected_status != "success":
        return GradeResult(passed=True)
    if fixture.grading == "exact_match":
        return grade_exact_match(fixture, result.answer)
    return await grade_llm_judge(fixture, result.answer, judge_model)
