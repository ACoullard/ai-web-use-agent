import asyncio
from types import SimpleNamespace

import pytest

import evals.grading as grading
from evals.grading import CriterionResult, RubricVerdict, grade_exact_match, grade_fixture, grade_llm_judge
from evals.models import Fixture
from webagent.result import AgentResult


def _fixture(**overrides):
    body = dict(id="f1", task="do it", url="https://example.com", grading="exact_match", expected={"x": 1})
    body.update(overrides)
    return Fixture(**body)


@pytest.mark.parametrize(
    "expected,answer,passed",
    [
        ({"pricing_url": "https://x/pricing"}, {"pricing_url": "https://x/pricing", "confidence": None}, True),
        ({"a": 1, "b": 2}, {"a": 1}, False),
        (5, 5, True),
        (5, 6, False),
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3], [3, 2, 1], False),
    ],
)
def test_grade_exact_match_partial_match_semantics(expected, answer, passed):
    result = grade_exact_match(_fixture(expected=expected), answer)
    assert result.passed is passed
    if not passed:
        assert result.rationale is not None


def test_grade_fixture_status_mismatch_short_circuits():
    fixture = _fixture()
    result = AgentResult(status="max_steps_exceeded", steps_taken=25)

    grade = asyncio.run(grade_fixture(fixture, result, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is False
    assert "max_steps_exceeded" in grade.rationale


def test_grade_fixture_non_success_expected_status_passes_without_expected_or_rubric():
    fixture = _fixture(expected=None, expected_status="max_steps_exceeded")
    result = AgentResult(status="max_steps_exceeded", steps_taken=1)

    grade = asyncio.run(grade_fixture(fixture, result, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is True
    assert grade.criteria_results is None


def test_grade_fixture_dispatches_to_exact_match():
    fixture = _fixture(expected={"x": 1})
    result = AgentResult(status="success", answer={"x": 1}, steps_taken=2)

    grade = asyncio.run(grade_fixture(fixture, result, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is True


def _monkeypatch_judge(monkeypatch, criteria):
    async def _fake_run(self, prompt, **kwargs):
        return SimpleNamespace(output=RubricVerdict(criteria=criteria))

    monkeypatch.setattr(grading.Agent, "run", _fake_run)


def test_grade_llm_judge_all_pass(monkeypatch):
    criteria = [
        CriterionResult(criterion="a", passes=True, reason="satisfied"),
        CriterionResult(criterion="b", passes=True, reason="satisfied"),
    ]
    _monkeypatch_judge(monkeypatch, criteria)
    fixture = _fixture(grading="llm_judge", expected=None, rubric=["a", "b"])

    grade = asyncio.run(grade_llm_judge(fixture, {"result": "answer"}, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is True
    assert grade.pass_fraction == 1.0
    assert grade.rationale is None


def test_grade_llm_judge_partial_pass_fails_overall_but_reports_fraction(monkeypatch):
    criteria = [
        CriterionResult(criterion="a", passes=True, reason="fine"),
        CriterionResult(criterion="b", passes=False, reason="missing detail"),
    ]
    _monkeypatch_judge(monkeypatch, criteria)
    fixture = _fixture(grading="llm_judge", expected=None, rubric=["a", "b"])

    grade = asyncio.run(grade_llm_judge(fixture, {"result": "answer"}, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is False
    assert grade.pass_fraction == 0.5
    assert "missing detail" in grade.rationale
    assert "fine" not in grade.rationale


def test_grade_llm_judge_all_fail(monkeypatch):
    criteria = [CriterionResult(criterion="a", passes=False, reason="nope")]
    _monkeypatch_judge(monkeypatch, criteria)
    fixture = _fixture(grading="llm_judge", expected=None, rubric=["a"])

    grade = asyncio.run(grade_llm_judge(fixture, {"result": "answer"}, judge_model="anthropic:claude-sonnet-5"))

    assert grade.passed is False
    assert grade.pass_fraction == 0.0
