from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator

GradingMode = Literal["exact_match", "llm_judge"]
ExpectedStatus = Literal["success", "validation_failed", "max_steps_exceeded"]


class Fixture(BaseModel):
    """A single eval case: a task/url to run through run_task(), and how to grade it.

    Exactly one of output_schema/output_description may be set, mirroring run_task()'s
    own mutual-exclusivity rule (checked here too, at load time rather than only at
    run time). For expected_status != "success" (the guardrail-path fixtures), only
    the resulting AgentResult.status is graded - expected/rubric are irrelevant and
    not required.
    """

    id: str
    task: str
    url: str
    is_live: bool = False

    output_schema: dict[str, Any] | None = None
    output_description: str | None = None

    grading: GradingMode
    expected: Any | None = None
    rubric: list[str] | None = None
    expected_status: ExpectedStatus = "success"

    max_steps: int | None = None
    max_reask_attempts: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> "Fixture":
        if self.output_schema is not None and self.output_description is not None:
            raise ValueError(f"{self.id}: pass at most one of output_schema, output_description")
        if self.expected_status == "success":
            if self.grading == "exact_match" and self.expected is None:
                raise ValueError(f"{self.id}: exact_match fixtures require `expected`")
            if self.grading == "llm_judge" and not self.rubric:
                raise ValueError(f"{self.id}: llm_judge fixtures require a non-empty `rubric` list")
        return self
