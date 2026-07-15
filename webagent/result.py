from typing import Any, Literal

from pydantic import BaseModel


class AgentResult(BaseModel):
    """Outcome of a run_task() call.

    `status` is intentionally a small, open-ended set for now - step 4
    (Guardrails) will add "blocked"/"timeout"/etc. and richer fields
    (step history) on top of this same type.
    """

    status: Literal["success", "validation_failed", "max_steps_exceeded"]
    answer: Any = None
    error: str | None = None
    attempts: int | None = None
