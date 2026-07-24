from typing import Any, Literal

from pydantic import BaseModel


class AgentResult(BaseModel):
    """Outcome of a run_task() call.

    `steps_taken` counts agent perceive-decide-act turns actually taken (0 for
    dry_run, which never enters the loop). A reask-only turn triggered by a failed
    self-check does not increment it, since it doesn't consume a browsing step.
    """

    status: Literal["success", "validation_failed", "max_steps_exceeded", "dry_run"]
    answer: Any = None
    error: str | None = None
    attempts: int | None = None
    steps_taken: int = 0
