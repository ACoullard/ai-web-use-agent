"""Durable, human- and AI-legible traces of an agent run.

A `Trace` is one run: filterable metadata (task, model, thinking effort, outcome, ...)
plus an ordered list of `Observation`s, of which there are exactly two kinds:

- `Generation`: an LLM call (deciding the next action, or the output self-check).
- `ToolCall`: a browser action execution (click/type/select/...) and its result.

A terminal `finish` action is not a browser action, so it appears as the `output` of the
final `Generation` rather than as a `ToolCall`.

Field names track OpenTelemetry's GenAI vocabulary where they map cleanly (`input_tokens`,
`output_tokens`, `finish_reason`, `provider_response_id`), but the shape is a plain typed
model, not raw spans.
"""

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Union

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    RetryPromptPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)

logger = logging.getLogger(__name__)


class Generation(BaseModel):
    """One LLM call within a run."""

    type: Literal["generation"] = "generation"
    name: str  # "decide_action" | "self_check"
    step: int
    duration_seconds: float
    model: str
    input_prompt: str  # the user-facing prompt sent this turn (system prompt lives on Trace)
    reasoning: str | None = None
    reasoning_encrypted: bool = False  # model reasoned but provider returned it encrypted
    output: Any = None  # the chosen action / verdict (or attempted output on a failed retry)
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None  # present even when reasoning text is withheld
    finish_reason: str | None = None
    provider_response_id: str | None = None


class ToolCall(BaseModel):
    """One browser action execution within a run."""

    type: Literal["tool"] = "tool"
    name: str  # action.type, e.g. "click"
    step: int
    args: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float
    status: Literal["ok", "error"]
    result: str | None = None  # e.g. a search-page-text hit fed back to the model
    error: str | None = None


Observation = Annotated[Union[Generation, ToolCall], Field(discriminator="type")]


class Trace(BaseModel):
    """One agent run: filterable metadata + ordered observations."""

    schema_version: int = 1
    trace_id: str
    created_at: datetime
    # --- filterable metadata ---
    task: str
    url: str
    model: str
    thinking: str
    output_mode: Literal["schema", "description", "freeform"]
    fixture_id: str | None = None
    run_id: str | None = None
    # --- outcome ---
    status: str
    steps_taken: int = 0
    duration_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # --- content ---
    system_prompt: str = ""
    observations: list[Observation] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Metadata only, dropping the heavy content fields - for listing many traces at once."""
        return self.model_dump(mode="json", exclude={"observations", "system_prompt"})


# --------------------------------------------------------------------------- helpers


def _request_user_text(request: ModelRequest) -> str | None:
    """The user-facing text of a ModelRequest (user prompt + any retry note), sans system prompt."""
    chunks: list[str] = []
    for part in request.parts:
        if isinstance(part, UserPromptPart):
            content = part.content
            chunks.append(content if isinstance(content, str) else "\n".join(str(c) for c in content))
        elif isinstance(part, RetryPromptPart):
            model_response = getattr(part, "model_response", None)
            chunks.append(model_response() if callable(model_response) else str(part.content))
    text = "\n\n".join(c for c in chunks if c)
    return text or None


def _reasoning(response: ModelResponse) -> tuple[str | None, bool]:
    """(joined thinking text or None, was-encrypted). Encrypted = a thinking part with a
    signature but no readable content (e.g. OpenAI reasoning models)."""
    thinking_parts = [p for p in response.parts if isinstance(p, ThinkingPart)]
    text = "\n\n".join(p.content for p in thinking_parts if p.content)
    encrypted = any(not p.content and p.signature for p in thinking_parts)
    return (text or None), encrypted


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "details", None) or {}
    for key, value in details.items():
        if "reasoning" in key:
            return value
    return None


def _attempted_output(response: ModelResponse) -> Any:
    """The tool call a (failed, intermediate) response attempted, for retry visibility."""
    for part in response.parts:
        if isinstance(part, (ToolCallPart, NativeToolCallPart)):
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            return {"tool_name": part.tool_name, "args": args}
    return None


# --------------------------------------------------------------------------- recorder


class TraceRecorder:
    """Accumulates observations for one run and builds the final `Trace`.

    Obtained from `Tracer.start()`, which supplies the `persist` hook deciding where the
    finished trace goes (or that it goes nowhere). One instance per run, so concurrent
    runs never share mutable state.
    """

    def __init__(
        self,
        *,
        task: str,
        url: str,
        model: str,
        thinking: str,
        output_mode: Literal["schema", "description", "freeform"],
        system_prompt: str,
        fixture_id: str | None = None,
        run_id: str | None = None,
        persist: Callable[[Trace], None] | None = None,
    ) -> None:
        self._persist = persist
        self.trace_id = uuid.uuid4().hex
        self.created_at = datetime.now(timezone.utc)
        self.task = task
        self.url = url
        self.model = model
        self.thinking = thinking
        self.output_mode = output_mode
        self.system_prompt = system_prompt
        self.fixture_id = fixture_id
        self.run_id = run_id
        self.observations: list[Observation] = []

    def record_generation(self, step: int, input_prompt: str, result: Any, duration: float) -> None:
        """Record the decide-next-action call."""
        self._record_llm_call("decide_action", step, input_prompt, result, duration)

    def record_self_check(self, step: int, input_prompt: str, result: Any, duration: float) -> None:
        """Record the output self-check LLM call. It's an `agent.run()` like any other, so its
        reasoning, verdict and token counts are read off the result the same way."""
        self._record_llm_call("self_check", step, input_prompt, result, duration)

    def _record_llm_call(self, name: str, step: int, input_prompt: str, result: Any, duration: float) -> None:
        """Turn one `agent.run()` result into `Generation`s. A single run can produce more than
        one `ModelResponse` when Pydantic AI retries on output validation; we record one
        `Generation` per response so retries are visible, the last carrying the final output."""
        messages = result.new_messages()
        last_request_text = input_prompt
        pairs: list[tuple[str, ModelResponse]] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                text = _request_user_text(message)
                if text is not None:
                    last_request_text = text
            elif isinstance(message, ModelResponse):
                pairs.append((last_request_text, message))

        for i, (req_text, response) in enumerate(pairs):
            is_final = i == len(pairs) - 1
            reasoning, encrypted = _reasoning(response)
            if is_final:
                output = result.output.model_dump(mode="json") if hasattr(result.output, "model_dump") else result.output
            else:
                output = _attempted_output(response)
            self.observations.append(
                Generation(
                    name=name,
                    step=step,
                    # the whole turn's latency lands on the response that actually produced
                    # the accepted output; intermediate retries within it aren't separately timed
                    duration_seconds=duration if is_final else 0.0,
                    model=response.model_name or self.model,
                    input_prompt=req_text,
                    reasoning=reasoning,
                    reasoning_encrypted=encrypted,
                    output=output,
                    input_tokens=response.usage.input_tokens or None,
                    output_tokens=response.usage.output_tokens or None,
                    reasoning_tokens=_reasoning_tokens(response.usage),
                    finish_reason=response.finish_reason,
                    provider_response_id=response.provider_response_id,
                )
            )

    def record_tool(
        self,
        action: Any,
        step: int,
        duration: float,
        *,
        status: Literal["ok", "error"],
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        args = action.model_dump(mode="json")
        args.pop("type", None)
        self.observations.append(
            ToolCall(
                name=action.type,
                step=step,
                args=args,
                duration_seconds=duration,
                status=status,
                result=result,
                error=error,
            )
        )

    def finish(self, *, status: str, steps_taken: int, duration: float) -> Trace:
        """Build the final `Trace` and hand it to the tracer's persist hook (if any)."""
        generations = [o for o in self.observations if isinstance(o, Generation)]
        trace = Trace(
            trace_id=self.trace_id,
            created_at=self.created_at,
            task=self.task,
            url=self.url,
            model=self.model,
            thinking=self.thinking,
            output_mode=self.output_mode,
            fixture_id=self.fixture_id,
            run_id=self.run_id,
            status=status,
            steps_taken=steps_taken,
            duration_seconds=duration,
            total_input_tokens=sum(g.input_tokens or 0 for g in generations),
            total_output_tokens=sum(g.output_tokens or 0 for g in generations),
            system_prompt=self.system_prompt,
            observations=self.observations,
        )
        if self._persist is not None:
            self._persist(trace)
        return trace


# --------------------------------------------------------------------------- tracers


class Recording(Protocol):
    """What a run writes observations to. `trace_id` is None when nothing is recorded."""

    trace_id: str | None

    def record_generation(self, step: int, input_prompt: str, result: Any, duration: float) -> None: ...

    def record_self_check(self, step: int, input_prompt: str, result: Any, duration: float) -> None: ...

    def record_tool(
        self,
        action: Any,
        step: int,
        duration: float,
        *,
        status: Literal["ok", "error"],
        result: str | None = None,
        error: str | None = None,
    ) -> None: ...

    def finish(self, *, status: str, steps_taken: int, duration: float) -> Trace | None: ...


class Tracer(Protocol):
    """Where traces go, decided once by the caller and injected into the run.

    Like a logger: the caller invokes `start()` and records into the returned `Recording`
    without knowing whether anything is written, or where. `with_labels()` derives a new
    tracer carrying extra filterable metadata (never mutating the original), so labels can
    be stamped at one level and apply to every run started beneath it.
    """

    def start(
        self,
        *,
        task: str,
        url: str,
        model: str,
        thinking: str,
        output_mode: Literal["schema", "description", "freeform"],
        system_prompt: str,
    ) -> Recording: ...

    def with_labels(self, **labels: str | None) -> "Tracer": ...


class NullRecording:
    """No-op `Recording`: drops every observation. The off-switch, branch-free."""

    trace_id: str | None = None

    def record_generation(self, step: int, input_prompt: str, result: Any, duration: float) -> None:
        pass

    def record_self_check(self, step: int, input_prompt: str, result: Any, duration: float) -> None:
        pass

    def record_tool(self, action: Any, step: int, duration: float, **kwargs: Any) -> None:
        pass

    def finish(self, *, status: str, steps_taken: int, duration: float) -> None:
        return None


class NullTracer:
    """Tracer that records nothing: the off-switch for tracing."""

    def start(self, **metadata: Any) -> Recording:
        return NullRecording()

    def with_labels(self, **labels: str | None) -> "NullTracer":
        return self


class FileTracer:
    """Tracer that writes each finished trace as one JSON file under `directory`."""

    def __init__(self, directory: Path, **labels: str | None) -> None:
        self.directory = directory
        self.labels = {k: v for k, v in labels.items() if v is not None}

    def start(self, **metadata: Any) -> Recording:
        return TraceRecorder(**metadata, **self.labels, persist=self._save)

    def with_labels(self, **labels: str | None) -> "FileTracer":
        return FileTracer(self.directory, **{**self.labels, **labels})

    def _save(self, trace: Trace) -> None:
        logger.info("trace saved to %s", save_trace(trace, self.directory))


# --------------------------------------------------------------------------- storage


def save_trace(trace: Trace, directory: Path) -> Path:
    """Write `trace` as one pretty-printed JSON file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace.created_at:%Y%m%dT%H%M%S}-{trace.trace_id[:8]}.json"
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_traces(directory: Path) -> list[Trace]:
    """Load every trace under `directory` (recursively), newest first.

    Recursion lets one traces root hold several subdirectories of runs and still be
    browsed as a single collection. Malformed files are skipped.
    """
    if not directory.exists():
        return []
    traces: list[Trace] = []
    for path in directory.rglob("*.json"):
        try:
            traces.append(Trace.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValidationError, ValueError, OSError):
            continue
    traces.sort(key=lambda t: t.created_at, reverse=True)
    return traces


def filter_traces(
    traces: list[Trace],
    *,
    task: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    status: str | None = None,
    fixture_id: str | None = None,
    run_id: str | None = None,
) -> list[Trace]:
    """In-memory filter. `task` is a case-insensitive substring; the rest are exact."""

    def keep(trace: Trace) -> bool:
        if task and task.lower() not in trace.task.lower():
            return False
        if model and trace.model != model:
            return False
        if thinking and trace.thinking != thinking:
            return False
        if status and trace.status != status:
            return False
        if fixture_id and trace.fixture_id != fixture_id:
            return False
        if run_id and trace.run_id != run_id:
            return False
        return True

    return [trace for trace in traces if keep(trace)]


def find_trace(traces: list[Trace], id_or_prefix: str) -> Trace | None:
    """Resolve a trace by exact id, or by an id-prefix if it matches exactly one trace."""
    exact = [t for t in traces if t.trace_id == id_or_prefix]
    if exact:
        return exact[0]
    matches = [t for t in traces if t.trace_id.startswith(id_or_prefix)]
    return matches[0] if len(matches) == 1 else None
