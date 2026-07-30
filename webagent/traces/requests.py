"""Opt-in capture of the exact input handed to the model on every request.

A `Trace`'s `Generation.input_prompt` holds only the *current* turn's user text - the
observation, plus any note prepended to it. That's a small fraction of what a request
actually carries: the system prompt, the whole accumulated message history (observations
stubbed exactly as the model saw them), and the tool definitions. On a 42-step run that
gap was 12% recorded vs. 100% billed, so a trace alone can't answer "what did the model
actually have in front of it at step 30?".

Set `TRACE_FULL_INPUT` and `CapturingModel` writes that missing 88% to a plain-text file.
It's a `WrapperModel`, so it sits between the agent and the provider and sees the same
three arguments the provider adapter does - that's the whole trick.

Each request is appended as it is sent, rather than buffered and written at the end. That
keeps the capture intact when a run crashes, which is when it's most worth having, and it
sidesteps an aliasing trap: the agent loop's `_stub_aged_observations` rewrites
`UserPromptPart.content` in place as steps age out, so anything holding message references
would retroactively show stubs that weren't there when the request went out. Writing at
request time is always faithful.

Off by default - the capture holds the full history for every request, so it runs several
times the size of the trace itself.
"""

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

_RULE = "=" * 88

# Part classes are a Pydantic AI detail; what the provider actually carries is a role. Name
# these the provider's way so a reader isn't decoding an abstraction the model never saw.
_ROLES = {
    "SystemPromptPart": "system",
    "UserPromptPart": "user",
    "TextPart": "assistant",
    "ThinkingPart": "assistant reasoning",
    "ToolCallPart": "assistant tool_call",
    "ToolReturnPart": "tool_result",
    "RetryPromptPart": "tool_result retry",
}


def _render_tool(tool: ToolDefinition) -> str:
    schema = json.dumps(tool.parameters_json_schema, indent=2)
    parts = (tool.name, (tool.description or "").strip(), schema)
    return "\n".join(p for p in parts if p)


def _role(part: Any) -> str:
    """The part's role as the provider receives it, falling back to the class name."""
    label = _ROLES.get(type(part).__name__, type(part).__name__)
    name = getattr(part, "tool_name", None)
    return f"{label} {name}" if name else label


def _part_body(part: Any) -> str:
    if isinstance(part, (ToolCallPart, ToolReturnPart)):
        payload = part.args if isinstance(part, ToolCallPart) else part.content
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, indent=2, default=str)
    content = getattr(part, "content", None)
    if content is None:
        return f"<{type(part).__name__} with no text content>"
    if isinstance(content, str):
        return content
    return "\n".join(str(chunk) for chunk in content)


def render_request(
    *,
    index: int,
    step: int,
    messages: Sequence[ModelMessage],
    tools: Sequence[ToolDefinition],
    instructions: Any = None,
) -> str:
    """Render one request as text. Empty `tools` omits that section entirely.

    A step can issue more than one request (Pydantic AI retries on output validation), so
    `index` and `step` are not 1:1 and both go in the header.
    """
    out = [_RULE, f"request {index}  ·  step {step}", _RULE, ""]

    if tools:
        out.append("tools:")
        out.extend(f"\n{_render_tool(t)}" for t in tools)
        out.append("")

    # Normally None: this codebase puts the task in a SystemPromptPart via `system_prompt=`.
    # Captured anyway so a switch to Pydantic AI's `instructions=` can't silently hide it.
    if instructions:
        out.append("instructions:")
        out.append(str(instructions))
        out.append("")

    for message in messages:
        for part in message.parts:
            out.append(f"{_role(part)}:")
            out.append(_part_body(part))
            out.append("")
    return "\n".join(out)


class CapturingModel(WrapperModel):
    """Appends the exact text of every outgoing request to `path`, then forwards it.

    Only constructed when capture is enabled (see `config.request_capture_path`), so there
    is no disabled state to carry: if this object exists, it writes.
    """

    def __init__(self, wrapped: Any, path: Path, current_step: Callable[[], int]) -> None:
        super().__init__(wrapped)
        self._path = path
        # Read at request time so it reports whichever step is currently running.
        self._current_step = current_step
        self._count = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: Any,
    ) -> ModelResponse:
        self._append(messages, model_request_parameters)
        return await super().request(messages, model_settings, model_request_parameters)

    def _append(self, messages: list[ModelMessage], params: Any) -> None:
        index = self._count
        self._count += 1
        # Tool definitions are fixed for a run - they come from the output union, built once -
        # so they go in the first section only rather than repeating on every request.
        tools = list(params.function_tools) + list(params.output_tools) if index == 0 else []
        text = render_request(
            index=index,
            step=self._current_step(),
            messages=messages,
            tools=tools,
            instructions=params.instruction_parts,
        )

        if index == 0:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on the first request so a re-used path can't leave stale sections behind.
        with self._path.open("w" if index == 0 else "a", encoding="utf-8") as handle:
            handle.write(text)


__all__ = ["CapturingModel", "render_request"]
