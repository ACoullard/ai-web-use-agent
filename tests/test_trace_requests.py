"""Opt-in capture of the exact model input (`TRACE_FULL_INPUT`).

`CapturingModel` is a Pydantic AI `WrapperModel` that appends each outgoing request to a
file before forwarding it - see webagent/traces/requests.py. Two properties matter:

- it writes *per request*, so a crashed run still leaves everything sent up to the failure;
- it writes at request time, so the agent loop's in-place observation stubbing can't
  retroactively rewrite what the file says was sent.
"""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition

from webagent.config import request_capture_path
from webagent.traces import CapturingModel


_TOOLS = [
    ToolDefinition(
        name="final_result_ClickAction",
        description="Click an interactive element.",
        parameters_json_schema={"properties": {"index": {"type": "integer"}}, "type": "object"},
    ),
    ToolDefinition(
        name="final_result_FinishAction",
        description="Report the final answer.",
        parameters_json_schema={"properties": {"answer": {"type": "string"}}, "type": "object"},
    ),
]


def _Params(tools=(), instructions=None) -> ModelRequestParameters:
    """The real parameters object, since the wrapped model consumes it too."""
    return ModelRequestParameters(
        function_tools=list(tools),
        output_tools=[],
        instruction_parts=instructions,
    )


def _messages(observation: str) -> list:
    return [
        ModelRequest(
            parts=[SystemPromptPart(content="SYSTEM: do the task"), UserPromptPart(content=observation)]
        ),
        ModelResponse(
            parts=[ThinkingPart(content="thinking"), ToolCallPart("final_result_ClickAction", {"index": 1})]
        ),
        ModelRequest(parts=[ToolReturnPart("final_result_ClickAction", "Final result processed.")]),
    ]


def _model(path: Path, step=lambda: 0) -> CapturingModel:
    return CapturingModel(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])), path, step)


def _send(model: CapturingModel, observation: str, tools=_TOOLS, instructions=None) -> None:
    asyncio.run(model.request(_messages(observation), None, _Params(tools, instructions)))


# --- what lands in the file ------------------------------------------------------


def test_capture_includes_system_prompt_tools_and_every_part(tmp_path: Path):
    path = tmp_path / "cap.txt"
    _send(_model(path), "Page title: Home")

    text = path.read_text(encoding="utf-8")
    assert "SYSTEM: do the task" in text  # the thing Generation.input_prompt omits
    assert "Page title: Home" in text
    assert "final_result_ClickAction" in text and "final_result_FinishAction" in text
    assert '"type": "integer"' in text  # the tool JSON schema, invisible in the trace
    assert "thinking" in text
    assert "Final result processed." in text
    assert "tools:" in text


def test_parts_are_labelled_with_provider_roles_not_pydantic_class_names(tmp_path: Path):
    """The point of the capture is reading what the model got, so the scaffolding around the
    content stays as thin as the provider's own vocabulary."""
    path = tmp_path / "cap.txt"
    _send(_model(path), "Page title: Home")

    text = path.read_text(encoding="utf-8")
    for role in ("system:", "user:", "assistant reasoning:", "assistant tool_call ", "tool_result "):
        assert role in text, role
    assert "ModelRequest" not in text and "ModelResponse" not in text
    assert "PromptPart" not in text and "ToolCallPart" not in text


def test_no_invented_part_numbering_to_collide_with_element_indices(tmp_path: Path):
    """Observations contain their own `[1] <a> ...` element indices; a second, unrelated
    bracketed numbering over parts made the two indistinguishable when reading or grepping."""
    path = tmp_path / "cap.txt"
    _send(_model(path), "Interactive elements:\n[1] <a> 'Home'\n[2] <a> 'Cart'")

    brackets = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("[")]
    assert brackets == ["[1] <a> 'Home'", "[2] <a> 'Cart'"]  # page content only


def test_the_first_request_truncates_a_stale_file(tmp_path: Path):
    """Sections are appended, so a re-used path must not keep a previous run's output."""
    path = tmp_path / "cap.txt"
    path.write_text("LEFTOVER from an earlier run", encoding="utf-8")

    model = _model(path)
    _send(model, "Page title: A")
    _send(model, "Page title: B")

    text = path.read_text(encoding="utf-8")
    assert "LEFTOVER" not in text
    assert text.startswith("=")  # straight into request 0's divider, no preamble
    assert "Page title: A" in text and "Page title: B" in text


def test_step_and_request_numbers_are_both_recorded(tmp_path: Path):
    """A step can issue several requests, so neither number alone identifies a section."""
    path = tmp_path / "cap.txt"
    step = {"n": 7}
    model = _model(path, lambda: step["n"])

    _send(model, "Page title: A")
    _send(model, "Page title: B")  # same step: an output-validation retry
    step["n"] = 8
    _send(model, "Page title: C")

    text = path.read_text(encoding="utf-8")
    assert "request 0  ·  step 7" in text
    assert "request 1  ·  step 7" in text
    assert "request 2  ·  step 8" in text


def test_instructions_are_captured_when_present(tmp_path: Path):
    """Normally None here, but a switch to Pydantic AI's `instructions=` must not hide the task."""
    path = tmp_path / "cap.txt"
    _send(_model(path), "Page title: Home", instructions=[InstructionPart(content="INSTRUCTIONS: the task")])

    assert "INSTRUCTIONS: the task" in path.read_text(encoding="utf-8")


def test_tool_definitions_appear_in_the_first_section_only(tmp_path: Path):
    """They're fixed for a run, so repeating them on every request is pure noise."""
    path = tmp_path / "cap.txt"
    model = _model(path)
    for _ in range(3):
        _send(model, "Page title: Home")

    text = path.read_text(encoding="utf-8")
    assert text.count('"type": "integer"') == 1  # full schemas exactly once
    assert text.count("tools:") == 1
    assert text.count("Page title: Home") == 3  # messages always in full


# --- the two properties that motivated the design --------------------------------


def test_each_request_is_flushed_before_the_next_so_a_crash_keeps_what_was_sent(tmp_path: Path):
    path = tmp_path / "cap.txt"
    model = _model(path)

    _send(model, "Page title: A")
    assert "Page title: A" in path.read_text(encoding="utf-8")  # on disk already

    _send(model, "Page title: B")
    text = path.read_text(encoding="utf-8")
    assert "Page title: A" in text and "Page title: B" in text


def test_written_at_request_time_so_later_mutation_cannot_rewrite_it(tmp_path: Path):
    """The agent loop rewrites UserPromptPart.content in place as observations age out."""
    path = tmp_path / "cap.txt"
    messages = _messages("Page title: Home\nURL: u\nlots of page text")

    asyncio.run(_model(path).request(messages, None, _Params(_TOOLS)))
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                part.content = "[Page state from an earlier step, omitted to save context.]"

    text = path.read_text(encoding="utf-8")
    assert "lots of page text" in text
    assert "omitted to save context" not in text


def test_response_is_passed_through_unchanged(tmp_path: Path):
    response = asyncio.run(_model(tmp_path / "cap.txt").request(_messages("Page title: A"), None, _Params()))

    assert isinstance(response, ModelResponse)


# --- the opt-in switch -----------------------------------------------------------


@pytest.mark.parametrize(
    "value,enabled",
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
     ("0", False), ("false", False), ("", False), (None, False)],
)
def test_request_capture_path_follows_the_env_var(monkeypatch, tmp_path: Path, value, enabled):
    monkeypatch.setenv("TRACES_DIR", str(tmp_path))
    if value is None:
        monkeypatch.delenv("TRACE_FULL_INPUT", raising=False)
    else:
        monkeypatch.setenv("TRACE_FULL_INPUT", value)

    path = request_capture_path("abc123def")

    if not enabled:
        assert path is None
    else:
        assert path == tmp_path / "requests" / "abc123def.requests.txt"


def test_capture_directory_is_created_on_first_write(tmp_path: Path):
    """The path points into a requests/ dir that need not exist yet."""
    path = tmp_path / "requests" / "abc123.requests.txt"
    assert not path.parent.exists()

    _send(_model(path), "Page title: Home")

    assert path.exists()
