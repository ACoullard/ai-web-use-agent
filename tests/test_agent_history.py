"""Context management in the agent loop: history accumulation and observation windowing.

The regression these guard against: `message_history = result.new_messages()` collapsed
the history to the single previous exchange, which also dropped the system prompt - and
with it the task - from step 2 onward. The agent then browsed with no idea what it had
been asked to do. See `_stub_aged_observations` and its call site in webagent/agent.py.
"""

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from webagent.agent import (
    _AGED_OBSERVATION_STUB,
    _HISTORY_WINDOW,
    _stub_aged_observations,
)
from webagent.page_snapshot import ElementInfo, PageSnapshot


def _observation(n: int) -> str:
    """A stand-in shaped like PageSnapshot.to_prompt() output."""
    return f"Page title: Page {n}\nURL: https://example.com/{n}\n\nInteractive elements:\n[1] <a> 'link {n}'"


def _exchange(n: int, note: str = "") -> list[ModelRequest | ModelResponse]:
    """One step's worth of messages: user observation, then the model's tool call."""
    return [
        ModelRequest(parts=[UserPromptPart(content=note + _observation(n))]),
        ModelResponse(parts=[ToolCallPart("final_result_ClickAction", {"index": 1})]),
        ModelRequest(parts=[ToolReturnPart("final_result_ClickAction", "Final result processed.")]),
    ]


def _history(steps: int) -> list[ModelRequest | ModelResponse]:
    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(parts=[SystemPromptPart(content="SYSTEM: the task"), UserPromptPart(content=_observation(0))]),
        ModelResponse(parts=[ToolCallPart("final_result_ClickAction", {"index": 1})]),
        ModelRequest(parts=[ToolReturnPart("final_result_ClickAction", "Final result processed.")]),
    ]
    for n in range(1, steps):
        messages.extend(_exchange(n))
    return messages


def _observation_texts(messages) -> list[str]:
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def test_keeps_only_the_last_n_observations_verbatim():
    messages = _stub_aged_observations(_history(5), keep_last=2)

    texts = _observation_texts(messages)
    assert len(texts) == 5
    assert texts[:3] == [_AGED_OBSERVATION_STUB] * 3  # steps 0-2 aged out
    assert texts[3] == _observation(3)  # last two survive intact
    assert texts[4] == _observation(4)


def test_history_no_longer_than_keep_last_is_untouched():
    messages = _stub_aged_observations(_history(2), keep_last=2)

    assert _observation_texts(messages) == [_observation(0), _observation(1)]


def test_system_prompt_survives_stubbing():
    """The bug's real damage: losing the task. Stubbing must never touch it."""
    messages = _stub_aged_observations(_history(6), keep_last=1)

    system = [p for m in messages for p in m.parts if isinstance(p, SystemPromptPart)]
    assert [p.content for p in system] == ["SYSTEM: the task"]


def test_tool_call_and_return_pairs_are_preserved():
    """Stubbing rewrites content in place; it must not drop or reorder messages.

    A sliced history can orphan a tool call from its result, which providers reject.
    """
    before = _history(5)
    counts_before = [type(m).__name__ for m in before]
    parts_before = [[type(p).__name__ for p in m.parts] for m in before]

    after = _stub_aged_observations(before, keep_last=2)

    assert [type(m).__name__ for m in after] == counts_before
    assert [[type(p).__name__ for p in m.parts] for m in after] == parts_before


def test_is_idempotent_across_repeated_passes():
    """The loop feeds the result back in as history each step, so re-stubbing is normal."""
    messages = _history(5)
    once = _observation_texts(_stub_aged_observations(messages, keep_last=2))
    twice = _observation_texts(_stub_aged_observations(messages, keep_last=2))

    assert once == twice
    assert twice.count(_AGED_OBSERVATION_STUB) == 3


def test_prepended_notes_survive_but_observation_is_stubbed():
    """Reask / failed-action notes are short and still relevant; the page state is not."""
    note = "Your last action (click(4)) failed: element not found. "
    messages = _history(1) + _exchange(1, note=note) + _exchange(2)

    texts = _observation_texts(_stub_aged_observations(messages, keep_last=1))

    assert texts[1] == note + _AGED_OBSERVATION_STUB
    assert texts[2] == _observation(2)


def test_long_note_is_capped_when_its_step_ages_out():
    """Playwright failures carry a multi-line 'Call log:' dump - 3.3k chars in the trace
    that prompted this. Worth showing when it happens, not for the rest of the run."""
    from webagent.agent import _AGED_NOTE_MAX_CHARS

    note = "Your last action failed: Timeout 30000ms exceeded. Call log: " + ("x" * 4000)
    messages = _history(1) + _exchange(1, note=note) + _exchange(2)

    texts = _observation_texts(_stub_aged_observations(messages, keep_last=1))

    aged = texts[1]
    assert aged.endswith(_AGED_OBSERVATION_STUB)
    assert len(aged) < _AGED_NOTE_MAX_CHARS + len(_AGED_OBSERVATION_STUB) + 10
    assert aged.startswith("Your last action failed: Timeout")  # the gist survives


def test_short_note_is_not_truncated():
    note = "Your last action (click(4)) failed: element not found. "
    messages = _history(1) + _exchange(1, note=note) + _exchange(2)

    texts = _observation_texts(_stub_aged_observations(messages, keep_last=1))

    assert texts[1] == note + _AGED_OBSERVATION_STUB


def test_note_without_an_observation_is_left_alone():
    """A bare note has no page state to strip and must not be clobbered."""
    messages = _history(2) + [ModelRequest(parts=[UserPromptPart(content="Reconsider and call finish() again.")])]

    texts = _observation_texts(_stub_aged_observations(messages, keep_last=1))

    assert texts[-1] == "Reconsider and call finish() again."
    assert texts[0] == _AGED_OBSERVATION_STUB


def test_non_string_user_content_is_skipped():
    """Multimodal content isn't a str; the marker search must not blow up on it."""
    messages = _history(1) + [ModelRequest(parts=[UserPromptPart(content=["a", "b"])])]

    assert _stub_aged_observations(messages, keep_last=0) is not None


# --- loop-level regression -------------------------------------------------------


class _FakeBrowser:
    """Minimal stand-in for BrowserController: every page looks the same but numbered."""

    def __init__(self) -> None:
        self.step = 0
        self.closed = False

    async def goto(self, url: str) -> None:
        pass

    async def observe(self) -> PageSnapshot:
        self.step += 1
        return PageSnapshot(
            title=f"Page {self.step}",
            url=f"https://example.com/{self.step}",
            elements=[ElementInfo(index=1, tag="a", name=f"link {self.step}")],
            text_summary=f"summary {self.step}",
        )

    async def execute(self, action) -> str | None:
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_loop_keeps_system_prompt_and_windows_observations(monkeypatch):
    """End-to-end over the loop: the task must be visible on every single step.

    Drives run_task with a scripted model, recording what the model actually receives.
    """
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from webagent import agent as agent_module

    # Snapshot each request's text eagerly. Storing the message objects would alias them:
    # _stub_aged_observations mutates UserPromptPart.content in place after every step, so
    # references would report the final trimmed state rather than what was actually sent.
    seen: list[dict] = []
    step = {"n": 0}

    def respond(messages, info: AgentInfo) -> ModelResponse:
        seen.append(
            {
                "system": [
                    p.content for m in messages for p in m.parts if isinstance(p, SystemPromptPart)
                ],
                "observations": [
                    p.content
                    for m in messages
                    for p in m.parts
                    if isinstance(p, UserPromptPart) and isinstance(p.content, str)
                ],
                "memories": [
                    p.args["memory"]
                    for m in messages
                    for p in m.parts
                    if isinstance(p, ToolCallPart) and isinstance(p.args, dict)
                ],
            }
        )
        step["n"] += 1
        if step["n"] >= 5:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "final_result_FinishAction",
                        {"type": "finish", "answer": "done", "memory": "finished, 2 of 2 products"},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_result_ClickAction",
                    {"type": "click", "index": 1, "memory": f"step {step['n']}, {step['n']} of 2 products"},
                )
            ]
        )

    async def fake_launch(headless: bool = True):
        return _FakeBrowser()

    monkeypatch.setattr(agent_module.BrowserController, "launch", staticmethod(fake_launch))
    # Provider plumbing expects a "<provider>:<model>" string; we pass a FunctionModel.
    monkeypatch.setattr(agent_module, "check_model_config", lambda model: None)
    monkeypatch.setattr(agent_module, "build_model_settings", lambda model, thinking: None)

    result = await agent_module.run_task(
        task="UNIQUE-TASK-MARKER: add exactly 2 products",
        url="https://example.com/",
        model=FunctionModel(respond),
        thinking=False,
        max_steps=10,
    )

    assert result.status == "success"
    assert len(seen) == 5

    for i, request in enumerate(seen):
        assert len(request["system"]) == 1, f"request {i}: {len(request['system'])} system prompts"
        assert "UNIQUE-TASK-MARKER" in request["system"][0], f"request {i} lost the task"

    # Every request carries exactly _HISTORY_WINDOW verbatim observations, counting the
    # current one - the invariant the system prompt promises the model. An off-by-one here
    # silently sends 50% more page text than advertised.
    for i, request in enumerate(seen):
        live = [o for o in request["observations"] if _AGED_OBSERVATION_STUB not in o]
        assert len(live) == min(i + 1, _HISTORY_WINDOW), (
            f"request {i}: {len(live)} verbatim observations, expected {min(i + 1, _HISTORY_WINDOW)}"
        )
        stubs = len(request["observations"]) - len(live)
        assert stubs == max(0, i + 1 - _HISTORY_WINDOW), f"request {i}: {stubs} stubs"

    # The memory trail survives in full, which is what makes the stubbing safe.
    assert seen[-1]["memories"] == [f"step {n}, {n} of 2 products" for n in range(1, 5)]
