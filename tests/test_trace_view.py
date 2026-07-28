from datetime import datetime, timezone

from webagent.trace import Generation, ToolCall, Trace
from webagent.trace_view import render_list, render_trace

_LONG = "x" * 500


def _trace() -> Trace:
    return Trace(
        trace_id="abcd1234 effff"[:12].replace(" ", ""),
        created_at=datetime(2026, 7, 24, 10, 30, tzinfo=timezone.utc),
        task="find the pricing page",
        url="https://example.com",
        model="openai:gpt-5.6",
        thinking="medium",
        output_mode="freeform",
        status="success",
        steps_taken=2,
        duration_seconds=3.1,
        total_input_tokens=110,
        total_output_tokens=22,
        system_prompt="you are an agent",
        observations=[
            Generation(
                name="decide_action",
                step=0,
                duration_seconds=1.2,
                model="openai:gpt-5.6",
                input_prompt=_LONG,
                reasoning="the pricing link is element 4",
                output={"type": "click", "index": 4},
                input_tokens=100,
                output_tokens=20,
            ),
            ToolCall(name="click", step=0, args={"index": 4}, duration_seconds=0.3, status="ok"),
            Generation(
                name="decide_action",
                step=1,
                duration_seconds=0.9,
                model="openai:gpt-5.6",
                input_prompt="short prompt",
                reasoning=None,
                reasoning_encrypted=True,
                reasoning_tokens=512,
                output={"type": "finish", "answer": "$20/mo"},
                input_tokens=10,
                output_tokens=2,
            ),
            ToolCall(name="click", step=1, args={"index": 9}, duration_seconds=0.1,
                     status="error", error="no element 9"),
        ],
    )


def test_render_list_human_and_agent():
    human = render_list([_trace()])
    assert "TASK" in human and "find the pricing page" in human
    assert "abcd1234"[:8] in human

    agent = render_list([_trace()], fmt="agent")
    assert agent.startswith("## ")
    assert "task: find the pricing page" in agent


def test_render_list_empty():
    assert "No traces" in render_list([])


def test_render_list_agent_highlights_tool_errors():
    agent = render_list([_trace()], fmt="agent")
    assert "tool errors" in agent


def test_render_trace_distinguishes_generation_and_tool():
    out = render_trace(_trace())
    assert "◆ generation" in out
    assert "▸ tool" in out
    assert "click(index=4)" in out


def test_render_trace_truncates_by_default_and_expands_with_raw():
    default = render_trace(_trace())
    assert _LONG not in default  # long input prompt hidden unless raw
    assert "--raw" in default  # hint shown

    raw = render_trace(_trace(), raw=True)
    assert _LONG in raw
    assert "input:" in raw


def test_render_trace_shows_reasoning_withheld_note():
    out = render_trace(_trace(), raw=True)
    assert "withheld by provider" in out
    assert "512 reasoning tokens" in out


def test_render_trace_flags_tool_error():
    out = render_trace(_trace(), raw=True)
    assert "no element 9" in out


def test_render_trace_step_focus():
    only_step_1 = render_trace(_trace(), step=1)
    assert "step 1" in only_step_1
    assert "index=4" not in only_step_1  # step 0's tool call excluded


def test_render_trace_agent_includes_system_prompt():
    agent = render_trace(_trace(), fmt="agent")
    assert "system prompt" in agent
    assert "you are an agent" in agent
