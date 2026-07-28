"""Text renderers for traces - the terminal (`human`) and agent-oriented (`agent`) views.

Both are plain text / markdown, dependency-free, so the same output reads well in a
terminal and when handed to AI. `human` truncates long fields and keeps things scannable;
`agent` drops truncation and emits full markdown for prompt-iteration analysis.
"""

import json
from typing import Any

from webagent.traces.trace import Generation, ToolCall, Trace

_PREVIEW_CHARS = 200


def _truncate(text: str, limit: int = _PREVIEW_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"... [+{len(text) - limit} chars]"


def _reasoning_note(gen: Generation) -> str | None:
    """What to show for reasoning: the text, a 'withheld by provider' note, or nothing."""
    if gen.reasoning:
        return gen.reasoning
    if gen.reasoning_encrypted or gen.reasoning_tokens:
        tokens = f"{gen.reasoning_tokens} reasoning tokens; " if gen.reasoning_tokens else ""
        return f"<withheld by provider — {tokens}set openai_reasoning_summary or use an Anthropic model>"
    return None


def _as_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _steps(trace: Trace) -> list[tuple[int, list]]:
    """Observations grouped by step, in first-seen step order."""
    grouped: dict[int, list] = {}
    for obs in trace.observations:
        grouped.setdefault(obs.step, []).append(obs)
    return sorted(grouped.items())


# --------------------------------------------------------------------------- list


def render_list(traces: list[Trace], *, fmt: str = "human") -> str:
    if not traces:
        return "No traces found."
    if fmt == "agent":
        return "\n\n".join(_agent_list_block(t) for t in traces)

    header = ("ID", "CREATED", "STATUS", "STEPS", "TOK(in/out)", "MODEL", "THINK", "FIXTURE", "TASK")
    rows = [header]
    for t in traces:
        rows.append(
            (
                t.trace_id[:8],
                t.created_at.strftime("%Y-%m-%d %H:%M"),
                t.status,
                str(t.steps_taken),
                f"{t.total_input_tokens}/{t.total_output_tokens}",
                _truncate(t.model, 22),
                t.thinking,
                t.fixture_id or "-",
                _truncate(t.task, 50),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _agent_list_block(t: Trace) -> str:
    lines = [
        f"## {t.trace_id[:8]}  ({t.status})",
        f"- task: {t.task}",
        f"- model: {t.model}  thinking: {t.thinking}  steps: {t.steps_taken}  "
        f"tokens: {t.total_input_tokens}/{t.total_output_tokens}",
    ]
    if t.fixture_id:
        lines.append(f"- fixture: {t.fixture_id}  run: {t.run_id or '-'}")
    highlight = _failure_highlight(t)
    if highlight:
        lines.append(f"- {highlight}")
    return "\n".join(lines)


def _failure_highlight(t: Trace) -> str | None:
    """A one-liner pointing at what likely went wrong, for scanning many runs at once."""
    errors = [o for o in t.observations if isinstance(o, ToolCall) and o.status == "error"]
    if errors:
        last = errors[-1]
        return f"tool errors: {len(errors)} (last: {last.name} → {_truncate(last.error or '', 120)})"
    if t.status != "success":
        return f"non-success status: {t.status}"
    return None


# --------------------------------------------------------------------------- show


def render_trace(trace: Trace, *, fmt: str = "human", step: int | None = None, raw: bool = False) -> str:
    raw = raw or fmt == "agent"
    lines = _header(trace, fmt)
    for step_no, observations in _steps(trace):
        if step is not None and step_no != step:
            continue
        lines.append("")
        lines.append(f"── step {step_no} ──" if fmt == "human" else f"### step {step_no}")
        for obs in observations:
            if isinstance(obs, Generation):
                lines.extend(_render_generation(obs, fmt, raw))
            else:
                lines.extend(_render_tool(obs, fmt, raw))
    if fmt == "human" and step is None and not raw:
        lines.append("")
        lines.append("(use --step N to focus a step, --raw to expand full input/output)")
    return "\n".join(lines)


def _header(trace: Trace, fmt: str) -> list[str]:
    title = f"# trace {trace.trace_id}" if fmt == "agent" else f"trace {trace.trace_id}"
    lines = [
        title,
        f"task: {trace.task}",
        f"url: {trace.url}",
        f"model: {trace.model}  thinking: {trace.thinking}  output_mode: {trace.output_mode}",
        f"status: {trace.status}  steps: {trace.steps_taken}  "
        f"duration: {trace.duration_seconds:.1f}s  tokens: {trace.total_input_tokens}/{trace.total_output_tokens}",
    ]
    if trace.fixture_id:
        lines.append(f"fixture: {trace.fixture_id}  run: {trace.run_id or '-'}")
    if fmt == "agent":
        lines.append("")
        lines.append("## system prompt")
        lines.append("```")
        lines.append(trace.system_prompt)
        lines.append("```")
    return lines


def _render_generation(gen: Generation, fmt: str, raw: bool) -> list[str]:
    marker = "◆ generation" if fmt == "human" else "**◆ generation**"
    lines = [f"{marker} [{gen.name}]  ({gen.duration_seconds:.1f}s, model={gen.model})"]
    reasoning = _reasoning_note(gen)
    if reasoning is not None:
        lines.append(f"  reasoning: {reasoning if raw else _truncate(reasoning)}")
    output = _as_json(gen.output)
    lines.append(f"  action: {output if raw else _truncate(output)}")
    tokens = f"  tokens: in={gen.input_tokens} out={gen.output_tokens}"
    if gen.reasoning_tokens:
        tokens += f" reasoning={gen.reasoning_tokens}"
    lines.append(tokens)
    if raw:
        lines.append("  input:")
        lines.extend(f"    {line}" for line in gen.input_prompt.splitlines())
    return lines


def _render_tool(tool: ToolCall, fmt: str, raw: bool) -> list[str]:
    marker = "▸ tool" if fmt == "human" else "**▸ tool**"
    args = ", ".join(f"{k}={_as_json(v)}" for k, v in tool.args.items())
    status = tool.status if tool.status == "ok" else f"ERROR: {tool.error}"
    lines = [f"{marker} {tool.name}({args}) → {status if raw else _truncate(status)}  ({tool.duration_seconds:.1f}s)"]
    if tool.result:
        lines.append(f"  result: {tool.result if raw else _truncate(tool.result)}")
    return lines
