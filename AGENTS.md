# ai-web-use-agent

An LLM-driven web browsing agent: it perceives a page as a list of interactive elements
plus a text summary (via Playwright), then takes actions (click, type, navigate, etc.) to
complete a task. `webagent/` is the agent; `evals/` is the offline eval harness.

## Environment

This project uses [uv](https://docs.astral.sh/uv/). Run everything through it, e.g.
`uv run pytest`, `uv run webagent ...`, `uv run python ...` — uv manages the virtualenv
and dependencies from `pyproject.toml`/`uv.lock`.

Runtime configuration lives in the environment, not in CLI flags. `webagent/config.py`
loads a `.env` in the working directory at import (real exported vars win over it); copy
`.env.example` to `.env` and fill it in. Variables: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
(provider keys), `TRACES_DIR` (where traces go, default `.webagent/traces`), `LOG_LEVEL`.

## Traces

Each `run_task()` records a durable trace (on by default; `--no-trace` to skip). Traces are
written under `$TRACES_DIR` (default `.webagent/traces/`), split into `live/` for `webagent
run` and `evals/` for the eval harness; `webagent trace` browses the whole tree. A trace is
a `Trace` — filterable
metadata plus an ordered list of `Observation`s, one JSON file per run (see `webagent/trace.py`).

`run_task()` never learns where traces go. It takes a `Tracer` (like a logger) and calls
`start()` to get a per-run `Recording` it writes observations into; the tracer decides
whether anything is persisted and where. `FileTracer(dir)` writes JSON; `NullTracer` is the
off-switch, so the loop needs no `if tracing` branches. `tracer.with_labels(...)` derives a
tracer carrying extra filterable metadata — the eval harness stamps `run_id` once at suite
level and `fixture_id` per fixture, which is why neither appears in `run_task`'s signature.
One `Recording` per run, so concurrent eval fixtures never share mutable state.
Following Langfuse's "observation type" idea there are two kinds: `Generation` (an LLM call —
reasoning, chosen action, tokens) and `ToolCall` (a browser action + its result/error). The
`finish` action is the final `Generation`'s output, not a tool. This is a self-rolled model
read straight off the objects the loop already holds — **not** Pydantic AI's OpenTelemetry
spans (which don't see browser-action execution and produce one span per step); field names
still track OTel's GenAI vocabulary for familiarity.

Browse them with `webagent trace`:
- `webagent trace list [--task/--model/--thinking/--status/--fixture/--run] [--format human|agent]`
- `webagent trace show <id-prefix|path> [--step N] [--raw] [--format human|agent]`
- `webagent trace serve` — a local (loopback-only) website to filter runs and drill into steps.
  Its page is plain static assets in `webagent/web/` (served under `/static/`, read per request,
  so a browser refresh picks up edits); `webagent/trace_web.py` is only routing and the JSON API.

`--format agent` emits markdown built for Claude to read when iterating on prompts. Note that
reasoning text may be empty even when the model reasoned: OpenAI reasoning models return it
**encrypted** (traces flag this via `reasoning_encrypted`/`reasoning_tokens`) — set
`openai_reasoning_summary` or use an Anthropic model to see the text.
