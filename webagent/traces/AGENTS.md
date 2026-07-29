# Traces — agent guide

Durable, human- and AI-legible records of an agent run. Every `run_task()` produces one
(on by default; `--no-trace` to skip). This file is the working guide for changing tracing;
the root `AGENTS.md` only points here.

## Layout

- `trace.py` — the whole model and machinery: `Trace`/`Generation`/`ToolCall`, the
  `Tracer`/`Recording` protocols, `FileTracer`/`NullTracer`, `TraceRecorder`, and the
  storage + query helpers (`save_trace`, `load_traces`, `filter_traces`, `find_trace`).
- `view.py` — renders traces for the terminal (`render_list`, `render_trace`).
- `server.py` — the loopback web browser for traces: routing + a read-only JSON API.
- `web/` — static assets for that page (`index.html`, `app.js`, `app.css`).
- `__init__.py` — re-exports the public surface. Import from `webagent.traces`, not from
  the submodules: `from webagent.traces import FileTracer, load_traces`.

## The model

A `Trace` is one run: filterable metadata plus an ordered list of `Observation`s, saved as
one JSON file. Following Langfuse's "observation type" idea there are exactly two kinds:
`Generation` (an LLM call — reasoning, chosen action, tokens) and `ToolCall` (a browser
action + its result/error). The `finish` action is the final `Generation`'s output, not a
tool.

This is a self-rolled model read straight off the objects the loop already holds
(`result.new_messages()`, `result.output`, `result.usage()`) — **not** Pydantic AI's
OpenTelemetry spans, which don't see browser-action execution and produce one span per
step. Field names still track OTel's GenAI vocabulary (`input_tokens`, `finish_reason`,
`provider_response_id`) so the JSON stays familiar and could be exported later.

## Injection: why `run_task()` takes a `Tracer`

`run_task()` never learns where traces go, and does no file I/O. It takes a `Tracer` — the
same shape as a logger — and calls `start()` to get a per-run `Recording` to write
observations into. The tracer decides whether anything is persisted and where.

- `FileTracer(dir)` writes one JSON file per run. `NullTracer` drops everything, so the
  loop body needs no `if tracing` branches and `Recording.trace_id` is `None`.
- `tracer.with_labels(...)` derives a *new* tracer carrying extra filterable metadata
  (never mutates the original). The eval harness stamps `run_id` once at suite level and
  `fixture_id` per fixture — which is why neither appears in `run_task()`'s signature.
- One `Recording` per run. This matters: `run_suite` runs fixtures concurrently, so a
  shared mutable accumulator would interleave observations from different fixtures. Keep
  the config/accumulator split intact if you touch this.

The only thing that crosses back is `AgentResult.trace_id`, so the CLI can print an id you
can pass to `webagent trace show`.

## Where files land

Under `$TRACES_DIR` (default `traces/`), split into `live/` for `webagent run` and
`evals/` for the harness. `load_traces()` recurses, so `webagent trace` browses both as one
collection. Filenames are `<timestamp>-<trace_id[:8]>.json`, which sorts chronologically.
The two `FileTracer(...)` construction sites in `webagent/cli.py` and `evals/cli.py` are the
only places a trace directory is named.

## Browsing

- `webagent trace list [--task/--model/--thinking/--status/--fixture/--run] [--format human|agent]`
- `webagent trace show <id-prefix|path> [--step N] [--raw] [--format human|agent]`
- `webagent trace serve` — loopback-only website to filter runs and drill into steps. Assets
  are read per request, so a browser refresh picks up edits to `web/`; `server.py` is only
  routing and the JSON API.

`--format agent` emits markdown built for an LLM to read when iterating on prompts.
