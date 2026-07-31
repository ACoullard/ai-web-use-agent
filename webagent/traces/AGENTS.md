# Traces — agent guide

Durable, human- and AI-legible records of an agent run. Every `run_task()` produces one
(on by default; `--no-trace` to skip). This file is the working guide for changing tracing;
the root `AGENTS.md` only points here.

## Layout

- `trace.py` — the whole model and machinery: `Trace`/`Generation`/`ToolCall`, the
  `Tracer`/`Recording` protocols, `FileTracer`/`NullTracer`, `TraceRecorder`, and the
  storage + query helpers (`save_trace`, `load_traces`, `filter_traces`, `find_trace`).
- `requests.py` — opt-in capture of the exact model input (`CapturingModel`,
  `render_request`).
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

`Generation.memory` is the agent's own running progress note, lifted out of the action
(it's stripped from both `Generation.output` and `ToolCall.args`, where it isn't an
argument). Because observations age out of the message history, `memory` is the whole of
what the agent carried forward — read it first when a run loses track of its task.

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
- `FileTracer(dir, live=True)` also decides *how often*: it rewrites the same file after
  every observation, so `trace serve` can follow a run in progress. `webagent run` turns
  this on; the eval harness leaves it off, because `run_suite` runs fixtures concurrently
  and would multiply the writes for nobody to watch.
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

Because the filename is derived from `created_at` + `trace_id` - both fixed when the
recorder is built - a live run's snapshots all overwrite one path, and the on-disk format
is the same whole-`Trace` JSON either way. Nothing that reads traces needs to know a run
was watched live. `save_trace` writes temp-then-`os.replace` so a reader never sees a
half-written file (on Windows the replace can lose a race with an open reader, hence the
short retry).

## Statuses

`Trace.status` is a plain `str`, not a `Literal`, and carries two values `AgentResult`
never has: `running` while a live run is in progress, and `interrupted` for a run that
died before reaching a return path. The latter comes from `run_task`'s `finally`, which
calls `finish()` unconditionally - `finish()` is idempotent, so it's a no-op on every
normal exit and only bites on a crash or Ctrl-C, which used to persist nothing at all.

## Browsing

- `webagent trace list [--task/--model/--thinking/--status/--fixture/--run] [--format human|agent]`
- `webagent trace show <id-prefix|path> [--step N] [--raw] [--format human|agent]`
- `webagent trace serve` — loopback-only website to filter runs and drill into steps. Assets
  are read per request, so a browser refresh picks up edits to `web/`; `server.py` is only
  routing and the JSON API.

The page polls: the list every 2s, an open run's detail view every 1s, stopping as soon as
the run reaches a terminal status. A live detail view appends only the observations it
hasn't seen (they're append-only), so expanded panels and scroll position survive; new
cards arrive expanded and the page follows the newest step unless you've scrolled up.
Run `webagent run --no-headless` beside it to watch the agent reason and act in real time.

`server.py` caches parsed traces on `(mtime_ns, size)` — at 1 Hz, re-parsing every trace on
disk to serve one is the difference between free and not. A file that briefly fails to open
(Windows, mid-replace) falls back to its last good parse rather than vanishing from the list.

`--format agent` emits markdown built for an LLM to read when iterating on prompts.
