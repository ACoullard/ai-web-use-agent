import asyncio
import enum
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from evals.cli import evals_app
from webagent.agent import run_task
from webagent.config import live_traces_dir, traces_root
from webagent.providers import (
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    ProviderConfigError,
    ThinkingLevel,
    resolve_thinking,
)
from webagent.result import AgentResult
from webagent.trace import FileTracer, NullTracer, Trace, filter_traces, find_trace, load_traces
from webagent.trace_view import render_list, render_trace
from webagent.trace_web import serve as serve_traces

# Scraped page text can contain arbitrary Unicode; the default console codepage
# (e.g. cp1252 on Windows) can't encode all of it, so force UTF-8 for stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="webagent",
    help="webagent - CLI for an LLM-driven browser automation agent.",
    no_args_is_help=True,
)
app.add_typer(evals_app, name="evals")

_EXIT_CODES = {
    "success": 0,
    "dry_run": 0,
    "validation_failed": 1,
    "max_steps_exceeded": 2,
}


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(version("ai-web-use-agent"))
        raise typer.Exit()


@app.callback()
def _app_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """webagent - CLI for an LLM-driven browser automation agent."""


def _configure_logging(log_level: LogLevel | None) -> None:
    level_name = log_level.value if log_level is not None else os.environ.get("LOG_LEVEL", "WARNING")
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Natural-language task/instruction for the agent."),
    url: str = typer.Option(..., "--url", "-u", help="Starting URL for the agent to navigate to."),
    schema: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Path to a JSON Schema file describing the expected output. Mutually exclusive with --description.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        help="Natural-language description of the expected output (best-effort, self-checked). "
        "Mutually exclusive with --schema.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help='Model identifier passed to Pydantic AI, e.g. "anthropic:claude-sonnet-5" or "openai:gpt-4o".',
    ),
    thinking: ThinkingLevel = typer.Option(
        DEFAULT_THINKING,
        "--thinking",
        help="Reasoning/thinking effort. Honored by reasoning models (e.g. Anthropic, OpenAI "
        "reasoning models); silently ignored by models that don't support it. Use 'off' to disable.",
    ),
    max_steps: int = typer.Option(25, "--max-steps", help="Max agent loop iterations before giving up."),
    max_reask_attempts: int = typer.Option(
        2, "--max-reask-attempts", help="Max re-ask attempts on output validation failure."
    ),
    headless: bool = typer.Option(
        True, "--headless/--no-headless", help="Run the browser headless or visible."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load the URL, take one observation, and print the exact system + first-turn "
        "prompt that would be sent to the model - no model calls, no actions taken, no LLM cost.",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Also write the JSON result to this file."),
    trace: bool = typer.Option(
        True, "--trace/--no-trace", help="Record a trace of the run under $TRACES_DIR/live."
    ),
    log_level: Optional[LogLevel] = typer.Option(
        None, "--log-level", help="Logging verbosity. [default: WARNING, or $LOG_LEVEL env var]"
    ),
) -> None:
    """Run the agent on TASK starting at URL and print the resulting AgentResult as JSON.

    At most one of --schema / --description may be given. If neither is given, the
    agent returns a freeform result (a JSON scalar or list of scalars).

    Exit codes:
      0  success / dry_run
      1  validation_failed
      2  max_steps_exceeded
      3  provider/config error (unsupported provider or missing API key)
    """
    _configure_logging(log_level)

    if schema is not None and description is not None:
        typer.echo("Error: pass at most one of --schema, --description", err=True)
        raise typer.Exit(code=2)

    output_schema = json.loads(schema.read_text()) if schema is not None else None
    trace_dir = live_traces_dir()

    try:
        result: AgentResult = asyncio.run(
            run_task(
                task=task,
                url=url,
                output_schema=output_schema,
                output_description=description,
                model=model,
                thinking=resolve_thinking(thinking),
                max_steps=max_steps,
                max_reask_attempts=max_reask_attempts,
                headless=headless,
                dry_run=dry_run,
                tracer=FileTracer(trace_dir) if trace else NullTracer(),
            )
        )
    except ProviderConfigError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=3)

    result_json = result.model_dump_json(indent=2)
    typer.echo(result_json)
    if output is not None:
        output.write_text(result_json, encoding="utf-8")
    # keep stdout clean JSON for scripting; the trace pointer goes to stderr.
    if result.trace_id is not None:
        typer.echo(f"trace {result.trace_id} saved to {trace_dir}", err=True)

    raise typer.Exit(code=_EXIT_CODES[result.status])


trace_app = typer.Typer(name="trace", help="Browse and inspect recorded traces.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")


@trace_app.command("list")
def trace_list(
    task: Optional[str] = typer.Option(None, "--task", help="Case-insensitive substring filter on the task."),
    model: Optional[str] = typer.Option(None, "--model", help="Exact model filter."),
    thinking: Optional[str] = typer.Option(None, "--thinking", help="Exact thinking-effort filter."),
    status: Optional[str] = typer.Option(None, "--status", help="Exact status filter."),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Exact fixture-id filter."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Exact eval run-id filter."),
    fmt: str = typer.Option("human", "--format", help="human (table) or agent (markdown for Claude)."),
) -> None:
    """List traces under $TRACES_DIR (live + evals), newest first, with optional filters."""
    traces = filter_traces(
        load_traces(traces_root()),
        task=task,
        model=model,
        thinking=thinking,
        status=status,
        fixture_id=fixture,
        run_id=run_id,
    )
    typer.echo(render_list(traces, fmt=fmt))


@trace_app.command("show")
def trace_show(
    trace_id: str = typer.Argument(..., help="Trace id (or unambiguous prefix), or a path to a trace JSON file."),
    step: Optional[int] = typer.Option(None, "--step", help="Focus a single step, fully expanded."),
    raw: bool = typer.Option(False, "--raw", help="Expand full input/output/reasoning (no truncation)."),
    fmt: str = typer.Option("human", "--format", help="human or agent (untruncated markdown for Claude)."),
) -> None:
    """Show one trace's step timeline, distinguishing generation vs tool observations."""
    root = traces_root()
    trace = find_trace(load_traces(root), trace_id)
    if trace is None and Path(trace_id).is_file():
        trace = Trace.model_validate_json(Path(trace_id).read_text(encoding="utf-8"))
    if trace is None:
        typer.echo(f"No trace matching {trace_id!r} in {root}", err=True)
        raise typer.Exit(code=1)
    typer.echo(render_trace(trace, fmt=fmt, step=step, raw=raw))


@trace_app.command("serve")
def trace_serve(
    port: int = typer.Option(8756, "--port", help="Port to bind on loopback."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the page in a browser on start."),
) -> None:
    """Serve a local website (127.0.0.1) for browsing and drilling into traces."""
    serve_traces(traces_root(), port=port, open_browser=open_browser)


if __name__ == "__main__":
    app()
