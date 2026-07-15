from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from webagent.agent import run_task
from webagent.result import AgentResult

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
        "anthropic:claude-sonnet-5",
        "--model",
        help='Model identifier passed to Pydantic AI, e.g. "anthropic:claude-sonnet-5" or "openai:gpt-4o".',
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
    """
    _configure_logging(log_level)

    if schema is not None and description is not None:
        typer.echo("Error: pass at most one of --schema, --description", err=True)
        raise typer.Exit(code=2)

    output_schema = json.loads(schema.read_text()) if schema is not None else None

    result: AgentResult = asyncio.run(
        run_task(
            task=task,
            url=url,
            output_schema=output_schema,
            output_description=description,
            model=model,
            max_steps=max_steps,
            max_reask_attempts=max_reask_attempts,
            headless=headless,
            dry_run=dry_run,
        )
    )

    result_json = result.model_dump_json(indent=2)
    typer.echo(result_json)
    if output is not None:
        output.write_text(result_json, encoding="utf-8")

    raise typer.Exit(code=_EXIT_CODES[result.status])


if __name__ == "__main__":
    app()
