from __future__ import annotations

import asyncio
import enum
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import typer

from evals.history import append_history, load_history, pass_rate_by_run
from evals.loader import filter_fixtures, load_fixture_paths
from evals.report import format_failure_section, format_progress_mark, format_summary_line
from evals.runner import run_suite
from webagent.providers import ProviderConfigError, ThinkingLevel, resolve_thinking

evals_app = typer.Typer(name="evals", help="Offline eval harness for the agent.")


class LogLevel(str, enum.Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def _configure_logging(log_level: Optional[LogLevel]) -> None:
    level_name = log_level.value if log_level is not None else os.environ.get("LOG_LEVEL", "WARNING")
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@evals_app.command("run")
def run(
    paths: Optional[List[Path]] = typer.Argument(
        None,
        help="Fixture files and/or directories to run, e.g. `local/boolean-in-stock.yaml` or `local` "
        "(a directory runs everything under it, recursively). Relative paths resolve against "
        "--fixtures-root. Omit to run everything under --fixtures-root.",
    ),
    fixtures_root: Path = typer.Option(
        Path("evals/fixtures"),
        "--fixtures-root",
        "--fixtures",
        help="Root fixtures directory: resolves relative PATHS above, and is the default "
        "selection when no PATHS are given.",
        exists=True,
    ),
    live: bool = typer.Option(
        False, "--live", help="Include live fixtures (real websites; excluded by default)."
    ),
    model: str = typer.Option("anthropic:claude-sonnet-5", "--model", help="Model under test."),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="Model for llm_judge grading. Defaults to --model."
    ),
    thinking: ThinkingLevel = typer.Option(
        ThinkingLevel.MEDIUM,
        "--thinking",
        help="Reasoning/thinking effort for the model under test. Honored by reasoning models; "
        "silently ignored by models that don't support it. Use 'off' to disable.",
    ),
    concurrency: int = typer.Option(1, "--concurrency", help="Max fixtures to run in parallel."),
    headless: bool = typer.Option(True, "--headless/--no-headless"),
    history: Path = typer.Option(
        Path("evals/runs/history.jsonl"), "--history", help="Append-only JSONL results log."
    ),
    report: Optional[Path] = typer.Option(None, "--report", "-o", help="Also write this run's records as JSON."),
    log_level: Optional[LogLevel] = typer.Option(
        None, "--log-level", help="Logging verbosity. [default: WARNING, or $LOG_LEVEL env var]"
    ),
) -> None:
    """Run fixtures at PATHS (or everything under --fixtures-root) against the agent
    and grade the results.

    Exit code 0 only if every selected fixture passes; 1 otherwise (or if no
    fixtures matched, or a given path doesn't exist); 3 for a provider/config error
    (unsupported provider or missing API key). Live fixtures are excluded unless
    --live is given.
    """
    _configure_logging(log_level)

    selection = paths or [fixtures_root]
    resolved_paths = []
    for raw in selection:
        candidate = raw if raw.is_absolute() else fixtures_root / raw
        resolved_paths.append(candidate if candidate.exists() else raw)

    missing = [p for p in resolved_paths if not p.exists()]
    if missing:
        for p in missing:
            typer.echo(f"Error: fixture path not found: {p}", err=True)
        raise typer.Exit(code=1)

    all_fixtures = load_fixture_paths(resolved_paths)
    selected = filter_fixtures(all_fixtures, include_live=live)

    if not selected:
        typer.echo("No fixtures matched the given filters.", err=True)
        raise typer.Exit(code=1)

    source = ", ".join(os.path.relpath(p) for p in resolved_paths)
    typer.echo(f"Running {len(selected)} fixture{'s' if len(selected) != 1 else ''} from {source}")

    def _on_complete(record) -> None:
        typer.echo(format_progress_mark(record), nl=False)

    try:
        records = asyncio.run(
            run_suite(
                selected,
                model=model,
                judge_model=judge_model or model,
                thinking=resolve_thinking(thinking),
                concurrency=concurrency,
                headless=headless,
                on_complete=_on_complete,
            )
        )
    except ProviderConfigError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=3)
    typer.echo()  # end the progress-mark line

    failure_section = format_failure_section(records)
    if failure_section:
        typer.echo()
        typer.echo(failure_section)

    typer.echo()
    typer.echo(format_summary_line(records))

    append_history(history, records)
    if report is not None:
        report.write_text(
            json.dumps([record.model_dump(mode="json") for record in records], indent=2),
            encoding="utf-8",
        )

    raise typer.Exit(code=0 if all(record.passed for record in records) else 1)


@evals_app.command("history")
def history_cmd(
    history: Path = typer.Option(Path("evals/runs/history.jsonl"), "--history", help="History JSONL log to read."),
    fixture_id: Optional[str] = typer.Option(None, "--id", help="Only show runs of this fixture id."),
) -> None:
    """Print pass-rate trend across past `evals run` invocations, oldest first."""
    records = load_history(history)
    if fixture_id is not None:
        records = [record for record in records if record.fixture_id == fixture_id]

    if not records:
        typer.echo("No history recorded yet.")
        return

    for run_id, run_at, pass_rate in pass_rate_by_run(records):
        typer.echo(f"{run_at.isoformat()}  {run_id}  {pass_rate:.0%} passed")
