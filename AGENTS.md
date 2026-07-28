# ai-web-use-agent

An LLM-driven web browsing agent: it perceives a page as a list of interactive elements
plus a text summary (via Playwright), then takes actions (click, type, navigate, etc.) to
complete a task. `webagent/` is the agent; `webagent/traces/` records what it did; `evals/`
is the offline eval harness.

## Environment

This project uses [uv](https://docs.astral.sh/uv/). Run everything through it, e.g.
`uv run pytest`, `uv run webagent ...`, `uv run python ...` — uv manages the virtualenv
and dependencies from `pyproject.toml`/`uv.lock`.

Runtime configuration lives in the environment, not in CLI flags. `webagent/config.py`
loads a `.env` in the working directory at import (real exported vars win over it); copy
`.env.example` to `.env` and fill it in. Variables: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
(provider keys), `TRACES_DIR` (where traces go, default `.webagent/traces`), `LOG_LEVEL`.

## Traces

Every `run_task()` records a durable trace of the run, browsable with `webagent trace
list|show|serve`. The code and its guide live in `webagent/traces/` — see
`webagent/traces/AGENTS.md` before changing anything about tracing.
