# ai-web-use-agent

An LLM-driven web browsing agent: it perceives a page as a list of interactive elements
plus a text summary (via Playwright), then takes actions (click, type, navigate, etc.) to
complete a task. `webagent/` is the agent; `evals/` is the offline eval harness.

## Environment

This project uses [uv](https://docs.astral.sh/uv/). Run everything through it, e.g.
`uv run pytest`, `uv run webagent ...`, `uv run python ...` — uv manages the virtualenv
and dependencies from `pyproject.toml`/`uv.lock`.
