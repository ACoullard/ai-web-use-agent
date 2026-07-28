# Evals — agent guide

Offline eval harness for the web agent. Each fixture is a `{task, url, expected}` case
run through `webagent.agent.run_task()` and graded automatically. See `README.md` in this
directory for the full user-facing reference; this file is the working guide for making
changes here.

## Layout

- `loader.py` — reads fixture YAML into `Fixture` models; resolves relative `url:` and
  `{fixture_dir}` substitutions; marks anything under a `live/` path as `is_live`.
- `models.py` — the `Fixture` / result dataclasses.
- `grading.py` — `exact_match` and `llm_judge` logic.
- `runner.py` — drives `run_task()` per fixture and grades the result.
- `report.py`, `history.py` — result formatting and the `runs/history.jsonl` trend log.
- `cli.py` — the `webagent evals ...` commands.
- `fixtures/local/` — deterministic `file://` fixtures (safe for CI).
- `fixtures/live/` — real-website fixtures, excluded unless `--live` is passed.

## Adding a fixture

One folder per fixture: `fixtures/local/<id>/fixture.yaml` plus every `.html` page it
needs as siblings. The folder is self-contained — copy/rename/delete it as one unit.

- `id` must be globally unique across all fixtures (the loader raises on duplicates).
- `url:` is resolved relative to the fixture's own folder, so use a bare filename
  (`store.html`), not an absolute path.
- For `file://` answers whose path is checkout-dependent, put `{fixture_dir}` in
  `expected` — the loader substitutes the resolved folder URI (see `pricing-page-link`).
- Multi-page flows: add more sibling HTML files and link between them
  (see `multi-step-docs-download/`, `go-back-recovery/`).

Prefer `fixtures/local/` for anything meant to run in CI — `live/` fixtures hit real
sites, drift when those pages change, and are smoke tests rather than a stable contract.

## Grading

- `exact_match` — `expected` is matched as a **recursive dict-subset**: every key in
  `expected` must be present and equal in the answer; extra answer keys are ignored.
- `llm_judge` — `rubric` is a checklist of strings; **all** must pass. The judge sees
  only `{task, output contract, answer, rubric}` — never the page. Write criteria
  checkable from the answer text alone; a criterion needing the source page will fail.
- `expected_status` other than `success` (e.g. `max_steps_exceeded`) grades only
  `AgentResult.status`; no `expected`/`rubric` needed (see `max-steps-guardrail`).

## Running & testing

```
webagent evals run                          # all local fixtures
webagent evals run local/<id>               # one fixture by folder
webagent evals run --live                   # include live fixtures
```

`--model`/`--judge-model` take a Pydantic AI `"<provider>:<model>"` id; `anthropic` and
`openai` are supported and need the matching API key in the environment.

**`evals run` makes real LLM calls** (costs money, needs a key) — don't invoke it just to
sanity-check a fixture. To verify a new fixture without a model, load it via
`loader.load_fixture_file(...)` and drive the page through `BrowserController.observe()`
directly. The pytest suite (`tests/`) is guarded against live LLM calls; keep it that way.

`runs/history.jsonl` is gitignored run-local telemetry — never commit it or treat it as
source.

## Traces

`evals run` records one trace per fixture run under `--trace-dir` (default `evals/runs/traces/`,
gitignored via `runs/`; `--no-trace` to skip). Each trace is stamped with `fixture_id` and the
suite's `run_id`, so after a run you can slice them with the top-level `webagent trace` commands
— e.g. `webagent trace list --fixture <id> --format agent` for a Claude-readable digest of a
fixture's runs, or `webagent trace serve --trace-dir evals/runs/traces` to browse. See the
top-level `AGENTS.md` for the trace model.
