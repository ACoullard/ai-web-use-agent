# Evals

Offline eval harness for the agent (design.md §4). Fixture-based: each fixture is a
`{task, url, expected output}` case, run through `run_task()` and graded automatically.

## Running

```
webagent evals run                                  # everything under evals/fixtures (local only by default)
webagent evals run local                             # a folder: everything under it, recursively
webagent evals run local/boolean-in-stock            # a single fixture, by its folder
webagent evals run local/a/fixture.yaml local/b/fixture.yaml  # multiple explicit paths
webagent evals run --live                            # also run live fixtures (real websites)
webagent evals run --model openai:gpt-4o --judge-model anthropic:claude-sonnet-5
webagent evals run --thinking off                    # disable model reasoning/thinking
webagent evals history                               # pass-rate trend across past runs
```

`--model`/`--judge-model` take a Pydantic AI `"<provider>:<model>"` identifier.
Supported providers are `anthropic` (`ANTHROPIC_API_KEY`) and `openai`
(`OPENAI_API_KEY`) - the corresponding key must be set in the environment. `--thinking`
(`minimal`/`low`/`medium`/`high`/`xhigh`/`off`) sets reasoning effort;
it's honored by reasoning models and silently ignored by models that don't support it.

Positional paths (files or directories) are resolved relative to `--fixtures-root`
(default `evals/fixtures`), or used as-is if absolute. Omit them to run everything
under `--fixtures-root`. `--fixtures` is kept as an alias for `--fixtures-root`.

Exit code is `0` only if every selected fixture passes; `3` for a provider/config error
(unsupported provider or a missing API key).

## Fixture layout

Each fixture gets its own folder containing its YAML (`fixture.yaml` by convention)
alongside every HTML page it needs - `evals/fixtures/local/<id>/fixture.yaml` plus
sibling `.html` files. A fixture's `url:` and any `{fixture_dir}` substitution in
`expected` resolve relative to that folder, so multi-page fixtures (a chain of
linked pages) are self-contained: nothing is shared across fixtures, and a folder
can be copied, renamed, or deleted as one unit. See `local/multi-step-docs-download/`
or `local/go-back-recovery/` for examples with more than one page.

- `fixtures/local/` - fixtures against local static `file://` pages. Deterministic, no network, safe for CI.
- `fixtures/live/` - fixtures against real websites (Wikipedia, MDN, docs.python.org). Marked live automatically (by directory), **excluded unless `--live` is passed**. These may need retuning if the target pages change - they're a smoke test, not a stable contract.

To run a specific fixture or subset, select it by path (see Running above) rather than by id - e.g. `webagent evals run local/boolean-in-stock`.

## Grading

- `exact_match`: the fixture's `expected` is matched against the agent's answer as a **recursive dict-subset** - every key in `expected` must be present and equal in the answer, but extra keys (e.g. optional schema fields the fixture doesn't care about) are ignored. Lists/scalars use plain equality.
- `llm_judge`: the fixture's `rubric` is a **checklist** of criteria (list of strings). A judge model (`--judge-model`, defaults to `--model`) evaluates each criterion independently. **All criteria must pass for the fixture to pass** - but the pass fraction (e.g. `3/4`) is always recorded and shown for failures, so partial credit stays visible even though the grade itself is strict all-or-nothing.
  - The judge only ever sees `{task, output contract, agent's answer, rubric}` - it has no access to the page the agent browsed. Write rubric criteria that are checkable from the answer text alone (e.g. "mentions watering frequency"). A criterion that requires comparing the answer against the source page (e.g. "doesn't state any fact not present in the article") is not gradable and will fail unpredictably - the judge has nothing to check it against.
- Fixtures with `expected_status` other than `success` (e.g. `max_steps_exceeded`) only grade `AgentResult.status` - no `expected`/`rubric` needed. Used for guardrail-path cases like `max-steps-guardrail`.

Results append to `runs/history.jsonl` (gitignored - run-local telemetry, not source).
