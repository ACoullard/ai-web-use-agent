
# # Summary
An agent / tool / workflow callable from cli that prompts and facilitates an LLMs exploration of a website and the return on information from that website in a structured format.

# Requirements
## Functional Requirements
- Callable from other agents
- Can operate autonomously
- Able to reliably navigate multi page websites
- Outputs in a structured format that can be defined in each usage. For example, some queries may look for a yes/no answer boolean and others may look for a JSON of info collected from the site or a particular link to a target resource or page.

## Non Functional Requirements
- Should be a good source of learning for how to create agent systems
- Not too expensive (less important)
- Provider agnostic
- Testable via evals

# Design Doc - Claude Written

## 1. Decisions locked in (from requirements clarification)

| Question | Decision |
|---|---|
| Browsing method | Headless browser, DOM/accessibility-tree based (not screenshots/vision) |
| Calling interface | CLI **and** MCP server, both backed by the same core |
| Language | Python |
| Autonomy scope | Read + safe interactions (search, filter, login) — no irreversible actions (purchases, deletes, sends) without explicit opt-in |
| Output spec | Caller supplies either a JSON Schema (strict) or a natural-language description (best-effort) |
| Guardrails | Open parameter — sized below, tunable per call |
| Evals | Custom benchmark suite (site + task + expected output), auto-graded |
| Session/auth state | Stateless per run for v1 — credentials passed in, nothing persisted |

## 2. Research findings that shape the design

- **browser-use** (MIT, Python) proves out an *indexed-element* DOM abstraction: inject JS, walk visible/interactive elements, assign each a `highlight_index`, let the LLM act via `click(index=5)` / `type(index=12, text=...)`. It also validates final output against a caller-supplied Pydantic model. Its OSS guardrails (step/time/cost limits) and MCP story are underdocumented/mostly live in their paid Cloud product — not something to depend on as-is.
- **Playwright MCP** (Microsoft, Apache-2.0) confirms the accessibility-tree snapshot is the right cheap/precise perception primitive (~2-5KB vs 500KB+ for a screenshot) and is a pure low-level tool server — no agent loop, no schema-constrained output. Whatever we build on top, *we* own the loop and the output contract.
- **Eval benchmarks** (WebArena, Mind2Web, WebVoyager): WebArena grades against sandboxed backend state (doesn't transfer — we don't own the sites), Mind2Web grades trajectory fidelity (wrong axis — we care about output correctness), **WebVoyager's LLM-as-judge pattern** (instruction + final answer + evidence → pass/fail) is the closest fit for "arbitrary caller-defined JSON output." BrowserGym + WebArena is a reasonable future public-benchmark target for external comparability.
- **Provider-agnostic tool calling**: LiteLLM only normalizes the wire format (translation, not validation); Instructor is a good validate-and-reask pattern but not a full loop; **Pydantic AI** is a small, readable, genuinely vendor-agnostic agent framework (swap the `Model`, keep the code) that already builds in Instructor-style retry-on-validation-failure. Best fit given the "good learning example" requirement — small enough to read end-to-end.

**Build vs. borrow call**: don't take browser-use as a dependency. The perception layer (DOM → indexed elements) and the agent loop (perceive → decide → act → repeat) are the actual learning content of this project, so hand-roll both, informed by browser-use's technique. Use Playwright directly for browser control (infra, not agent logic) and Pydantic AI only as the thin multi-provider model-calling seam (not its full agent abstraction) — this keeps the loop's logic visible in our own code rather than hidden inside a framework.

## 3. Architecture

```
                         ┌───────────────────────────┐
   CLI (typer)  ───────► │                           │
                         │       Core Agent           │
   MCP Server   ───────► │  (run_task orchestrator)   │
   (mcp SDK)             │                           │
                         └─────────────┬─────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
     ┌──────────────────┐   ┌────────────────────┐    ┌───────────────────────┐
     │ LLM Adapter       │   │ Browser Controller  │    │ Output Validator       │
     │ (Pydantic AI Model│   │ (Playwright +        │    │ (JSON Schema ⇄        │
     │  swap: Anthropic /│   │  DOM→element         │    │  Pydantic model,       │
     │  OpenAI / etc.)   │   │  extractor)          │    │  reask-on-failure)     │
     └──────────────────┘   └────────────────────┘    └───────────────────────┘
                                       │
                              ┌────────┴────────┐
                              │ Safety Policy    │
                              │ (action allow /  │
                              │  deny classifier)│
                              └─────────────────┘

     Evals harness (separate, offline): fixtures → runs Core Agent → LLM-judge grading → report
```

### 3.1 Core Agent loop (`run_task`)

The heart of the project — one function, hand-written and readable, roughly:

```
def run_task(task: str, output_schema: dict | None, url: str, guardrails: Guardrails) -> AgentResult:
    browser = BrowserController.launch()
    browser.goto(url)
    history = []
    for step in range(guardrails.max_steps):
        if elapsed() > guardrails.timeout: return AgentResult.partial_failure(history, "timeout")
        observation = browser.observe()               # indexed elements + page text summary
        action = llm.decide_next_action(task, observation, history, output_schema)
        if action.is_final_answer:
            return validate_and_return(action.answer, output_schema)
        if not safety_policy.allowed(action):
            return AgentResult.blocked(action, history)
        browser.execute(action)
        history.append((observation, action))
    return AgentResult.partial_failure(history, "max_steps_exceeded")
```

This loop, the observation format, and the action schema are the parts of the codebase meant to be read and understood — everything else (Playwright plumbing, provider SDK details) is infrastructure.

### 3.2 Browser Controller / perception layer

- Own module, `Playwright` (Python, sync or async API) underneath.
- Borrow browser-use's technique: inject a small JS snippet to walk the DOM, filter to visible+interactive elements (links, buttons, inputs, selects), assign each a stable `index` for this observation, build a `selector_map: index -> ElementHandle`.
- Observation sent to the LLM = compact text: page title/URL, a numbered list of interactive elements with role + accessible name + (for inputs) current value, plus a trimmed text summary of visible content. Not raw HTML, not a screenshot (keeps token cost and latency low, matches the "not too expensive" NFR).
- Actions: `click(index)`, `type(index, text)`, `select(index, option)`, `scroll(direction)`, `navigate(url)`, `go_back()`, `finish(answer)`.
- Multi-page navigation is handled naturally — `navigate`/`click` on a link changes the page, next `observe()` reflects the new DOM.

### 3.3 Safety policy (autonomy scoping)

Given "read + safe interactions, no irreversible actions":

- Maintain a small denylist of action signatures that are blocked by default: submit buttons / element text matching patterns like *buy, purchase, checkout, pay, delete, remove account, send message, confirm order*.
- `type`/`click`/`select`/`navigate`/search-box interactions and login-form submission are allowed by default.
- Any denylisted action requires the caller to pass `allow_state_changing=True` explicitly for that run — off by default. This is intentionally a simple heuristic classifier for v1, not a general-purpose safety model; flag as an area to revisit if false positives/negatives show up in evals.

### 3.4 Output Validator

- Caller provides **either**:
  - a JSON Schema → converted to a dynamic Pydantic model (`pydantic.create_model` from schema, or `datamodel-code-generator`-style translation) → strict validation.
  - a natural-language description → wrapped in a generic `{"result": ...}` contract, best-effort validation (valid JSON, matches described intent via a lightweight LLM self-check) rather than strict schema conformance.
- On validation failure, reask the model once or twice with the validation error appended (Instructor's pattern) before failing the run.

### 3.5 Guardrails (sizing the open parameter)

Defaults, overridable per call:
- `max_steps`: 25
- `timeout`: 180s wall-clock
- `max_tokens` or `max_llm_calls`: cap to bound cost (exact number tunable once evals give us a sense of typical task cost)
- On any limit hit: return a structured **partial failure** result (not a silent truncation) — includes the step history so the caller can see how far it got.

### 3.6 Interfaces

- **CLI** (`typer`): `webagent run --url ... --task "..." --schema schema.json` → prints structured JSON result to stdout, non-zero exit code on failure/blocked. Scriptable/subprocess-friendly for other agents that just shell out.
- **MCP server** (Python `mcp` SDK): exposes a single tool, e.g. `browse_web(task, url, output_schema?, allow_state_changing?, guardrails?)`, returning the same structured result. This is the native "callable from other agents" path for MCP-aware hosts (Claude Code, Claude Desktop, etc.) — no subprocess wrapping needed.
- Both interfaces call the same `run_task()` — no logic duplicated between them.

### 3.7 Provider abstraction

- `LLM Adapter` = thin usage of Pydantic AI's `Model` abstraction for the "decide next action" and "produce final structured answer" calls.
- Model choice passed as a config string (`anthropic:claude-...`, `openai:gpt-...`) — mirrors browser-use's provider-prefix pattern, easy to extend.
- Keep our own code responsible for *what* we ask the model (the loop, the observation format, the action schema) — Pydantic AI only owns *how* that request reaches whichever provider is configured.

### 3.8 Vision fallback (set-of-mark screenshots)

> Deferred past v1 (see §6), but the perception layer should leave room for it: `Observation` gains an optional `screenshot` field from day one, even if phase 1 never populates it.

**Motivation.** The indexed-element DOM extraction (§3.2) breaks down for canvas-rendered UIs, broken/missing accessibility trees, and layouts where visual position or appearance *is* the information (e.g. "which button is highlighted," a calendar widget, a map, a CAPTCHA). Text-only perception has no signal in these cases. Rather than switching wholesale to screenshots (which reintroduces the cost/latency problems §2 already ruled out), add vision as an **escalation path** layered on top of the existing indexed-element mechanism.

**Set-of-mark annotation.** The key idea: don't send a screenshot as an unrelated second channel — annotate it with the *same* indices already assigned during DOM extraction, so the model has one grounding key across both modalities (a technique used by GPT-4V/SoM-style web agents). Concretely:

- During the existing DOM walk (§3.2), after elements are tagged with `data-webagent-index`, draw a colored bounding box + index label over each element (either via injected in-page CSS/overlay before calling `page.screenshot()`, or by post-processing the raw screenshot in Python with the element bounding-box coordinates already collected).
- Result: index `4` means the same thing whether the model is reading `[4] link "Pricing"` in the text list or seeing a labeled box around the Pricing link in the image. `click(index=4)` is unambiguous regardless of which channel surfaced it.

**Observation model.**
```python
class Observation(BaseModel):
    title: str
    url: str
    elements: list[ElementInfo]
    text_summary: str
    screenshot: bytes | None = None   # set-of-mark annotated PNG; populated only on escalation
```

**Trigger mechanism — screenshots are an escalation, not the default.** Two complementary paths, so a typical step never pays image-token cost:
- *Model-requested*: add an 8th `Action` variant, `RequestScreenshotAction` — doesn't touch the browser, just tells the loop to include an annotated screenshot on the *next* observation. The model reaches for this when the text observation feels insufficient for the task at hand.
- *Heuristic auto-trigger*: `BrowserController.observe()` can also attach a screenshot on its own initiative when it detects low-signal conditions — a `<canvas>` covering most of the viewport, a suspiciously low interactive-element count relative to visible page area, or the loop detecting no progress (identical observation twice in a row).

**Sending it to the model.** Swap the plain-string prompt for Pydantic AI's multimodal content list only when a screenshot is present:
```python
prompt = [observation.to_prompt()]
if observation.screenshot:
    prompt.append(BinaryContent(data=observation.screenshot, media_type="image/png"))
result = await agent.run(prompt, message_history=message_history)
```
(Verify exact Pydantic AI multimodal content-part API against current docs at implementation time — surface may shift between versions.)

**Cost guardrail.** Because screenshots are materially more expensive than the text path, they get their own budget, separate from `max_steps` — e.g. `max_screenshots` per task — so a model that gets stuck and repeatedly requests screenshots can't quietly blow the cost budget while still under the step cap. This slots into the Guardrails sizing already discussed in §3.5.

**Where this lands in the build order (§7):** insert as a new step after the existing 8 (Vision fallback / set-of-mark screenshots), since it depends on the indexed-element perception layer (step 1), the Action/Observation schema (steps 1–2), and ideally sits after Guardrails (step 4) so `max_screenshots` has somewhere to live.

## 4. Evals

- Fixture format (YAML/JSON), one per test case:
  ```yaml
  id: find-pricing-page
  url: https://example.com
  task: "Find the URL of the pricing page"
  output_schema: { type: object, properties: { pricing_url: {type: string} } }
  expected: { pricing_url: "https://example.com/pricing" }
  grading: exact_match   # or: llm_judge
  ```
- Grading modes:
  - `exact_match` / field-level comparison for deterministic fields (URLs, booleans, counts).
  - `llm_judge` (WebVoyager-style) for open-ended fields: judge sees {task, schema, agent output, final page evidence} → pass/fail + rationale.
- `webagent evals run` — runs the fixture suite against the current agent, produces a pass/fail report; track results over time (simple JSON/SQLite log) to see accuracy trend as the agent changes.
- Start with a small hand-built suite (10-20 fixtures across a handful of real sites) for fast iteration. Treat BrowserGym + WebArena adoption as a later milestone once the core loop is stable, for external comparability.

## 5. Tech stack

- Python 3.13, `playwright` (Python bindings), `pydantic` + `pydantic-ai`, `typer` (CLI), `mcp` (official Python MCP SDK), `pytest` for unit tests, custom `evals/` harness for agent-level evals.

## 6. Open questions / future work

- Exact `max_steps` / `timeout` / cost-cap defaults — size these empirically once the eval suite is running and we can see real per-task cost/step counts.
- Session/auth persistence (deferred out of v1) — revisit if real usage needs repeated logins to the same site.
- Whether to adopt BrowserGym + WebArena for public-benchmark comparability once the core agent is stable.
- Safety-policy denylist is a heuristic v1; consider a small LLM-based risk classifier per action if false positives/negatives show up in practice.
- Vision fallback (screenshot-based) — designed in §3.8 (set-of-mark annotated screenshots, escalation-triggered, own cost budget); not in v1 scope, scheduled as a build-order step after Guardrails.

## 7. Suggested build order

1. Browser Controller: Playwright launch + navigate + indexed-element DOM extraction + action execution.
2. Core Agent loop against one provider (Anthropic direct), hardcoded simple task, no schema — prove the perceive/decide/act loop works end to end.
3. Output Validator: JSON Schema → Pydantic model, reask-on-failure.
4. Guardrails: max_steps/timeout/partial-failure result shape.
5. Safety policy denylist + `allow_state_changing` flag.
6. CLI wrapper.
7. Eval fixture suite (10-20 cases) + exact-match and LLM-judge grading + `evals run` command.
8. MCP server wrapper.
9. Vision fallback: set-of-mark annotated screenshots, `RequestScreenshotAction` + heuristic auto-trigger, `max_screenshots` guardrail (§3.8).
