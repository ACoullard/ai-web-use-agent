import json
import logging
import time
from typing import Any

from playwright.async_api import Error as PlaywrightError
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    NativeToolCallPart,
    ToolCallPart,
    UserPromptPart,
)

from webagent.actions import resolve_action_type
from webagent.browser import BrowserController, ElementActionError
from webagent.config import capture_full_input, request_capture_path
from webagent.output_spec import generic_answer_model, json_schema_to_model, self_check
from webagent.page_snapshot import PageSnapshot
from webagent.providers import (
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    build_model_settings,
    check_model_config,
    resolve_thinking,
)
from webagent.result import AgentResult
from webagent.traces import CapturingModel, NullTracer, Tracer

logger = logging.getLogger(__name__)

_SUMMARY_PREVIEW_CHARS = 200

# How many observations the model sees verbatim per request, counting the current one.
# Everything older is replaced by _AGED_OBSERVATION_STUB; the agent carries what it still
# needs forward in each action's `memory` field.
_HISTORY_WINDOW = 2

# Marks the start of an observation inside a user prompt. Anything a step prepended
# before it (a reask note, a failed-action note) is worth keeping, so the stub replaces
# only the observation itself - see _stub_aged_observations.
_OBSERVATION_MARKER = "Page title: "
_AGED_NOTE_MAX_CHARS = 200

_AGED_OBSERVATION_STUB = (
    "[Page state from an earlier step, omitted to save context. Element indices from "
    "it are stale and no longer valid. What you recorded in `memory` is your record of "
    "what happened here.]"
)

SYSTEM_PROMPT_TEMPLATE = """\
You are a web browsing agent. Your task is:

{task}

You perceive the page as a numbered list of interactive elements plus a text summary.
On each turn, respond with exactly one action:
- click(index): click an element
- type(index, text): fill a text input
- select(index, option): choose an option in a <select>
- scroll(direction): scroll "up" or "down"
- navigate(url): go directly to a URL
- go_back(): return to the previous page
- search_page_text(query): search the page's full text for a keyword/phrase when the
  text summary was truncated and you need one specific fact
- read_more_text(): keep reading the page's full text sequentially, continuing from
  where the summary or the last read_more_text() call left off
- finish(answer): call this once you have completed the task, with your final answer

Every action also takes a `memory` field. Only the {history_window} most recent observations stay
in your context - older ones are replaced by a placeholder, so `memory` is the only
record you keep of earlier steps. Use it to note what the last action achieved and how
far along the task is, and always carry a count when the task asks for several of
something ("added 1 of 2 products"). Re-read your own last `memory` before deciding: it
tells you what you have already done, and the task above tells you what is still left.

Only refer to element indices that appear in the most recent observation - they change every step.
Call finish() as soon as you have the answer; do not keep browsing after you know the answer.
{answer_instructions}\
"""

_SCHEMA_ANSWER_INSTRUCTIONS = """
Your finish() answer must be a JSON object matching this schema:
{schema}
"""

_DESCRIPTION_ANSWER_INSTRUCTIONS = """
Your finish() answer must be a JSON object of the form {{"result": ...}}, where the
value of "result" satisfies this description: {description}
"""


def _format_elements(observation: PageSnapshot) -> str:
    if not observation.elements:
        return "(none found)"
    return "\n".join(el.to_prompt_line() for el in observation.elements)


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _stub_aged_observations(
    messages: list[ModelMessage], keep_last: int = _HISTORY_WINDOW - 1
) -> list[ModelMessage]:
    """Replace all but the `keep_last` most recent observations with a short stub.

    Rewrites the user prompts in place rather than dropping messages: slicing a
    message list risks orphaning a tool call from its result, or a reasoning item
    from the call it belongs to, which providers reject. Editing in place keeps every
    request/response pair intact and only shrinks what it holds.

    Idempotent - the loop feeds the returned list back in as history each step, so an
    observation stubbed once stays stubbed and only the newly aged-out one is
    rewritten per step.
    """
    seen = 0
    for message in reversed(messages):
        for part in reversed(message.parts):
            if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                continue
            marker = part.content.find(_OBSERVATION_MARKER)
            if marker == -1:
                continue  # a note with no observation attached; leave it alone
            seen += 1
            if seen <= keep_last:
                continue
            note = part.content[:marker]
            if len(note) > _AGED_NOTE_MAX_CHARS:
                note = note[:_AGED_NOTE_MAX_CHARS].rstrip() + "... "
            part.content = note + _AGED_OBSERVATION_STUB
    return messages


def _log_model_response(response: ModelResponse) -> None:
    if response.thinking:
        logger.info("model reasoning:\n%s", response.thinking)
    for part in response.parts:
        if isinstance(part, (ToolCallPart, NativeToolCallPart)):
            logger.debug("raw tool call: %s(%s)", part.tool_name, part.args)


async def run_task(
    task: str,
    url: str,
    output_schema: dict[str, Any] | None = None,
    output_description: str | None = None,
    model: str = DEFAULT_MODEL,
    thinking: str | bool = resolve_thinking(DEFAULT_THINKING),
    max_steps: int = 25,
    max_reask_attempts: int = 2,
    headless: bool = True,
    dry_run: bool = False,
    tracer: Tracer | None = None,
) -> AgentResult:
    if output_schema is not None and output_description is not None:
        raise ValueError("Pass at most one of output_schema, output_description")

    answer_model: Any = None
    answer_adapter: TypeAdapter[Any] | None = None
    answer_instructions = ""
    if output_schema is not None:
        answer_model = json_schema_to_model(output_schema)
        answer_adapter = TypeAdapter(answer_model)
        answer_instructions = _SCHEMA_ANSWER_INSTRUCTIONS.format(schema=json.dumps(output_schema))
    elif output_description is not None:
        answer_model = generic_answer_model()
        answer_instructions = _DESCRIPTION_ANSWER_INSTRUCTIONS.format(description=output_description)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        task=task,
        answer_instructions=answer_instructions,
        history_window=_HISTORY_WINDOW,
    )

    if dry_run:
        browser = await BrowserController.launch(headless=headless)
        try:
            await browser.goto(url)
            observation = await browser.observe()
        finally:
            await browser.close()
        return AgentResult(
            status="dry_run",
            answer={"system_prompt": system_prompt, "observation_prompt": observation.to_prompt()},
        )

    check_model_config(model)
    model_settings = build_model_settings(model, thinking)

    run_started = time.monotonic()
    recording = (tracer or NullTracer()).start(
        task=task,
        url=url,
        model=model if isinstance(model, str) else str(model),
        thinking="off" if thinking is False else thinking if isinstance(thinking, str) else "on",
        output_mode=(
            "schema" if output_schema is not None else "description" if output_description is not None else "freeform"
        ),
        system_prompt=system_prompt,
    )

    def _persist(result: AgentResult) -> AgentResult:
        """Close the recording on any exit path, stamping its id onto the result."""
        recording.finish(
            status=result.status,
            steps_taken=result.steps_taken,
            duration=time.monotonic() - run_started,
        )
        result.trace_id = recording.trace_id
        return result

    action_type = resolve_action_type(answer_model)

    browser = await BrowserController.launch(headless=headless)
    message_history: list[ModelMessage] | None = None
    step = 0
    reask_attempts_used = 0
    pending_reask_note: str | None = None

    model_to_run: Any = model
    if capture_full_input() and recording.trace_id:
        model_to_run = CapturingModel(model, request_capture_path(recording.trace_id), lambda: step)

    agent: Agent[None, Any] = Agent(
        model_to_run,
        output_type=action_type,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries={"tools": 1, "output": max_reask_attempts},
    )

    try:
        await browser.goto(url)
        while step < max_steps:
            observation = await browser.observe()
            logger.info("step %d elements:\n%s", step, _format_elements(observation))
            logger.info(
                "step %d page summary: %s",
                step,
                _truncate(observation.text_summary, _SUMMARY_PREVIEW_CHARS),
            )
            logger.debug("step %d full observation:\n%s", step, observation.to_prompt())

            prompt = observation.to_prompt()
            if pending_reask_note is not None:
                prompt = f"{pending_reask_note}\n\n{prompt}"
                pending_reask_note = None

            gen_started = time.monotonic()
            result = await agent.run(prompt, message_history=message_history)
            gen_duration = time.monotonic() - gen_started
            for message in result.new_messages():
                if isinstance(message, ModelResponse):
                    _log_model_response(message)
            recording.record_generation(step, prompt, result, gen_duration)
            message_history = _stub_aged_observations(result.all_messages())
            action = result.output
            logger.debug("step %d memory: %s", step, action.memory)
            logger.info("step %d action: %r", step, action)

            if action.type == "finish":
                if answer_model is None:
                    logger.info("finished after %d steps: %s", step + 1, action.answer)
                    return _persist(AgentResult(status="success", answer=action.answer, steps_taken=step + 1))

                if output_schema is not None:
                    logger.info("finished after %d steps with schema-validated answer", step + 1)
                    return _persist(
                        AgentResult(
                            status="success",
                            answer=answer_adapter.dump_python(action.answer, mode="json"),
                            steps_taken=step + 1,
                        )
                    )

                # output_description mode: structurally valid ({"result": ...}), but
                # still needs a semantic self-check against the caller's description.
                sc_started = time.monotonic()
                sc_result = await self_check(
                    task, output_description, action.answer.result, model, model_settings
                )
                verdict = sc_result.output
                sc_input = (
                    "Does the produced result satisfy the expected output description?\n"
                    f"Description: {output_description}\n"
                    f"Result: {json.dumps(action.answer.result, default=str)}"
                )
                recording.record_self_check(step, sc_input, sc_result, time.monotonic() - sc_started)
                if verdict.passes:
                    logger.info("finished after %d steps, self-check passed", step + 1)
                    return _persist(
                        AgentResult(status="success", answer=action.answer.model_dump(), steps_taken=step + 1)
                    )

                if reask_attempts_used >= max_reask_attempts:
                    logger.warning(
                        "output validation failed after %d reask attempt(s): %s",
                        reask_attempts_used,
                        verdict.reason,
                    )
                    return _persist(
                        AgentResult(
                            status="validation_failed",
                            error=verdict.reason,
                            attempts=reask_attempts_used,
                            steps_taken=step,
                        )
                    )

                reask_attempts_used += 1
                logger.info(
                    "self-check failed (attempt %d/%d): %s",
                    reask_attempts_used,
                    max_reask_attempts,
                    verdict.reason,
                )
                pending_reask_note = (
                    f"Your finish() answer did not satisfy the task: {verdict.reason}. "
                    "Please reconsider and call finish() again."
                )
                continue  # a reask attempt doesn't consume a browsing step

            tool_started = time.monotonic()
            try:
                action_result = await browser.execute(action)
                if action_result is not None:
                    pending_reask_note = action_result
                recording.record_tool(
                    action, step, time.monotonic() - tool_started, status="ok", result=action_result
                )
            except ElementActionError as e:
                recording.record_tool(
                    action, step, time.monotonic() - tool_started, status="error", error=str(e)
                )
                logger.warning("step %d action %r failed: %s", step, action, e)
                pending_reask_note = f"Your last action ({action!r}) failed: {e} {e.advice}"
            except PlaywrightError as e:
                recording.record_tool(
                    action, step, time.monotonic() - tool_started, status="error", error=str(e)
                )
                logger.warning("step %d action %r failed: %s", step, action, e)
                pending_reask_note = (
                    f"Your last action ({action!r}) failed: {e}."
                )
            step += 1
        logger.warning("max_steps_exceeded after %d steps", max_steps)
        return _persist(AgentResult(status="max_steps_exceeded", steps_taken=step))
    finally:
        recording.finish(
            status="interrupted", steps_taken=step, duration=time.monotonic() - run_started
        )
        await browser.close()
