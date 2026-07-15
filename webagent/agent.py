from __future__ import annotations

import json
import logging
from typing import Any

from playwright.async_api import Error as PlaywrightError
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, NativeToolCallPart, ToolCallPart

from webagent.actions import resolve_action_type
from webagent.browser import BrowserController, ElementNotFoundError
from webagent.output_spec import generic_answer_model, json_schema_to_model, self_check
from webagent.page_snapshot import PageSnapshot
from webagent.result import AgentResult

logger = logging.getLogger(__name__)

_SUMMARY_PREVIEW_CHARS = 200

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
- finish(answer): call this once you have completed the task, with your final answer

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
    model: str = "anthropic:claude-sonnet-5",
    max_steps: int = 25,
    max_reask_attempts: int = 2,
    headless: bool = True,
    dry_run: bool = False,
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

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(task=task, answer_instructions=answer_instructions)

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

    action_type = resolve_action_type(answer_model)

    agent: Agent[None, Any] = Agent(
        model,
        output_type=action_type,
        system_prompt=system_prompt,
        model_settings={"thinking": "medium"},
        retries={"tools": 1, "output": max_reask_attempts},
    )

    browser = await BrowserController.launch(headless=headless)
    message_history: list[ModelMessage] | None = None
    reask_attempts_used = 0
    pending_reask_note: str | None = None
    try:
        await browser.goto(url)
        step = 0
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

            result = await agent.run(prompt, message_history=message_history)
            for message in result.new_messages():
                if isinstance(message, ModelResponse):
                    _log_model_response(message)
            message_history = result.new_messages()
            action = result.output
            logger.info("step %d action: %r", step, action)

            if action.type == "finish":
                if answer_model is None:
                    logger.info("finished after %d steps: %s", step + 1, action.answer)
                    return AgentResult(status="success", answer=action.answer)

                if output_schema is not None:
                    logger.info("finished after %d steps with schema-validated answer", step + 1)
                    return AgentResult(
                        status="success",
                        answer=answer_adapter.dump_python(action.answer, mode="json"),
                    )

                # output_description mode: structurally valid ({"result": ...}), but
                # still needs a semantic self-check against the caller's description.
                verdict = await self_check(task, output_description, action.answer.result, model)
                if verdict.passes:
                    logger.info("finished after %d steps, self-check passed", step + 1)
                    return AgentResult(status="success", answer=action.answer.model_dump())

                if reask_attempts_used >= max_reask_attempts:
                    logger.warning(
                        "output validation failed after %d reask attempt(s): %s",
                        reask_attempts_used,
                        verdict.reason,
                    )
                    return AgentResult(
                        status="validation_failed",
                        error=verdict.reason,
                        attempts=reask_attempts_used,
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

            try:
                await browser.execute(action)
            except ElementNotFoundError as e:
                logger.warning("step %d action %r failed: %s", step, action, e)
                pending_reask_note = (
                    f"Your last action ({action!r}) failed: {e} "
                    "The index you used no longer refers to anything - re-check the "
                    "observation below before trying again."
                )
            except PlaywrightError as e:
                logger.warning("step %d action %r failed: %s", step, action, e)
                pending_reask_note = (
                    f"Your last action ({action!r}) failed: {e}."
                )
            step += 1
        logger.warning("max_steps_exceeded after %d steps", max_steps)
        return AgentResult(status="max_steps_exceeded")
    finally:
        await browser.close()
