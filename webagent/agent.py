from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, NativeToolCallPart, ToolCallPart

from webagent.actions import Action, FinishAction, PageSnapshot
from webagent.browser import BrowserController

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
"""


def _format_elements(observation: PageSnapshot) -> str:
    if not observation.elements:
        return "(none found)"
    lines = []
    for el in observation.elements:
        role_part = f" role={el.role}" if el.role else ""
        lines.append(f"[{el.index}] <{el.tag}>{role_part} {el.name!r}")
    return "\n".join(lines)


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
    model: str = "anthropic:claude-sonnet-5",
    max_steps: int = 25,
    headless: bool = True,
) -> str:
    agent: Agent[None, Action] = Agent(
        model,
        output_type=Action,
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(task=task),
        model_settings={"thinking": "medium"},
    )

    browser = await BrowserController.launch(headless=headless)
    message_history: list[ModelMessage] | None = None
    try:
        await browser.goto(url)
        for step in range(max_steps):
            observation = await browser.observe()
            logger.info("step %d elements:\n%s", step, _format_elements(observation))
            logger.info(
                "step %d page summary: %s",
                step,
                _truncate(observation.text_summary, _SUMMARY_PREVIEW_CHARS),
            )
            logger.debug("step %d full observation:\n%s", step, observation.to_prompt())
            result = await agent.run(observation.to_prompt(), message_history=message_history)
            for message in result.new_messages():
                if isinstance(message, ModelResponse):
                    _log_model_response(message)
            message_history = result.new_messages()
            action = result.output
            logger.info("step %d action: %r", step, action)
            if isinstance(action, FinishAction):
                logger.info("finished after %d steps: %s", step + 1, action.answer)
                return action.answer
            await browser.execute(action)
        logger.warning("max_steps_exceeded after %d steps", max_steps)
        return "max_steps_exceeded"
    finally:
        await browser.close()
