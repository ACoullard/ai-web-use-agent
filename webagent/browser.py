from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import Browser, Page, Playwright, async_playwright

from webagent.actions import (
    Action,
    ClickAction,
    FinishAction,
    GoBackAction,
    NavigateAction,
    PageSnapshot,
    ScrollAction,
    SelectAction,
    TypeAction,
)

logger = logging.getLogger(__name__)

_EXTRACT_JS = (Path(__file__).parent / "js" / "extract_elements.js").read_text(encoding="utf-8")


class BrowserController:
    def __init__(self, playwright: Playwright, browser: Browser, page: Page) -> None:
        self._playwright = playwright
        self._browser = browser
        self._page = page

    @classmethod
    async def launch(cls, headless: bool = True) -> "BrowserController":
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page()
        return cls(playwright, browser, page)

    async def goto(self, url: str) -> None:
        await self._page.goto(url)

    async def observe(self) -> PageSnapshot:
        result = await self._page.evaluate(_EXTRACT_JS)
        return PageSnapshot.model_validate(result)

    async def execute(self, action: Action) -> None:
        logger.debug("executing action: %r", action)
        if isinstance(action, ClickAction):
            await self._locator(action.index).click()
        elif isinstance(action, TypeAction):
            await self._locator(action.index).fill(action.text)
        elif isinstance(action, SelectAction):
            await self._locator(action.index).select_option(action.option)
        elif isinstance(action, ScrollAction):
            delta = 600 if action.direction == "down" else -600
            await self._page.mouse.wheel(0, delta)
        elif isinstance(action, NavigateAction):
            await self._page.goto(action.url)
        elif isinstance(action, GoBackAction):
            await self._page.go_back()
        elif isinstance(action, FinishAction):
            raise ValueError("FinishAction should be handled by the agent loop, not executed")
        else:
            raise TypeError(f"Unknown action type: {action!r}")

        # click() doesn't block on navigation it may have triggered (goto()/go_back()
        # already wait for load internally) - without this, observe() can run against
        # a mid-navigation document and see an empty body.
        await self._page.wait_for_load_state("domcontentloaded")

    def _locator(self, index: int):
        return self._page.locator(f'[data-webagent-index="{index}"]')

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()
