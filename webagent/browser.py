from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import Browser, Locator, Page, Playwright, async_playwright

from webagent.actions import (
    BrowserAction,
    ClickAction,
    GoBackAction,
    NavigateAction,
    ScrollAction,
    SelectAction,
    TypeAction,
)
from webagent.page_snapshot import PageSnapshot

logger = logging.getLogger(__name__)

_EXTRACT_JS = (Path(__file__).parent / "js" / "extract_elements.js").read_text(encoding="utf-8")


class ElementNotFoundError(Exception):
    """Raised when an action's index no longer resolves to exactly one element.

    Checked eagerly via Locator.count() (instant, no waiting) rather than letting
    a doomed click()/fill()/select_option() run out Playwright's full actionability
    timeout - the index was tagged during a prior observe() and may be stale by the
    time an action executes (e.g. the page re-rendered in between).
    """

    def __init__(self, index: int, count: int) -> None:
        self.index = index
        self.count = count
        if count == 0:
            message = f"No element currently matches index {index} - it may no longer be present on the page."
        else:
            message = f"Index {index} unexpectedly matches {count} elements."
        super().__init__(message)


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

    async def execute(self, action: BrowserAction) -> None:
        logger.debug("executing action: %r", action)
        if isinstance(action, ClickAction):
            await (await self._resolve_locator(action.index)).click()
        elif isinstance(action, TypeAction):
            await (await self._resolve_locator(action.index)).fill(action.text)
        elif isinstance(action, SelectAction):
            await (await self._resolve_locator(action.index)).select_option(action.option)
        elif isinstance(action, ScrollAction):
            delta = 600 if action.direction == "down" else -600
            await self._page.mouse.wheel(0, delta)
        elif isinstance(action, NavigateAction):
            await self._page.goto(action.url)
        elif isinstance(action, GoBackAction):
            await self._page.go_back()
        else:
            raise TypeError(f"Unknown action type: {action!r}")

        # click() doesn't block on navigation it may have triggered (goto()/go_back()
        # already wait for load internally) - without this, observe() can run against
        # a mid-navigation document and see an empty body.
        await self._page.wait_for_load_state("domcontentloaded")

    def _locator(self, index: int) -> Locator:
        return self._page.locator(f'[data-webagent-index="{index}"]')

    async def _resolve_locator(self, index: int) -> Locator:
        locator = self._locator(index)
        count = await locator.count()
        if count != 1:
            raise ElementNotFoundError(index=index, count=count)
        return locator

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()
