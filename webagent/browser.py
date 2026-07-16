from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable

from playwright.async_api import (
    Browser,
    ElementHandle,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from webagent.actions import (
    BrowserAction,
    ClickAction,
    GoBackAction,
    NavigateAction,
    ReadMoreTextAction,
    ScrollAction,
    SearchPageTextAction,
    SelectAction,
    TypeAction,
)
from webagent.page_snapshot import PageSnapshot

logger = logging.getLogger(__name__)

_EXTRACT_JS = (Path(__file__).parent / "js" / "extract_elements.js").read_text(encoding="utf-8")
_FULL_TEXT_JS = "() => document.body ? document.body.innerText.trim() : ''"

TEXT_SUMMARY_CHARS = 2000
_SEARCH_CONTEXT_CHARS = 150

# Bounded wait for network quiet after a navigation, so observe() doesn't cache
# element references a moment before the page's own client-side JS replaces them
# (e.g. an initial SSR/cached paint that a data refetch swaps out shortly after
# load). Best-effort: some pages never go fully idle (polling, websockets,
# analytics beacons), so this is capped rather than awaited unconditionally.
_SETTLE_TIMEOUT_MS = 1000


class ElementNotFoundError(Exception):
    """Raised when an action's index no longer resolves to a live element.

    Checked eagerly via an element-handle lookup plus an isConnected check (both
    instant, no waiting) rather than letting a doomed click()/fill()/select_option()
    run out Playwright's full actionability timeout - the index was cached during a
    prior observe() and may be stale by the time an action executes (e.g. the page
    re-rendered and the referenced node was removed in between).
    """

    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__(f"No element currently matches index {index} - it may no longer be present on the page.")


class BrowserController:
    def __init__(self, playwright: Playwright, browser: Browser, page: Page) -> None:
        self._playwright = playwright
        self._browser = browser
        self._page = page
        self._last_url: str | None = None
        self._text_offset: int = TEXT_SUMMARY_CHARS

    @classmethod
    async def launch(cls, headless: bool = True) -> "BrowserController":
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page()
        return cls(playwright, browser, page)

    async def goto(self, url: str) -> None:
        await self._page.goto(url)
        await self._settle()

    async def observe(self) -> PageSnapshot:
        result = await self._page.evaluate(_EXTRACT_JS, TEXT_SUMMARY_CHARS)

        url = result.get("url")
        if url != self._last_url:
            self._last_url = url
            self._text_offset = TEXT_SUMMARY_CHARS
        return PageSnapshot.model_validate(result)

    async def execute(self, action: BrowserAction) -> str | None:
        logger.debug("executing action: %r", action)
        if isinstance(action, ClickAction):
            await self._act_on_element(action.index, lambda el: el.click())
        elif isinstance(action, TypeAction):
            await self._act_on_element(action.index, lambda el: el.fill(action.text))
        elif isinstance(action, SelectAction):
            await self._act_on_element(action.index, lambda el: el.select_option(action.option))
        elif isinstance(action, ScrollAction):
            delta = 600 if action.direction == "down" else -600
            await self._page.mouse.wheel(0, delta)
        elif isinstance(action, NavigateAction):
            await self._page.goto(action.url)
        elif isinstance(action, GoBackAction):
            await self._page.go_back()
        elif isinstance(action, SearchPageTextAction):
            return await self._search_page_text(action.query)
        elif isinstance(action, ReadMoreTextAction):
            return await self._read_more_text()
        else:
            raise TypeError(f"Unknown action type: {action!r}")

        # click() doesn't block on navigation it may have triggered (goto()/go_back()
        # already wait for load internally) - without this, observe() can run against
        # a mid-navigation document and see an empty body.
        await self._page.wait_for_load_state("domcontentloaded")
        await self._settle()
        return None

    async def _search_page_text(self, query: str) -> str:
        text = await self._page.evaluate(_FULL_TEXT_JS)
        lowered = text.lower()
        # `|` separates alternatives (grep -E style OR), e.g. "cat|dog" matches either
        # term - each alternative is still a plain substring match, not a full regex,
        # so a model-supplied query can't trigger catastrophic regex backtracking.
        terms = [t.strip() for t in query.split("|") if t.strip()]
        if not terms:
            return f"No matches for {query!r} in the page text."

        matches = []
        for term in terms:
            needle = term.lower()
            start = 0
            while True:
                found = lowered.find(needle, start)
                if found == -1:
                    break
                matches.append((found, found + len(term)))
                start = found + max(len(term), 1)
        matches.sort()

        snippets = []
        last_window_end = -1
        for match_start, match_end in matches:
            window_start = max(0, match_start - _SEARCH_CONTEXT_CHARS)
            window_end = min(len(text), match_end + _SEARCH_CONTEXT_CHARS)
            if window_start < last_window_end:
                continue  # skip matches whose context overlaps a snippet already included
            snippets.append(text[window_start:window_end].strip())
            last_window_end = window_end

        if not snippets:
            return f"No matches for {query!r} in the page text."
        return f"{len(snippets)} match(es) for {query!r}:\n\n" + "\n---\n".join(snippets)

    async def _read_more_text(self) -> str:
        text = await self._page.evaluate(_FULL_TEXT_JS)
        if self._text_offset >= len(text):
            return "No more text remaining on this page."
        chunk = text[self._text_offset : self._text_offset + TEXT_SUMMARY_CHARS]
        self._text_offset += len(chunk)
        return chunk

    async def _settle(self) -> None:
        try:
            await self._page.wait_for_load_state("networkidle", timeout=_SETTLE_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            pass

    async def _resolve_handle(self, index: int) -> ElementHandle:
        handle = await self._page.evaluate_handle(
            "(index) => (window.__webagentElements || [])[index] ?? null", index
        )
        element = handle.as_element()
        if element is None:
            await handle.dispose()
            raise ElementNotFoundError(index=index)
        if not await element.evaluate("el => el.isConnected"):
            await element.dispose()
            raise ElementNotFoundError(index=index)
        return element

    async def _act_on_element(self, index: int, action: Callable[[ElementHandle], Awaitable[None]]) -> None:
        element = await self._resolve_handle(index)
        try:
            await action(element)
        finally:
            await element.dispose()

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()
