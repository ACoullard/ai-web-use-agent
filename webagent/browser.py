import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from playwright.async_api import (
    Browser,
    ElementHandle,
    Error as PlaywrightError,
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

_JS_DIR = Path(__file__).parent / "js"
_EXTRACT_JS = (_JS_DIR / "extract_elements.js").read_text(encoding="utf-8")
_WATCH_DOM_CHANGES_JS = (_JS_DIR / "watch_dom_changes.js").read_text(encoding="utf-8")
_OCCLUDING_ELEMENT_JS = (_JS_DIR / "occluding_element.js").read_text(encoding="utf-8")
_FULL_TEXT_JS = "() => document.body ? document.body.innerText.trim() : ''"

TEXT_SUMMARY_CHARS = 2000
_SEARCH_CONTEXT_CHARS = 150

# Bounded wait for network quiet after a navigation, so observe() doesn't cache
# element references a moment before the page's own client-side JS replaces them
# (e.g. an initial SSR/cached paint that a data refetch swaps out shortly after
# load). Best-effort: some pages never go fully idle (polling, websockets,
# analytics beacons), so this is capped rather than awaited unconditionally.
_SETTLE_TIMEOUT_MS = 1000

# How long the DOM must go unmodified, with nothing in flight, before an action counts
# as finished - plus the ceiling on waiting for that and how often it's checked.
_DOM_QUIET_MS = 200
_ACTION_SETTLE_TIMEOUT_MS = 1500
_QUIET_POLL_INTERVAL_S = 0.05

# Ceiling on one element action.
_ACTION_TIMEOUT_MS = 5000

_QUIET_POLL_JS = """
(quietMs) => {
  const state = window.__webagentDomQuiet;
  // A navigation replaced the document the watcher was installed in - the load-state
  // wait already covered that, so there is nothing left to wait for here.
  if (!state) return true;
  return performance.now() - state.lastMutation >= quietMs;
}
"""


class ElementActionError(Exception):
    """Base for the reasons an action can't be attempted against its target element.

    Carries `advice`: what the agent should do about it, appended to the failure note fed
    back into the next prompt.
    """

    advice: str = ""


class ElementNotFoundError(ElementActionError):
    """Raised when an action's index no longer resolves to a live element on the page.

    Indices are cached during an observation, so they can go stale before the action
    using them runs - e.g. the page re-rendered and the referenced node was removed.
    """

    advice = (
        "The index you used no longer refers to anything - re-check the observation "
        "below before trying again."
    )

    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__(f"No element currently matches index {index} - it may no longer be present on the page.")


class ElementBlockedError(ElementActionError):
    """Raised when something is painted over the element a click would land on.

    The click is refused rather than attempted: it would hit the covering element
    instead, and Playwright would retry it until its actionability timeout expired.
    """

    advice = (
        "Deal with whatever is on top first - dismiss it, or use one of its own "
        "controls - then come back to this element. The observation below lists what "
        "is currently on the page."
    )

    def __init__(self, index: int, occluder: str) -> None:
        self.index = index
        self.occluder = occluder
        super().__init__(f"Element {index} is covered by {occluder}, which would receive the click instead.")


class BrowserController:
    def __init__(self, playwright: Playwright, browser: Browser, page: Page) -> None:
        self._playwright = playwright
        self._browser = browser
        self._page = page
        self._last_url: str | None = None
        self._text_offset: int = TEXT_SUMMARY_CHARS
        self._inflight = 0
        self._last_request_event = 0.0
        page.on("request", lambda _: self._track_request(1))
        page.on("requestfinished", lambda _: self._track_request(-1))
        page.on("requestfailed", lambda _: self._track_request(-1))

    def _track_request(self, delta: int) -> None:
        # Clamped: a request cancelled by a navigation can settle without a matching
        # start being seen, and a negative count would make every later wait time out.
        self._inflight = max(0, self._inflight + delta)
        self._last_request_event = time.monotonic()

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
            await self._act_on_element(action.index, lambda el: self._click(el, action.index))
        elif isinstance(action, TypeAction):
            await self._act_on_element(action.index, lambda el: el.fill(action.text, timeout=_ACTION_TIMEOUT_MS))
        elif isinstance(action, SelectAction):
            await self._act_on_element(
                action.index, lambda el: el.select_option(action.option, timeout=_ACTION_TIMEOUT_MS)
            )
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
        await self._settle_after_action()
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
                continue
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

    async def _settle_after_action(self) -> None:
        """Wait until the page stops reacting, so observe() sees what the action caused."""
        quiet_seconds = _DOM_QUIET_MS / 1000
        deadline = time.monotonic() + _ACTION_SETTLE_TIMEOUT_MS / 1000
        try:
            await self._page.evaluate(_WATCH_DOM_CHANGES_JS)
            while time.monotonic() < deadline:
                network_quiet = (
                    self._inflight == 0
                    and time.monotonic() - self._last_request_event >= quiet_seconds
                )
                if network_quiet and await self._page.evaluate(_QUIET_POLL_JS, _DOM_QUIET_MS):
                    return
                await asyncio.sleep(_QUIET_POLL_INTERVAL_S)
        except PlaywrightError:
            # A navigation can tear down the document the watcher was installed in; the
            # load-state wait before this call is what covers that case.
            pass

    async def _resolve_handle(self, index: int) -> ElementHandle:
        # Staleness is checked eagerly (handle lookup + isConnected, both instant) rather
        # than letting a doomed click()/fill()/select_option() run out Playwright's full
        # actionability timeout against an element that is already gone.
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

    async def _click(self, element: ElementHandle, index: int) -> None:
        """Click, refusing up front a click that something else would receive."""
        try:
            await element.scroll_into_view_if_needed(timeout=_ACTION_TIMEOUT_MS)
        except PlaywrightError:
            pass  # let click() below report why it couldn't be reached
        occluder = await element.evaluate(_OCCLUDING_ELEMENT_JS)
        if occluder is not None:
            raise ElementBlockedError(index=index, occluder=occluder)
        await element.click(timeout=_ACTION_TIMEOUT_MS)

    async def _act_on_element(self, index: int, action: Callable[[ElementHandle], Awaitable[None]]) -> None:
        element = await self._resolve_handle(index)
        try:
            await action(element)
        finally:
            await element.dispose()

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()
