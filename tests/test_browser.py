import asyncio
from contextlib import asynccontextmanager

import pytest

from webagent.actions import ClickAction, ReadMoreTextAction, SearchPageTextAction, TypeAction
from webagent.browser import TEXT_SUMMARY_CHARS, BrowserController, ElementNotFoundError


@asynccontextmanager
async def _launched_browser():
    # BrowserController's Playwright connection is bound to the asyncio event
    # loop it was launched under, so it must be created and closed within the
    # same asyncio.run() call as everything else in a test - it can't be shared
    # across tests via a fixture backed by its own asyncio.run().
    browser = await BrowserController.launch(headless=True)
    try:
        yield browser
    finally:
        await browser.close()


def _button_index(observation):
    return next(el.index for el in observation.elements if el.tag == "button")


def _input_index(observation):
    return next(el.index for el in observation.elements if el.tag == "input")


def test_click_basic_roundtrip(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "basic.html"
            html_path.write_text(
                "<button onclick=\"document.title = 'clicked'\">Click me</button>"
            )
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            await browser.execute(ClickAction(index=_button_index(observation)))
            assert await browser._page.title() == "clicked"

    asyncio.run(_test())


def test_click_survives_in_place_mutation(tmp_path):
    """The regression this change targets: a re-render that mutates the same DOM
    node (e.g. attributes added/removed) without replacing it must not break the
    index - only genuine node replacement should.
    """

    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "mutate.html"
            html_path.write_text(
                "<button onclick=\"document.title = 'clicked'\">Click me</button>"
            )
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            index = _button_index(observation)

            # Simulate a framework re-render that touches the node's attributes/class
            # in place without recreating it.
            await browser._page.evaluate(
                "() => { const b = document.querySelector('button'); "
                "b.className = 'rerendered'; b.setAttribute('data-something-else', '1'); }"
            )

            await browser.execute(ClickAction(index=index))
            assert await browser._page.title() == "clicked"

    asyncio.run(_test())


def test_click_raises_when_node_replaced(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "replace.html"
            html_path.write_text(
                "<button onclick=\"document.title = 'clicked'\">Click me</button>"
            )
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            index = _button_index(observation)

            # Simulate a genuine unmount/remount: a structurally-identical but
            # distinct node object replaces the original.
            await browser._page.evaluate(
                "() => { const b = document.querySelector('button'); "
                "b.replaceWith(b.cloneNode(true)); }"
            )

            with pytest.raises(ElementNotFoundError):
                await browser.execute(ClickAction(index=index))
            assert await browser._page.title() != "clicked"

    asyncio.run(_test())


def test_observe_excludes_inert_and_aria_hidden_elements(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "suppressed.html"
            html_path.write_text(
                "<button id='visible'>Visible</button>"
                "<button id='inert-btn' inert>Inert</button>"
                "<div aria-hidden='true'><button id='hidden-btn'>Hidden</button></div>"
            )
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            names = {el.name for el in observation.elements}
            assert names == {"Visible"}

    asyncio.run(_test())


def test_type_action_roundtrip(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "type.html"
            html_path.write_text("<input type='text' />")
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            await browser.execute(TypeAction(index=_input_index(observation), text="hello"))
            value = await browser._page.eval_on_selector("input", "el => el.value")
            assert value == "hello"

    asyncio.run(_test())


def _long_text_html(total_chars: int) -> str:
    body = "".join(f"word{i} " for i in range(total_chars // 7 + 1))[:total_chars]
    return f"<body>{body}</body>"


def test_observe_reports_truncated_text_total_length(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "long.html"
            html_path.write_text(_long_text_html(TEXT_SUMMARY_CHARS + 500))
            await browser.goto(html_path.as_uri())
            observation = await browser.observe()
            assert len(observation.text_summary) == TEXT_SUMMARY_CHARS
            assert observation.text_total_length >= TEXT_SUMMARY_CHARS + 500
            assert "more character" in observation.to_prompt()

    asyncio.run(_test())


def test_search_page_text_finds_match_with_context(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "search.html"
            html_path.write_text(_long_text_html(TEXT_SUMMARY_CHARS + 500) + "<p>needle-phrase found here</p>")
            await browser.goto(html_path.as_uri())
            await browser.observe()
            result = await browser.execute(SearchPageTextAction(query="needle-phrase"))
            assert "needle-phrase found here" in result

            miss = await browser.execute(SearchPageTextAction(query="not-present-anywhere"))
            assert "No matches" in miss

    asyncio.run(_test())


def test_search_page_text_supports_or_query(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "search_or.html"
            filler = "filler " * 100  # keeps the two markers' context windows from overlapping
            html_path.write_text(f"<p>alpha-marker here</p><p>{filler}</p><p>beta-marker here too</p>")
            await browser.goto(html_path.as_uri())
            await browser.observe()

            result = await browser.execute(SearchPageTextAction(query="alpha-marker|beta-marker"))
            assert "2 match(es)" in result
            assert "alpha-marker" in result
            assert "beta-marker" in result

            miss = await browser.execute(SearchPageTextAction(query="nope-1|nope-2"))
            assert "No matches" in miss

    asyncio.run(_test())


def test_search_page_text_merges_nearby_or_matches_into_one_snippet(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "search_or_nearby.html"
            # alpha-marker and beta-marker sit well within one another's context
            # window, so this must produce a single merged snippet, not two
            # overlapping/duplicate ones.
            html_path.write_text("<p>alpha-marker close to beta-marker here</p>")
            await browser.goto(html_path.as_uri())
            await browser.observe()

            result = await browser.execute(SearchPageTextAction(query="alpha-marker|beta-marker"))
            header, _, body = result.partition("\n\n")
            assert "1 match(es)" in header
            assert body.count("alpha-marker") == 1
            assert body.count("beta-marker") == 1

    asyncio.run(_test())


def test_read_more_text_paginates_and_resets_on_navigation(tmp_path):
    async def _test():
        async with _launched_browser() as browser:
            html_path = tmp_path / "read_more.html"
            html_path.write_text(_long_text_html(TEXT_SUMMARY_CHARS * 3))
            await browser.goto(html_path.as_uri())
            await browser.observe()

            first_chunk = await browser.execute(ReadMoreTextAction())
            assert len(first_chunk) == TEXT_SUMMARY_CHARS
            second_chunk = await browser.execute(ReadMoreTextAction())
            assert second_chunk != first_chunk

            other_path = tmp_path / "other.html"
            other_path.write_text("<body>short page</body>")
            await browser.goto(other_path.as_uri())
            await browser.observe()
            reset_chunk = await browser.execute(ReadMoreTextAction())
            assert "No more text remaining" in reset_chunk

    asyncio.run(_test())
