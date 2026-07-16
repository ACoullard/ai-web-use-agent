import asyncio
from contextlib import asynccontextmanager

import pytest

from webagent.actions import ClickAction, TypeAction
from webagent.browser import BrowserController, ElementNotFoundError


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
