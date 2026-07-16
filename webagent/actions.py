from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, create_model


_INDEX_DESCRIPTION = (
    "The element index shown in brackets in the most recent observation's "
    "'Interactive elements' list, e.g. 5 for '[5] <a> ...'. Indices are reassigned "
    "on every observation, so only use an index from the latest one - never reuse "
    "an index from an earlier step."
)


class ClickAction(BaseModel):
    """Click an interactive element: a link, button, checkbox, radio button, or similar.

    Use this to follow a link, submit a form, or toggle a checkbox/radio from the
    current observation's numbered element list. Do not use this on a <select> element
    - use `select` instead. Clicking may trigger navigation to a new page; the next
    observation will reflect whatever page loads afterward.
    """

    type: Literal["click"] = "click"
    index: int = Field(description=_INDEX_DESCRIPTION)


class TypeAction(BaseModel):
    """Fill a text input, textarea, or contenteditable element with a given value.

    This replaces the element's current value entirely - it does not append to
    existing text. Use it for search boxes, login fields, and other form inputs. It
    does not submit the form; follow up with a separate `click` on a submit
    button/link if submission is required.
    """

    type: Literal["type"] = "type"
    index: int = Field(description=_INDEX_DESCRIPTION)
    text: str = Field(description="The full text to enter into the field, replacing any existing value.")


class SelectAction(BaseModel):
    """Choose an option in a <select> dropdown element.

    Only use this on an element whose observation entry lists `options=[...]` -
    `option` must match one of those options' quoted values, not their visible
    label. For example, given `options=['2' (February), '3' (March)]`, pass '2' to
    select February, not 'February'.
    """

    type: Literal["select"] = "select"
    index: int = Field(description=_INDEX_DESCRIPTION)
    option: str = Field(
        description=(
            "The option's value (the quoted string in the element's options=[...] "
            "list), not its visible label."
        )
    )


class ScrollAction(BaseModel):
    """Scroll the page up or down.

    Note: the 'Interactive elements' list already includes elements anywhere on the
    page, not just what's currently in the viewport, so scrolling is rarely needed
    just to make an element clickable by index. It's mainly useful for triggering
    content that lazy-loads on scroll. To read more of the page's text, use
    `search_page_text` or `read_more_text` instead - scrolling does not reveal more text.
    """

    type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"] = Field(description="Which direction to scroll.")


class SearchPageTextAction(BaseModel):
    """Search the full page text for a keyword or phrase.

    Use this when you need one specific fact from a page whose text summary was
    truncated, rather than reading through it sequentially. Returns matching snippets
    with surrounding context, not the whole page. Separate multiple terms with `|` to
    match any of them, e.g. 'cat|dog' matches text containing either "cat" or "dog".
    """

    type: Literal["search_page_text"] = "search_page_text"
    query: str = Field(
        description=(
            "The keyword or phrase to search for in the page's full text. Separate "
            "multiple terms with `|` for an OR search, e.g. 'cat|dog'."
        )
    )


class ReadMoreTextAction(BaseModel):
    """Continue reading the page's full text sequentially from where the last
    summary or read_more_text call left off.

    Use this when you need to read a long page top-to-bottom rather than search for
    one specific fact.
    """

    type: Literal["read_more_text"] = "read_more_text"


class NavigateAction(BaseModel):
    """Go directly to a URL, bypassing element clicks.

    Use this when you already know the destination URL - e.g. an element's `href` in
    the observation, or a discovery link (such as an RSS/Atom feed URL) from the
    'Discovery links' section - instead of trying to click something that points there.
    """

    type: Literal["navigate"] = "navigate"
    url: str = Field(description="A fully-qualified URL to navigate to, e.g. https://example.com/page.")


class GoBackAction(BaseModel):
    """Return to the previous page in browser history, like clicking the browser's back button.

    Use this to back out of a page that turned out to be a dead end, without having to
    re-navigate or re-search from the start.
    """

    type: Literal["go_back"] = "go_back"


class FinishAction(BaseModel):
    """Signal that the task is complete and report the final answer.

    Call this exactly once, as soon as you have the information the task asked for -
    do not keep browsing after you already know the answer. No further actions are
    taken once this is returned.
    """

    type: Literal["finish"] = "finish"
    answer: str = Field(description="The final answer to the task, as plain text.")


_BROWSER_ACTION_TYPES = (
    ClickAction,
    TypeAction,
    SelectAction,
    ScrollAction,
    NavigateAction,
    GoBackAction,
    SearchPageTextAction,
    ReadMoreTextAction,
)

BrowserAction = Annotated[Union[_BROWSER_ACTION_TYPES], Field(discriminator="type")]


def resolve_action_type(answer_model: Any | None) -> type:
    """Resolve the Action union to use for a run_task() call: BrowserAction plus a finish variant.

    With no answer_model, the finish variant is just the static FinishAction above
    (answer: str). With an answer_model (from a caller-supplied output schema/
    description) - a BaseModel for object schemas, or a generic alias like
    `list[SomeModel]` for a top-level array schema - a fresh finish variant is built
    for this call whose `answer` field is typed to match. Since it's a new class each
    time, callers must check `action.type == "finish"` rather than
    `isinstance(action, FinishAction)`.
    """
    if answer_model is None:
        finish_action = FinishAction
    else:
        finish_action = create_model(
            "FinishAction",
            __doc__=(
                "Signal that the task is complete and report the final answer. Call this "
                "exactly once, as soon as you have the information the task asked for - "
                "do not keep browsing after you already know the answer. `answer` must "
                "conform to the required output contract described in the task "
                "instructions, not plain free text."
            ),
            type=(Literal["finish"], "finish"),
            answer=(
                answer_model,
                Field(description="The final answer, structured to match the required output contract."),
            ),
        )

    return Annotated[
        Union[_BROWSER_ACTION_TYPES + (finish_action,)],
        Field(discriminator="type"),
    ]
