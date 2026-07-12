from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ClickAction(BaseModel):
    type: Literal["click"] = "click"
    index: int


class TypeAction(BaseModel):
    type: Literal["type"] = "type"
    index: int
    text: str


class SelectAction(BaseModel):
    type: Literal["select"] = "select"
    index: int
    option: str


class ScrollAction(BaseModel):
    type: Literal["scroll"] = "scroll"
    direction: Literal["up", "down"]


class NavigateAction(BaseModel):
    type: Literal["navigate"] = "navigate"
    url: str


class GoBackAction(BaseModel):
    type: Literal["go_back"] = "go_back"


class FinishAction(BaseModel):
    type: Literal["finish"] = "finish"
    answer: str


Action = Annotated[
    Union[
        ClickAction,
        TypeAction,
        SelectAction,
        ScrollAction,
        NavigateAction,
        GoBackAction,
        FinishAction,
    ],
    Field(discriminator="type"),
]


class ElementInfo(BaseModel):
    index: int
    tag: str
    role: str | None = None
    name: str
    value: str | None = None


class PageSnapshot(BaseModel):
    title: str
    url: str
    elements: list[ElementInfo]
    text_summary: str

    def to_prompt(self) -> str:
        lines = [f"Page title: {self.title}", f"URL: {self.url}", "", "Interactive elements:"]
        if self.elements:
            for el in self.elements:
                role_part = f" role={el.role}" if el.role else ""
                value_part = f" value={el.value!r}" if el.value else ""
                lines.append(f"[{el.index}] <{el.tag}>{role_part} {el.name!r}{value_part}")
        else:
            lines.append("(none found)")
        lines.append("")
        lines.append("Page text summary:")
        lines.append(self.text_summary or "(empty)")
        return "\n".join(lines)
