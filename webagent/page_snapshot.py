from pydantic import BaseModel


class SelectOption(BaseModel):
    value: str
    label: str


class ElementInfo(BaseModel):
    index: int
    tag: str
    role: str | None = None
    name: str
    value: str | None = None
    href: str | None = None
    options: list[SelectOption] | None = None

    def to_prompt_line(self) -> str:
        role_part = f" role={self.role}" if self.role else ""
        value_part = f" value={self.value!r}" if self.value else ""
        href_part = f" href={self.href!r}" if self.href else ""
        options_part = ""
        if self.options:
            rendered = ", ".join(f"{o.value!r} ({o.label})" for o in self.options)
            options_part = f" options=[{rendered}]"
        return f"[{self.index}] <{self.tag}>{role_part} {self.name!r}{value_part}{href_part}{options_part}"


class HeadLink(BaseModel):
    rel: str
    type: str | None = None
    href: str | None = None
    title: str | None = None

    def to_prompt_line(self) -> str:
        type_part = f" type={self.type!r}" if self.type else ""
        title_part = f" title={self.title!r}" if self.title else ""
        return f"- rel={self.rel!r}{type_part} href={self.href!r}{title_part}"


class PageSnapshot(BaseModel):
    title: str
    url: str
    elements: list[ElementInfo]
    head_links: list[HeadLink] = []
    text_summary: str

    def to_prompt(self) -> str:
        lines = [f"Page title: {self.title}", f"URL: {self.url}", "", "Interactive elements:"]
        if self.elements:
            for el in self.elements:
                lines.append(el.to_prompt_line())
        else:
            lines.append("(none found)")
        if self.head_links:
            lines.append("")
            lines.append("Discovery links (from <head>, not directly clickable - e.g. RSS/Atom feeds, canonical URL):")
            for link in self.head_links:
                lines.append(link.to_prompt_line())
        lines.append("")
        lines.append("Page text summary:")
        lines.append(self.text_summary or "(empty)")
        return "\n".join(lines)
