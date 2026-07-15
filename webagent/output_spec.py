from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, create_model
from pydantic_ai import Agent

_UNSUPPORTED_KEYWORDS = ("$ref", "oneOf", "anyOf", "allOf")

_SCALAR_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


@dataclass(frozen=True)
class OutputSpec:
    """Caller-supplied contract for a run_task() answer - exactly one of the two."""

    json_schema: dict[str, Any] | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if bool(self.json_schema) == bool(self.description):
            raise ValueError("OutputSpec requires exactly one of json_schema or description")


def _check_unsupported(schema: dict[str, Any], name: str) -> None:
    for keyword in _UNSUPPORTED_KEYWORDS:
        if keyword in schema:
            raise NotImplementedError(
                f"json_schema_to_model does not support the {keyword!r} keyword (at {name!r})"
            )


def _resolve_type(prop_schema: dict[str, Any], name: str) -> Any:
    _check_unsupported(prop_schema, name)

    if "enum" in prop_schema:
        return Literal[tuple(prop_schema["enum"])]

    prop_type = prop_schema.get("type")
    if prop_type in _SCALAR_TYPES:
        return _SCALAR_TYPES[prop_type]
    if prop_type == "object":
        if "properties" not in prop_schema:
            return dict[str, Any]
        return json_schema_to_model(prop_schema, name)
    if prop_type == "array":
        items_schema = prop_schema.get("items")
        if items_schema is None:
            return list[Any]
        return list[_resolve_type(items_schema, f"{name}_item")]

    raise NotImplementedError(f"json_schema_to_model does not support type={prop_type!r} (at {name!r})")


def json_schema_to_model(schema: dict[str, Any], name: str = "Answer") -> Any:
    """Convert a supported subset of JSON Schema into a Pydantic-compatible type.

    Object schemas become a dynamic BaseModel (recursing into nested objects). A
    top-level array or scalar schema resolves directly instead (e.g. a top-level
    `array` of objects becomes `list[SomeModel]`) - callers only need a valid field
    type annotation for FinishAction.answer, not necessarily a BaseModel itself.

    Supports: object/properties/required, nested objects, arrays, enums, and the
    string/integer/number/boolean scalar types. Does not support $ref/oneOf/anyOf/allOf.
    """
    _check_unsupported(schema, name)

    schema_type = schema.get("type", "object")
    if schema_type != "object":
        return _resolve_type(schema, name)

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        field_type = _resolve_type(prop_schema, f"{name}_{prop_name}")
        if prop_name in required:
            fields[prop_name] = (field_type, ...)
        else:
            fields[prop_name] = (field_type | None, None)

    return create_model(name, **fields)


JSONScalar = Union[str, int, float, bool]


class GenericAnswer(BaseModel):
    """Fallback answer contract for the natural-language-description mode.

    `result` is a concrete JSON-value union rather than bare `Any` - some providers'
    strict structured-output mode (e.g. Anthropic) rejects schemas whose fields have
    no `type`/`anyOf`, which is what `Any` produces. Object-shaped values are
    deliberately not supported here: Anthropic's strict mode silently coerces open
    `dict` fields to `{}` instead of erroring, which would look like a successful but
    empty answer. Callers who need a structured object answer should use
    `output_schema` instead, which validates properly.
    """

    result: JSONScalar | list[JSONScalar]


def generic_answer_model() -> type[BaseModel]:
    return GenericAnswer


class SelfCheckVerdict(BaseModel):
    passes: bool
    reason: str


async def self_check(task: str, description: str, result: Any, model: str) -> SelfCheckVerdict:
    """Lightweight LLM-judge check: does `result` satisfy the caller's NL description?"""
    checker: Agent[None, SelfCheckVerdict] = Agent(
        model,
        output_type=SelfCheckVerdict,
        system_prompt=(
            "You are grading whether a web-browsing agent's final answer satisfies its "
            "task and the caller's expected-output description. Set passes=true only if "
            "the produced result plausibly and fully satisfies that description; "
            "otherwise passes=false with a concise `reason` the agent can act on to "
            "try again."
        ),
    )
    prompt = (
        f"Task: {task}\n"
        f"Expected output description: {description}\n"
        f"Produced result: {json.dumps(result, default=str)}\n\n"
        "Does the produced result satisfy the expected output description?"
    )
    verdict = await checker.run(prompt)
    return verdict.output
