import pytest
from pydantic import TypeAdapter, ValidationError

from webagent.output_spec import json_schema_to_model


def test_flat_object_required_and_optional_fields():
    schema = {
        "type": "object",
        "properties": {
            "pricing_url": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["pricing_url"],
    }
    model = json_schema_to_model(schema)

    instance = model(pricing_url="https://example.com/pricing")
    assert instance.pricing_url == "https://example.com/pricing"
    assert instance.confidence is None

    with pytest.raises(ValidationError):
        model()  # missing required field


def test_nested_object_property():
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        },
        "required": ["address"],
    }
    model = json_schema_to_model(schema)

    instance = model(address={"city": "Boston"})
    assert instance.address.city == "Boston"


def test_array_of_scalars():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
    }
    model = json_schema_to_model(schema)

    instance = model(tags=["a", "b"])
    assert instance.tags == ["a", "b"]


def test_array_of_objects():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        },
        "required": ["items"],
    }
    model = json_schema_to_model(schema)

    instance = model(items=[{"name": "widget"}])
    assert instance.items[0].name == "widget"


def test_enum_becomes_literal():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["pending", "done"]}},
        "required": ["status"],
    }
    model = json_schema_to_model(schema)

    instance = model(status="done")
    assert instance.status == "done"

    with pytest.raises(ValidationError):
        model(status="unknown")


def test_top_level_array_schema_resolves_to_list_of_model():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
            },
            "required": ["title", "url"],
        },
    }
    answer_type = json_schema_to_model(schema)
    adapter = TypeAdapter(answer_type)

    value = adapter.validate_python([{"title": "Job A", "url": "https://example.com/1"}])
    assert value[0].title == "Job A"
    assert adapter.dump_python(value, mode="json") == [{"title": "Job A", "url": "https://example.com/1"}]


def test_top_level_array_of_scalars_resolves_to_list_of_scalar():
    schema = {"type": "array", "items": {"type": "string"}}
    answer_type = json_schema_to_model(schema)
    adapter = TypeAdapter(answer_type)

    value = adapter.validate_python(["a", "b"])
    assert value == ["a", "b"]


def test_unsupported_keyword_raises_not_implemented():
    schema = {
        "type": "object",
        "properties": {"value": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["value"],
    }
    with pytest.raises(NotImplementedError):
        json_schema_to_model(schema)
