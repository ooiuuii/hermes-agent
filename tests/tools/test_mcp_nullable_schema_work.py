"""MCP nullable schema normalization preserves values without repeated subtree work."""

import copy
import cProfile

import pytest

from tools.mcp_tool_schema import _normalize_mcp_input_schema


def _optional_object_chain(depth, union_key):
    schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
    expected = copy.deepcopy(schema)
    for _ in range(depth):
        schema = {
            "type": "object", "required": [],
            "properties": {"child": {union_key: [schema, {"type": "null"}], "default": None}},
        }
        expected = {
            "type": "object", "required": [],
            "properties": {"child": {**expected, "nullable": True, "default": None}},
        }
    return schema, expected


@pytest.mark.parametrize("union_key", ["anyOf", "oneOf"])
def test_normalization_work_scales_with_optional_schema_size(union_key):
    # Profile the real entry point, without replacing its traversal or timing a busy host.
    # Warm its lazy imports outside the measurement. Four times the schema size should
    # take roughly four times the work; the fifth multiple leaves room for fixed overhead.
    _normalize_mcp_input_schema({"type": "object"})
    calls = []
    for depth in (16, 64):
        schema, expected = _optional_object_chain(depth, union_key)
        before = copy.deepcopy(schema)
        profiler = cProfile.Profile()
        result = profiler.runcall(_normalize_mcp_input_schema, schema)
        calls.append(sum(entry.callcount for entry in profiler.getstats()))
        assert result == expected
        assert schema == before
    assert calls[1] <= 5 * calls[0], f"4x schema depth took {calls[1] / calls[0]:.2f}x calls: {calls}"


def test_nested_nullable_survivors_keep_metadata_and_independent_output():
    schema = {
        "type": "object",
        "definitions": {"Id": {"type": "string"}},
        "properties": {
            "payload": {
                "oneOf": [{
                    "anyOf": [{
                        "type": "object", "description": "Branch description", "required": ["id"],
                        "properties": {
                            "id": {"anyOf": [{"$ref": "#/definitions/Id"}, {"type": "null"}],
                                   "default": None},
                            "choice": {"oneOf": [{"type": "string"}, {"type": "integer"},
                                                  {"type": "null"}]},
                        },
                    }, {"type": "null"}],
                    "title": "Inner payload",
                }, {"type": "null"}],
                "description": "Outer description", "default": None, "examples": [{"id": "a"}],
            },
        },
    }
    before = copy.deepcopy(schema)
    result = _normalize_mcp_input_schema(schema)
    assert result == {
        "type": "object", "required": [], "$defs": {"Id": {"type": "string"}},
        "properties": {"payload": {
            "type": "object", "description": "Branch description", "required": ["id"],
            "properties": {
                "id": {"$ref": "#/$defs/Id", "nullable": True},
                "choice": {"oneOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            },
            "nullable": True, "title": "Inner payload", "default": None, "examples": [{"id": "a"}],
        }},
    }
    assert schema == before
    result["properties"]["payload"]["examples"][0]["id"] = "changed"
    result["$defs"]["Id"]["type"] = "integer"
    assert schema == before
