"""Relay-rebuilt Anthropic messages retain JSON tool inputs at normalization."""
import copy
import json

from agent.relay_llm import AnthropicStreamAccumulator
from agent.tool_executor import _parse_tool_arguments
from agent.transports.anthropic import AnthropicTransport


def _accumulated_response(raw_arguments):
    accumulator = AnthropicStreamAccumulator()
    accumulator.observe({"type": "content_block_start", "index": 0,
                         "content_block": {"type": "tool_use", "id": "call-1",
                                           "name": "read_file", "input": {}}})
    for chunk in (raw_arguments[:7], raw_arguments[7:-1], raw_arguments[-1:]):
        accumulator.observe({"type": "content_block_delta", "index": 0,
                             "delta": {"type": "input_json_delta", "partial_json": chunk}})
    accumulator.observe({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})
    return accumulator.response()


def test_rebuilt_stream_preserves_valid_json_arguments_through_normalization():
    nested = {"value": "end"}
    for _ in range(24):
        nested = {"child": nested}
    expected = {"path": "report.txt", "_private_key_name": "ordinary-data",
                "nested": nested, "items": [{"brace": "}"}, None, False, 3]}
    response = _accumulated_response(json.dumps(expected))
    before = copy.deepcopy(response)
    normalized = AnthropicTransport().normalize_response(response)
    parsed, error = _parse_tool_arguments(normalized.tool_calls[0].arguments)
    assert error is None
    assert parsed == expected
    assert response == before


def test_rebuilt_stream_does_not_repair_malformed_arguments_into_an_object():
    response = _accumulated_response('{"path":"report.txt"}oops')
    normalized = AnthropicTransport().normalize_response(response)
    parsed, error = _parse_tool_arguments(normalized.tool_calls[0].arguments)
    assert parsed == {}
    assert error is not None
