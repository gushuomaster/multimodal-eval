import math

import pytest

from evalscope.api.benchmark.adapters.function_call_adapter import FunctionCallAdapter
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model.generate_config import ResponseSchema
from evalscope.api.tool import ToolCall, ToolFunction, ToolInfo, parse_tool_call, validate_tool_arguments
from evalscope.metrics.aggregators.aggregators import Mean


def _raw_tool(schema: dict) -> list[dict]:
    return [
        {
            'type': 'function',
            'function': {
                'name': 'inspect',
                'description': 'Inspect structured values.',
                'parameters': schema,
            },
        }
    ]


def _call(arguments: dict) -> ToolCall:
    return ToolCall(id='call-1', function=ToolFunction(name='inspect', arguments=arguments))


@pytest.mark.parametrize(
    ('schema', 'valid_arguments', 'invalid_arguments'),
    [
        (
            {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
            {'name': 'ok'},
            {},
        ),
        (
            {'type': 'object', 'properties': {'count': {'type': 'integer'}}},
            {'count': 2},
            {'count': '2'},
        ),
        (
            {'type': 'object', 'properties': {'mode': {'enum': ['fast', 'safe']}}},
            {'mode': 'safe'},
            {'mode': 'other'},
        ),
        (
            {'type': 'object', 'properties': {'ratio': {'type': 'number', 'minimum': 0, 'maximum': 1}}},
            {'ratio': 0.5},
            {'ratio': 1.5},
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'nested': {
                        'type': 'object',
                        'properties': {'enabled': {'type': 'boolean'}},
                        'required': ['enabled'],
                    }
                },
            },
            {'nested': {'enabled': True}},
            {'nested': {}},
        ),
        (
            {'type': 'object', 'properties': {'items': {'type': 'array', 'items': {'type': 'integer'}}}},
            {'items': [1, 2]},
            {'items': [1, '2']},
        ),
        (
            {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'additionalProperties': False},
            {'name': 'ok'},
            {'name': 'ok', 'extra': True},
        ),
    ],
)
def test_general_fc_validator_honors_raw_json_schema_keywords(
    schema: dict, valid_arguments: dict, invalid_arguments: dict
) -> None:
    assert FunctionCallAdapter.validate_tool_call([_call(valid_arguments)], _raw_tool(schema)) == (True, '')

    passed, reason = FunctionCallAdapter.validate_tool_call([_call(invalid_arguments)], _raw_tool(schema))

    assert passed is False
    assert reason.startswith("Schema validation failed for tool 'inspect':")


def test_function_call_parsing_accepts_nonstandard_json_constants() -> None:
    call = ToolCall(id='call-1', function={'name': 'inspect', 'arguments': '{"ratio": NaN}'})

    assert math.isnan(call.function.arguments['ratio'])
    assert FunctionCallAdapter.validate_tool_call(
        [call],
        _raw_tool({'type': 'object', 'properties': {'ratio': {'type': 'number'}}, 'required': ['ratio']}),
    ) == (True, '')


def test_general_fc_validator_does_not_consume_tool_parse_error() -> None:
    tool = ToolInfo.model_validate(
        {
            'name': 'inspect',
            'description': 'Inspect structured values.',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
        }
    )
    call = parse_tool_call('call-1', 'inspect', '{malformed', [tool])

    assert call.parse_error is not None
    assert call.function.arguments == {}
    assert FunctionCallAdapter.validate_tool_call(
        [call],
        _raw_tool({'type': 'object', 'properties': {}, 'additionalProperties': False}),
    ) == (True, '')


def test_typed_response_and_tool_schemas_drop_unsupported_range_keywords() -> None:
    response_schema = ResponseSchema.model_validate(
        {
            'name': 'structured_result',
            'json_schema': {
                'type': 'object',
                'properties': {'ratio': {'type': 'number', 'minimum': 0, 'maximum': 1}},
            },
        }
    )
    tool = ToolInfo.model_validate(
        {
            'name': 'inspect',
            'description': 'Inspect structured values.',
            'parameters': {
                'type': 'object',
                'properties': {'ratio': {'type': 'number', 'minimum': 0, 'maximum': 1}},
            },
        }
    )

    response_ratio = response_schema.json_schema.model_dump(exclude_none=True)['properties']['ratio']
    tool_ratio = tool.parameters.model_dump(exclude_none=True)['properties']['ratio']

    assert response_ratio == {'type': 'number'}
    assert tool_ratio == {'type': 'number'}
    assert validate_tool_arguments(_call({'ratio': 2}), tool) is None


def test_mean_aggregates_each_dynamic_score_key_over_its_present_samples() -> None:
    scores = [
        SampleScore(sample_id='a', score=Score(value={'json_valid': 1.0, 'field_accuracy': 0.5})),
        SampleScore(sample_id='b', score=Score(value={'json_valid': 0.0})),
    ]

    aggregates = {aggregate.metric_name: aggregate for aggregate in Mean()(scores)}

    assert aggregates['json_valid'].score == 0.5
    assert aggregates['json_valid'].num == 2
    assert aggregates['field_accuracy'].score == 0.5
    assert aggregates['field_accuracy'].num == 1
