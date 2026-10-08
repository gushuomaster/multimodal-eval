import json
from pathlib import Path
from typing import Any

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.metric import SampleScore
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

FIXTURE_PATH = Path(__file__).parents[2] / 'custom_eval' / 'multimodal' / 'customer_v1' / 'example.jsonl'


def load_records() -> list[dict]:
    with FIXTURE_PATH.open(encoding='utf-8') as fixture_file:
        return [json.loads(line) for line in fixture_file if line.strip()]


def load_first_record() -> dict:
    return load_records()[0]


def test_customer_record_contains_messages_and_expected_fields():
    record = load_first_record()

    assert record['id'] == 'customer-v1-single-001'
    assert len(record['messages']) == 1
    assert len(record['messages'][0]['content']) == 2
    assert record['expected'] == {'object': 'dog', 'color': 'black-and-white', 'count': 1}


def test_customer_single_image_record_uses_openai_image_part():
    record = load_first_record()
    image_parts = [part for part in record['messages'][0]['content'] if part['type'] == 'image_url']

    assert len(image_parts) == 1
    assert image_parts[0]['image_url']['url'] == 'custom_eval/multimodal/images/dog.jpg'


def test_customer_multi_image_record_preserves_four_image_order():
    record = load_records()[1]
    image_urls = [
        part['image_url']['url']
        for part in record['messages'][0]['content']
        if part['type'] == 'image_url'
    ]

    assert image_urls == [
        'custom_eval/multimodal/images/dog.jpg',
        'custom_eval/multimodal/images/AMNH.jpg',
        'custom_eval/multimodal/images/tesla.jpg',
        'custom_eval/multimodal/images/tokyo.jpg',
    ]


def test_customer_multi_image_record_contains_scalar_expected_fields():
    record = load_records()[1]

    assert record['id'] == 'customer-v1-multi-001'
    assert record['expected'] == {'has_dog': True, 'image_count': 4}
    assert all(isinstance(value, (bool, int, float, str)) for value in record['expected'].values())


def test_customer_prompts_require_json_with_expected_field_names():
    for record in load_records():
        prompt = ' '.join(
            part['text'] for part in record['messages'][0]['content'] if part['type'] == 'text'
        )

        assert 'JSON' in prompt
        assert '不要输出 Markdown' in prompt
        assert all(field_name in prompt for field_name in record['expected'])


def _adapter():
    return get_benchmark('customer_multimodal_v1', TaskConfig(model='mock', datasets=['customer_multimodal_v1']))


def _task_state() -> TaskState:
    return TaskState(model='mock', sample=Sample(input='question', target=''))


def test_customer_multimodal_evaluation_version_is_v1_3() -> None:
    assert _adapter().benchmark_meta.evaluation_version == 'v1.3'


def test_customer_record_to_sample_converts_openai_image_messages():
    adapter = _adapter()
    record = load_first_record()

    sample = adapter.record_to_sample(record)

    assert [content.type for content in sample.input[0].content] == ['text', 'image']
    assert isinstance(sample.input[0].content[0], ContentText)
    assert isinstance(sample.input[0].content[1], ContentImage)
    assert sample.target == '{"color": "black-and-white", "count": 1, "object": "dog"}'
    assert sample.metadata['id'] == 'customer-v1-single-001'
    assert sample.metadata['expected_fields'] == ['object', 'color', 'count']


def test_customer_record_to_sample_preserves_multi_image_order() -> None:
    sample = _adapter().record_to_sample(load_records()[1])

    assert [content.image for content in sample.input[0].content if isinstance(content, ContentImage)] == [
        'custom_eval/multimodal/images/dog.jpg',
        'custom_eval/multimodal/images/AMNH.jpg',
        'custom_eval/multimodal/images/tesla.jpg',
        'custom_eval/multimodal/images/tokyo.jpg',
    ]


def test_customer_record_to_sample_preserves_business_contract() -> None:
    record = load_first_record()
    record['expected'] = {'target_id': 'item-1', 'confidence': 0.91, 'description': 'expected'}
    record['schema'] = {
        'type': 'object',
        'properties': {
            'target_id': {'type': 'string'},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'description': {'type': 'string'},
        },
        'required': ['target_id', 'confidence', 'description'],
        'additionalProperties': False,
    }
    record['tolerance'] = {'confidence': {'absolute': 0.01}}
    record['critical_fields'] = ['target_id', 'confidence']

    sample = _adapter().record_to_sample(record)

    assert sample.metadata['schema'] == record['schema']
    assert sample.metadata['absolute_tolerances'] == {'confidence': 0.01}
    assert sample.metadata['critical_fields'] == ['target_id', 'confidence']


def test_customer_match_score_applies_schema_tolerance_and_critical_fields() -> None:
    record = load_first_record()
    record['expected'] = {'target_id': 'item-1', 'confidence': 0.91, 'description': 'expected'}
    record['schema'] = {
        'type': 'object',
        'properties': {
            'target_id': {'type': 'string'},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'description': {'type': 'string'},
        },
        'required': ['target_id', 'confidence', 'description'],
        'additionalProperties': False,
    }
    record['tolerance'] = {'confidence': {'absolute': 0.01}}
    record['critical_fields'] = ['target_id', 'confidence']
    sample = _adapter().record_to_sample(record)
    task_state = TaskState(model='mock', sample=sample)
    response = '{"target_id":"item-1","confidence":0.915,"description":"different"}'

    score = _adapter().match_score(response, response, sample.target, task_state)

    assert score.value == {
        'target_id_accuracy': 1.0,
        'confidence_accuracy': 1.0,
        'description_accuracy': 0.0,
        'field_accuracy': 2 / 3,
        'overall_accuracy': 2 / 3,
        'accuracy': 2 / 3,
        'overall_command_correct': 1.0,
        'schema_valid': 1.0,
    }
    assert score.metadata['value_mismatches']['description'] == {
        'expected': 'expected',
        'actual': 'different',
    }


@pytest.mark.parametrize('field', ['id', 'expected'])
def test_customer_record_rejects_missing_required_fields(field: str) -> None:
    record = load_first_record()
    del record[field]

    with pytest.raises(ValueError, match=field):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('record_id', ['', '  ', 123, None])
def test_customer_record_requires_non_empty_string_id(record_id: Any) -> None:
    record = load_first_record()
    record['id'] = record_id

    with pytest.raises(ValueError, match='id.*non-empty string'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('expected', [None, [], 'dog', {}])
def test_customer_record_requires_non_empty_expected_object(expected: Any) -> None:
    record = load_first_record()
    record['expected'] = expected

    with pytest.raises(ValueError, match='expected.*non-empty object'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('field', ['', '  ', 1])
def test_customer_record_rejects_invalid_expected_field_names(field: Any) -> None:
    record = load_first_record()
    record['expected'] = {field: 'dog'}

    with pytest.raises(ValueError, match='expected.*non-empty strings'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('field_name', ['field', 'Field', 'FIELD', 'field-', 'overall', 'Overall'])
def test_customer_record_rejects_fields_reserved_for_aggregate_metrics(field_name: str) -> None:
    record = load_first_record()
    record['expected'] = {field_name: 'dog', 'count': 1}

    with pytest.raises(ValueError, match=rf'{field_name}.*reserved'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('value', [[], {}, float('nan'), float('inf'), float('-inf')])
def test_customer_record_rejects_non_scalar_or_non_finite_targets(value: Any) -> None:
    record = load_first_record()
    record['expected'] = {'object': value}

    with pytest.raises(ValueError, match='expected.*object.*JSON scalar'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize(
    ('tolerance', 'message'),
    [
        ([], 'tolerance.*object'),
        ('confidence', 'tolerance.*object'),
        ({'missing': {'absolute': 0.1}}, 'tolerance.*missing.*expected field'),
        ({'confidence': {}}, 'tolerance.*confidence.*absolute'),
        (
            {'confidence': {'absolute': 0.1, 'relative': 0.1}},
            'tolerance.*confidence.*only supports absolute',
        ),
        ({'confidence': {'absolute': -0.1}}, 'tolerance.*confidence.*non-negative finite'),
        ({'confidence': {'absolute': True}}, 'tolerance.*confidence.*non-negative finite'),
    ],
)
def test_customer_record_rejects_invalid_tolerance_contract(tolerance: Any, message: str) -> None:
    record = load_first_record()
    record['expected'] = {'confidence': 0.91}
    record['tolerance'] = tolerance

    with pytest.raises(ValueError, match=message):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize(
    ('critical_fields', 'message'),
    [
        ([], 'critical_fields.*non-empty list'),
        ('object', 'critical_fields.*non-empty list'),
        ([{}], 'critical_fields.*non-empty strings'),
        ([''], 'critical_fields.*non-empty strings'),
        (['missing'], 'critical_fields.*missing.*expected field'),
        (['object', 'object'], 'critical_fields.*duplicate'),
    ],
)
def test_customer_record_rejects_invalid_critical_fields_contract(critical_fields: Any, message: str) -> None:
    record = load_first_record()
    record['critical_fields'] = critical_fields

    with pytest.raises(ValueError, match=message):
        _adapter().record_to_sample(record)


def test_customer_record_rejects_tolerance_for_non_numeric_expected_field() -> None:
    record = load_first_record()
    record['expected'] = {'label': 'dog'}
    record['tolerance'] = {'label': {'absolute': 0.1}}

    with pytest.raises(ValueError, match='tolerance.*label.*numeric expected field'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('schema', ['not-an-object', {'type': 'unsupported'}])
def test_customer_record_rejects_invalid_json_schema(schema: Any) -> None:
    record = load_first_record()
    record['schema'] = schema

    with pytest.raises(ValueError, match='schema.*valid JSON Schema object'):
        _adapter().record_to_sample(record)


def test_customer_strings_use_exact_matching_without_normalization() -> None:
    adapter = _adapter()
    record = load_first_record()
    record['expected'] = {'object': ' Dog ', 'color': None}
    sample = adapter.record_to_sample(record)
    response = '{"object":"DOG", "color":null}'

    score = adapter.match_score(response, response, sample.target, _task_state())

    assert score.value == {
        'object_accuracy': 0.0,
        'color_accuracy': 1.0,
        'field_accuracy': 0.5,
        'overall_accuracy': 0.5,
        'accuracy': 0.5,
        'overall_command_correct': 0.0,
    }


def test_customer_schema_can_pass_while_business_value_fails() -> None:
    record = load_first_record()
    record['expected'] = {'color': 'black-and-white'}
    record['schema'] = {
        'type': 'object',
        'properties': {'color': {'type': 'string'}},
        'required': ['color'],
        'additionalProperties': False,
    }
    sample = _adapter().record_to_sample(record)
    task_state = TaskState(model='mock', sample=sample)
    response = '{"color":"black and white"}'

    score = _adapter().match_score(response, response, sample.target, task_state)

    assert score.value['schema_valid'] == 1.0
    assert score.value['field_accuracy'] == 0.0
    assert score.value['overall_command_correct'] == 0.0
    assert score.metadata['value_mismatches']['color'] == {
        'expected': 'black-and-white',
        'actual': 'black and white',
    }


def test_customer_schema_failure_does_not_erase_business_field_accuracy() -> None:
    record = load_first_record()
    record['expected'] = {'color': 'black-and-white'}
    record['schema'] = {
        'type': 'object',
        'properties': {'color': {'type': 'string'}},
        'required': ['color'],
        'additionalProperties': False,
    }
    sample = _adapter().record_to_sample(record)
    task_state = TaskState(model='mock', sample=sample)
    response = '{"color":"black-and-white","extra":true}'

    score = _adapter().match_score(response, response, sample.target, task_state)

    assert score.value['schema_valid'] == 0.0
    assert score.value['field_accuracy'] == 1.0
    assert score.value['overall_command_correct'] == 0.0
    assert score.metadata['schema_errors'][0]['validator'] == 'additionalProperties'


def test_customer_match_score_reports_field_accuracy_for_wrong_count():
    adapter = _adapter()

    score = adapter.match_score(
        '{"object":"dog","color":"black-and-white","count":2}',
        '{"object":"dog","color":"black-and-white","count":2}',
        '{"object":"dog","color":"black-and-white","count":1}',
        _task_state(),
    )

    assert score.value['object_accuracy'] == 1.0
    assert score.value['color_accuracy'] == 1.0
    assert score.value['count_accuracy'] == 0.0
    assert score.value['overall_accuracy'] == 2 / 3


def test_customer_match_score_marks_malformed_json_as_parse_error():
    adapter = _adapter()

    score = adapter.match_score('not json', 'not json', '{"object":"dog"}', _task_state())

    assert score.value['overall_accuracy'] == 0.0
    assert score.metadata['parse_error'] is True


def test_customer_match_score_scores_missing_expected_fields_as_zero():
    adapter = _adapter()

    score = adapter.match_score('{"object":"dog"}', '{"object":"dog"}', '{"object":"dog","color":"black","count":1}', _task_state())

    assert score.value['object_accuracy'] == 1.0
    assert score.value['color_accuracy'] == 0.0
    assert score.value['count_accuracy'] == 0.0
    assert score.value['overall_accuracy'] == 1 / 3


def test_customer_match_score_does_not_match_boolean_to_number():
    adapter = _adapter()

    score = adapter.match_score('{"count":true}', '{"count":true}', '{"count":1}', _task_state())

    assert score.value['count_accuracy'] == 0.0
    assert score.value['overall_accuracy'] == 0.0


def test_customer_match_score_marks_non_object_json_as_parse_error():
    adapter = _adapter()

    score = adapter.match_score('[]', '[]', '{"object":"dog","count":1}', _task_state())

    assert score.metadata['parse_error'] is True
    assert score.value['object_accuracy'] == 0.0
    assert score.value['count_accuracy'] == 0.0
    assert score.value['overall_accuracy'] == 0.0


@pytest.mark.parametrize('response', ['not json', '[]', '{"object":"dog","extra":NaN}',
                                      '{"object":"dog","extra":Infinity}',
                                      '{"object":"dog","extra":-Infinity}'])
def test_customer_invalid_json_zeroes_all_expected_metrics(response: str) -> None:
    score = _adapter().match_score(response, response, '{"object":"dog","count":1}', _task_state())

    assert score.metadata['parse_error'] is True
    assert score.value == {
        'object_accuracy': 0.0,
        'count_accuracy': 0.0,
        'field_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'accuracy': 0.0,
        'overall_command_correct': 0.0,
    }


def test_customer_invalid_json_zeroes_schema_and_business_metrics() -> None:
    record = load_first_record()
    record['expected'] = {'object': 'dog'}
    record['schema'] = {
        'type': 'object',
        'properties': {'object': {'type': 'string'}},
        'required': ['object'],
    }
    sample = _adapter().record_to_sample(record)
    task_state = TaskState(model='mock', sample=sample)

    score = _adapter().match_score('not json', 'not json', sample.target, task_state)

    assert score.metadata['parse_error'] is True
    assert score.value['schema_valid'] == 0.0
    assert score.value['field_accuracy'] == 0.0
    assert score.value['overall_command_correct'] == 0.0


def test_customer_aggregation_counts_parse_errors_without_excluding_zero_scores() -> None:
    adapter = _adapter()
    responses = ['{"object":"dog"}', 'not json']
    sample_scores = [
        SampleScore(
            score=adapter.match_score(response, response, '{"object":"dog"}', _task_state()),
            sample_id=index,
        )
        for index, response in enumerate(responses)
    ]

    metrics = {score.metric_name: score for score in adapter.aggregate_scores(sample_scores)}

    for name in ['accuracy', 'overall_accuracy']:
        assert metrics[name].score == 0.5
        assert metrics[name].num == 2
        assert metrics[name].metadata == {'parse_error_count': 1, 'sample_count': 2}


@pytest.mark.parametrize(
    ('config_name', 'config_value'),
    [
        ('field_weights', {'confidence': 2}),
        ('relative_tolerance', {'confidence': 0.01}),
        ('fuzzy_matching', True),
        ('normalization', {'trim': True}),
    ],
)
def test_customer_record_rejects_unsupported_scoring_configuration(
    config_name: str,
    config_value: Any,
) -> None:
    record = load_first_record()
    record['expected'] = {'confidence': 0.91}
    record[config_name] = config_value

    with pytest.raises(ValueError, match=rf'{config_name}.*not supported'):
        _adapter().record_to_sample(record)


def test_customer_record_allows_unrelated_dataset_metadata() -> None:
    record = load_first_record()
    record['business_case'] = 'retail-image-check'

    sample = _adapter().record_to_sample(record)

    assert sample.metadata['id'] == record['id']
