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


def test_customer_record_rejects_reserved_overall_field() -> None:
    record = load_first_record()
    record['expected'] = {'overall': 'dog', 'count': 1}

    with pytest.raises(ValueError, match='overall.*reserved'):
        _adapter().record_to_sample(record)


@pytest.mark.parametrize('value', [[], {}, float('nan'), float('inf'), float('-inf')])
def test_customer_record_rejects_non_scalar_or_non_finite_targets(value: Any) -> None:
    record = load_first_record()
    record['expected'] = {'object': value}

    with pytest.raises(ValueError, match='expected.*object.*JSON scalar'):
        _adapter().record_to_sample(record)


def test_customer_null_and_string_targets_support_scalar_normalization() -> None:
    adapter = _adapter()
    record = load_first_record()
    record['expected'] = {'object': ' Dog ', 'color': None}
    sample = adapter.record_to_sample(record)
    response = '{"object":"DOG", "color":null}'

    score = adapter.match_score(response, response, sample.target, _task_state())

    assert score.value == {'object_accuracy': 1.0, 'color_accuracy': 1.0, 'overall_accuracy': 1.0, 'accuracy': 1.0}


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
    assert score.value == {'object_accuracy': 0.0, 'count_accuracy': 0.0, 'overall_accuracy': 0.0, 'accuracy': 0.0}


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
