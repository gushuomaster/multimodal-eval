import json
from pathlib import Path

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentImage, ContentText
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
