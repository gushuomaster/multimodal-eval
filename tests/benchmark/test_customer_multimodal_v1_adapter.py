import json
from pathlib import Path


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
