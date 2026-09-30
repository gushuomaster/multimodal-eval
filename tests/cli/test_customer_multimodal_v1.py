import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evalscope.api.model import ModelOutput
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task


@pytest.mark.parametrize(('second_response', 'overall', 'parse_errors'), [
    ('{"has_dog":true,"image_count":4}', 1.0, 0),
    ('{"has_dog":true,"image_count":NaN}', 0.5, 1),
])
def test_customer_multimodal_v1_native_run_persists_exact_scores(
    tmp_path: Path, second_response: str, overall: float, parse_errors: int
) -> None:
    """The native runner evaluates both fixture records and persists aggregate evidence."""
    original_init = MockLLM.__init__

    def scripted_init(model_self: MockLLM, *args: Any, **kwargs: Any) -> None:
        kwargs['custom_outputs'] = [
            ModelOutput.from_content(
                model='text_generation',
                content='{"object":"dog","color":"black-and-white","count":1}',
            ),
            ModelOutput.from_content(model='text_generation', content=second_response),
        ]
        original_init(model_self, *args, **kwargs)

    with patch.object(MockLLM, '__init__', scripted_init):
        reports = run_task(
            TaskConfig(
                model='text_generation',
                eval_type='mock_llm',
                datasets=['customer_multimodal_v1'],
                dataset_args={
                    'customer_multimodal_v1': {
                        'local_path': 'custom_eval/multimodal/customer_v1',
                        'subset_list': ['example'],
                    }
                },
                limit=2,
                eval_batch_size=1,
                work_dir=str(tmp_path),
                no_timestamp=True,
            )
        )

    report = reports['customer_multimodal_v1']
    assert report.execution_summary is not None
    assert report.execution_summary.requested == 2
    assert report.execution_summary.succeeded == 2
    assert report.execution_summary.errored == 0

    report_path = tmp_path / 'reports' / 'text_generation' / 'customer_multimodal_v1.json'
    assert report_path.is_file()
    persisted = json.loads(report_path.read_text(encoding='utf-8'))
    assert persisted['execution_summary']['succeeded'] == 2

    metrics = {metric['identity']['name']: metric for metric in persisted['metrics']}
    assert persisted['primary_metric_identity']['name'] == 'accuracy'
    assert metrics['accuracy']['score'] == pytest.approx(overall)
    assert metrics['accuracy']['num'] == 2
    assert metrics['overall_accuracy']['score'] == pytest.approx(overall)
    assert metrics['overall_accuracy']['num'] == 2
    assert metrics['field_accuracy']['score'] == pytest.approx(overall)
    assert metrics['field_accuracy']['num'] == 2
    assert metrics['overall_command_correct']['score'] == pytest.approx(overall)
    assert metrics['overall_command_correct']['num'] == 2
    expected_num = {
        'object_accuracy': 1,
        'color_accuracy': 1,
        'count_accuracy': 1,
        'has_dog_accuracy': 1,
        'image_count_accuracy': 1,
    }
    for name, expected in expected_num.items():
        expected_score = 1.0 - parse_errors if name in ['has_dog_accuracy', 'image_count_accuracy'] else 1.0
        assert metrics[name]['score'] == pytest.approx(expected_score)
        assert metrics[name]['num'] == expected

    review_path = tmp_path / 'reviews' / 'text_generation' / 'customer_multimodal_v1_example.jsonl'
    reviews = [json.loads(line)['sample_score'] for line in review_path.read_text(encoding='utf-8').splitlines()]
    assert len(reviews) == 2
    assert sum(review['score']['metadata'].get('parse_error') is True for review in reviews) == parse_errors

    diagnostics_path = tmp_path / 'reports' / 'text_generation' / 'customer_multimodal_v1_diagnostics.jsonl'
    diagnostics = [json.loads(line) for line in diagnostics_path.read_text(encoding='utf-8').splitlines()]
    assert len(diagnostics) == 1
    assert diagnostics[0]['subset'] == 'example'
    aggregates = {metric['metric_name']: metric for metric in diagnostics[0]['aggregates']}
    assert set(aggregates) == {'accuracy', 'overall_accuracy'}
    for aggregate in aggregates.values():
        assert aggregate['score'] == pytest.approx(overall)
        assert aggregate['num'] == 2
        assert aggregate['metadata'] == {'parse_error_count': parse_errors, 'sample_count': 2}
