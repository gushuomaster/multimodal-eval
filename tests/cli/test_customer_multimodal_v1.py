import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evalscope.api.model import ModelOutput
from evalscope.config import TaskConfig
from evalscope.models.mockllm import MockLLM
from evalscope.run import run_task


def _scripted_init(model_self, *args, **kwargs):
    """Inject one complete JSON response for each customer fixture record."""
    original_init = _scripted_init.original_init
    kwargs['custom_outputs'] = [
        ModelOutput.from_content(
            model='text_generation',
            content='{"object":"dog","color":"black-and-white","count":1}',
        ),
        ModelOutput.from_content(
            model='text_generation',
            content='{"has_dog":true,"image_count":4}',
        ),
    ]
    original_init(model_self, *args, **kwargs)


def test_customer_multimodal_v1_native_run_persists_exact_scores(tmp_path: Path) -> None:
    """The native runner evaluates both fixture records and persists aggregate evidence."""
    _scripted_init.original_init = MockLLM.__init__
    with patch.object(MockLLM, '__init__', _scripted_init):
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
    assert metrics['accuracy']['score'] == pytest.approx(1.0)
    assert metrics['accuracy']['num'] == 2
    assert metrics['overall_accuracy']['score'] == pytest.approx(1.0)
    assert metrics['overall_accuracy']['num'] == 2
    expected_num = {
        'object_accuracy': 1,
        'color_accuracy': 1,
        'count_accuracy': 1,
        'has_dog_accuracy': 1,
        'image_count_accuracy': 1,
    }
    for name, expected in expected_num.items():
        assert metrics[name]['score'] == pytest.approx(1.0)
        assert metrics[name]['num'] == expected
