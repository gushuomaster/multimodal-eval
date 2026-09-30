import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.models.utils.openai import chat_messages_from_openai
from evalscope.report import Report


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Invalid JSON constant: {value}')


@register_benchmark(
    BenchmarkMeta(
        name='customer_multimodal_v1',
        pretty_name='Customer Multimodal v1',
        dataset_id='customer_multimodal_v1',
        tags=[Tags.CUSTOM, Tags.MULTI_MODAL, Tags.QA],
        metric_list=['accuracy'],
        primary_metric=MetricSelector(name='accuracy', aggregation='mean'),
        few_shot_num=0,
        eval_split='test',
        train_split=None,
        evaluation_version='v1.1',
        description="""
## Overview

Customer Multimodal v1 evaluates structured visual question answering over customer-provided image messages.

## Task Description

- **Task Type**: Structured visual question answering
- **Input**: OpenAI-compatible text and image messages
- **Output**: A JSON object containing the expected scalar fields
- **Domain**: Customer image understanding

## Key Features

- Supports local multimodal fixture records with stable expected-field targets
- Scores every expected scalar field independently and reports an overall accuracy
- Preserves record identifiers and expected field names for review and debugging

## Evaluation Notes

- The primary metric is the mean `accuracy` across samples; `overall_accuracy` is persisted as a diagnostic aggregate alias
- Responses must be JSON objects; malformed or non-object responses receive zero accuracy
- Non-standard JSON constants (`NaN`, `Infinity`, and `-Infinity`) are parse errors and receive zero accuracy
- Aggregate metadata includes `parse_error_count` and `sample_count` for both accuracy metrics. These records are saved
  per subset in `customer_multimodal_v1_diagnostics.jsonl` beside the standard report; the standard report schema does
  not retain aggregate metadata
- Evaluation uses the `test` split and requires no few-shot examples or network access

## Local Dataset Configuration

Run from the repository root so relative image paths resolve correctly. The local loader reads `example.jsonl`
when `local_path='custom_eval/multimodal/customer_v1'` and `subset_list=['example']`.

Each JSONL record requires a non-empty string `id`, OpenAI-compatible `messages`, and a non-empty `expected` object.
Expected field names must be non-empty strings; `overall` is reserved for the aggregate metric.
Expected values must be JSON scalars (string, finite number, boolean, or null); nested objects and arrays are invalid.
Invalid dataset records raise an error during conversion instead of being scored. String comparisons ignore surrounding
whitespace and case; numeric comparisons do not equate booleans with numbers.

This offline smoke run checks local loading and scoring with the mock model. Its default response is not JSON, so zero
accuracy and parse errors are expected. For model evaluation, use `eval_type='openai_api'` and configure `model`, `api_url`,
and `api_key` for your endpoint while retaining the same `dataset_args`.

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='text_generation',
    eval_type='mock_llm',
    datasets=['customer_multimodal_v1'],
    dataset_args={
        'customer_multimodal_v1': {
            'local_path': 'custom_eval/multimodal/customer_v1',
            'subset_list': ['example'],
        },
    },
    limit=2,
))
```

The equivalent CLI command (Bash or PowerShell) is:

```bash
evalscope eval --model text_generation --eval-type mock_llm --datasets customer_multimodal_v1 --dataset-args '{"customer_multimodal_v1":{"local_path":"custom_eval/multimodal/customer_v1","subset_list":["example"]}}' --limit 2
```
""",
    )
)
class CustomerMultimodalV1Adapter(VisionLanguageAdapter):
    """Adapter for the deterministic customer multimodal fixture benchmark."""

    def load_from_disk(self, **kwargs) -> Any:
        """Load local JSONL fixture files through EvalScope's local data loader."""
        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert an OpenAI message record into an EvalScope sample."""
        record_id = record.get('id')
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError('id must be a non-empty string')
        expected = record.get('expected')
        if not isinstance(expected, dict) or not expected:
            raise ValueError('expected must be a non-empty object')
        for name, value in expected.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError('expected field names must be non-empty strings')
            if name == 'overall':
                raise ValueError("expected field 'overall' is reserved for the aggregate metric")
            if (value is not None and not isinstance(value, (str, int, float, bool))) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f'expected field {name!r} must contain a JSON scalar with finite numbers')
        messages = record['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        target = json.dumps(expected, ensure_ascii=False, sort_keys=True)
        return Sample(
            input=chat_messages_from_openai(model='', messages=messages),
            target=target,
            metadata={'id': record_id, 'expected_fields': list(expected.keys())},
        )

    @staticmethod
    def _parse_object(value: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(value, parse_constant=_reject_json_constant)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalize_scalar(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @classmethod
    def _scalars_equal(cls, predicted: Any, expected: Any) -> bool:
        """Compare scalar values without conflating booleans and numbers."""
        predicted = cls._normalize_scalar(predicted)
        expected = cls._normalize_scalar(expected)
        if isinstance(predicted, bool) or isinstance(expected, bool):
            return type(predicted) is type(expected) and predicted == expected
        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            return predicted == expected
        return predicted == expected

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Compare each expected scalar field in a model JSON response."""
        expected = self._parse_object(reference)
        predicted = self._parse_object(filtered_prediction)
        score = Score(prediction=original_prediction, extracted_prediction=filtered_prediction)
        field_names = list(expected.keys()) if expected is not None else []
        if expected is None or predicted is None:
            score.metadata['parse_error'] = True
            score.value.update({f'{name}_accuracy': 0.0 for name in field_names})
            score.value['overall_accuracy'] = 0.0
            score.value['accuracy'] = 0.0
            score.main_score_name = 'overall_accuracy'
            return score

        accuracies = {
            f'{name}_accuracy': float(name in predicted and self._scalars_equal(predicted[name], value))
            for name, value in expected.items()
        }
        score.value.update(accuracies)
        score.value['overall_accuracy'] = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
        score.value['accuracy'] = score.value['overall_accuracy']
        score.main_score_name = 'overall_accuracy'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Include response parsing coverage in the overall accuracy aggregates."""
        aggregates = super().aggregate_scores(sample_scores)
        parse_error_count = sum(sample.score.metadata.get('parse_error') is True for sample in sample_scores)
        for aggregate in aggregates:
            if aggregate.metric_name in ('accuracy', 'overall_accuracy'):
                aggregate.metadata = {
                    **(aggregate.metadata or {}),
                    'parse_error_count': parse_error_count,
                    'sample_count': len(sample_scores),
                }
        return aggregates

    def generate_report(
        self, scores: Dict[str, List[AggScore]], model_name: str, output_dir: str, **kwargs: Any
    ) -> Report:
        """Persist aggregate parsing diagnostics alongside the standard report."""
        report = super().generate_report(scores, model_name, output_dir, **kwargs)
        diagnostics_path = Path(output_dir) / 'customer_multimodal_v1_diagnostics.jsonl'
        with diagnostics_path.open('w', encoding='utf-8', newline='\n') as diagnostics_file:
            for subset, aggregates in scores.items():
                record = {
                    'subset': subset,
                    'aggregates': [
                        aggregate.model_dump(mode='json')
                        for aggregate in aggregates
                        if aggregate.metric_name in ('accuracy', 'overall_accuracy')
                    ],
                }
                diagnostics_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        return report
