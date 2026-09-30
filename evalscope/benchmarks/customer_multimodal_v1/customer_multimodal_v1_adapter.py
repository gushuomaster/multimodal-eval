import json
from typing import Any, Dict, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.models.utils.openai import chat_messages_from_openai


@register_benchmark(
    BenchmarkMeta(
        name='customer_multimodal_v1',
        pretty_name='Customer Multimodal v1',
        dataset_id='customer_multimodal_v1',
        tags=[Tags.CUSTOM, Tags.MULTI_MODAL, Tags.QA],
        metric_list=['overall_accuracy'],
        primary_metric=MetricSelector(name='overall_accuracy', aggregation='mean'),
        few_shot_num=0,
        eval_split='test',
        train_split=None,
        evaluation_version='v1.0',
        description='''
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

- The primary metric is the mean `overall_accuracy` across samples
- Responses must be JSON objects; malformed or non-object responses receive zero accuracy
- Evaluation uses the `test` split and requires no few-shot examples or network access
''',
    )
)
class CustomerMultimodalV1Adapter(VisionLanguageAdapter):
    """Adapter for the deterministic customer multimodal fixture benchmark."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert an OpenAI message record into an EvalScope sample."""
        messages = record['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        expected = record.get('expected', {})
        target = json.dumps(expected, ensure_ascii=False, sort_keys=True)
        return Sample(
            input=chat_messages_from_openai(model='', messages=messages),
            target=target,
            metadata={'id': record.get('id'), 'expected_fields': list(expected.keys())},
        )

    @staticmethod
    def _parse_object(value: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalize_scalar(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

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
            score.main_score_name = 'overall_accuracy'
            return score

        accuracies = {
            f'{name}_accuracy': float(name in predicted and self._normalize_scalar(predicted[name]) == self._normalize_scalar(value))
            for name, value in expected.items()
        }
        score.value.update(accuracies)
        score.value['overall_accuracy'] = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0
        score.main_score_name = 'overall_accuracy'
        return score
