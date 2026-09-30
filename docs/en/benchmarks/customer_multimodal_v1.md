# Customer Multimodal v1


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


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `customer_multimodal_v1` |
| **Dataset ID** | `customer_multimodal_v1` |
| **Paper** | N/A |
| **Tags** | `Custom`, `MultiModal`, `QA` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets customer_multimodal_v1 \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['customer_multimodal_v1'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
