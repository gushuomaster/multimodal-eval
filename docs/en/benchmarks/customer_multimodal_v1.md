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
- Evaluation uses the `test` split and requires no few-shot examples or network access


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
