# Customer Multimodal v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `customer_multimodal_v1` benchmark that evaluates local single-image and multi-image samples using deterministic field-level JSON accuracy, without modifying EvalScope Core.

**Architecture:** Reuse `VisionLanguageAdapter` for OpenAI-compatible multimodal message parsing and existing native evaluation/output machinery. The new adapter converts each JSONL record into a `Sample` whose target is a serialized `expected` object, parses the model's JSON response, and emits per-field plus overall accuracy through the standard `Score`/`SampleScore` path. Local fixtures and focused tests prove parsing, scoring, registry discovery, and report persistence.

**Tech Stack:** Python 3.10+, Pydantic `Sample`, EvalScope `VisionLanguageAdapter`, `BenchmarkMeta`, registry decorators, JSONL fixtures, pytest/unittest-compatible tests, native `mock_llm` smoke evaluation.

**Spec:** `docs/superpowers/specs/2026-09-30-customer-multimodal-v1-design.md`

## Global Constraints

- Do not modify any file under `evalscope/api`, `evalscope/evaluator`, `evalscope/models`, `evalscope/metrics`, or `evalscope/report`.
- Do not change `pyproject.toml`, optional dependency groups, `requires-python`, or `rich` versions.
- Reuse `VisionLanguageAdapter`, `Sample`, `Score`, `BenchmarkMeta`, and `register_benchmark`; do not add a parallel runner or metric framework.
- The dataset contract is JSONL with `id`, OpenAI-compatible `messages`, and scalar-valued `expected` fields.
- Malformed or non-JSON model output records a parse error and receives zero correct fields; it is never full credit.
- Single-image and multi-image media order must remain unchanged from the input record through the model request.
- API keys must come from environment variables and never be committed or written to fixtures.
- Generated benchmark documentation and `_index.json` must be updated only through repository generation commands, never hand-edited.
- All maintained Python files use English comments/docstrings, single quotes, 120-column width, and trailing newlines.

---

### Task 1: Add customer multimodal fixtures and schema characterization tests

**Files:**
- Create: `custom_eval/multimodal/customer_v1/example.jsonl`
- Reuse: `custom_eval/multimodal/images/dog.jpg`
- Reuse: `custom_eval/multimodal/images/AMNH.jpg`
- Reuse: `custom_eval/multimodal/images/tesla.jpg`
- Test: `tests/benchmark/test_customer_multimodal_v1_adapter.py`

**Interfaces:**
- Consumes: The JSONL contract from `docs/superpowers/specs/2026-09-30-customer-multimodal-v1-design.md`.
- Produces: Literal fixture records and failing tests for record validation, one-image parsing, four-image ordering, and scalar expected fields.

- [ ] **Step 1: Write the failing fixture/contract tests**

```python
def test_customer_record_contains_messages_and_expected_fields():
    record = load_first_record()
    assert record['id'] == 'customer-v1-single-001'
    assert len(record['messages'][0]['content']) == 2
    assert record['expected'] == {'object': 'dog', 'color': 'black-and-white', 'count': 1}


def test_customer_multi_image_record_preserves_four_image_order():
    record = load_records()[1]
    image_urls = [part['image_url']['url'] for part in record['messages'][0]['content'] if part['type'] == 'image_url']
    assert image_urls == [
        'custom_eval/multimodal/images/dog.jpg',
        'custom_eval/multimodal/images/AMNH.jpg',
        'custom_eval/multimodal/images/tesla.jpg',
        'custom_eval/multimodal/images/tokyo.jpg',
    ]
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -q`

Expected: FAIL because `custom_eval/multimodal/customer_v1/example.jsonl` does not exist.

- [ ] **Step 3: Add the two literal JSONL fixtures**

The first record must contain one `image_url` part and expected fields `object`, `color`, and `count`. The second record must contain four image parts in the exact order asserted above and expected fields `has_dog` and `image_count`.

- [ ] **Step 4: Run the focused tests to verify the fixture contract passes**

Run: `pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -q`

Expected: The fixture-only tests pass; adapter import/scoring tests remain skipped or fail only because the adapter is not implemented yet.

- [ ] **Step 5: Commit the fixture contract**

```powershell
git add custom_eval/multimodal/customer_v1/example.jsonl tests/benchmark/test_customer_multimodal_v1_adapter.py
git commit -m "test: define customer multimodal v1 fixture contract"
```

### Task 2: Implement the adapter and deterministic field scorer

**Files:**
- Create: `evalscope/benchmarks/customer_multimodal_v1/__init__.py`
- Create: `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py`
- Modify: `tests/benchmark/test_customer_multimodal_v1_adapter.py`

**Interfaces:**
- Consumes: Fixture records from Task 1 and `VisionLanguageAdapter` message conversion.
- Produces: `CustomerMultimodalV1Adapter.record_to_sample(record: Dict[str, Any]) -> Sample`, `match_score(original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState) -> Score`, and registered benchmark metadata `name='customer_multimodal_v1'`.

- [ ] **Step 1: Add failing adapter tests**

```python
def test_record_to_sample_converts_image_messages_and_serializes_expected(adapter):
    sample = adapter.record_to_sample(load_records()[0])
    assert len(sample.input) == 1
    assert [part.type for part in sample.input[0].content] == ['text', 'image']
    assert json.loads(sample.target) == {'object': 'dog', 'color': 'black-and-white', 'count': 1}


def test_match_score_reports_each_field_and_overall_accuracy(adapter, task_state):
    score = adapter.match_score(
        '{"object":"dog","color":"black-and-white","count":2}',
        '{"object":"dog","color":"black-and-white","count":2}',
        json.dumps({'object': 'dog', 'color': 'black-and-white', 'count': 1}),
        task_state,
    )
    assert score.value['object_accuracy'] == 1.0
    assert score.value['color_accuracy'] == 1.0
    assert score.value['count_accuracy'] == 0.0
    assert score.value['overall_accuracy'] == 2 / 3


def test_malformed_json_receives_zero_fields_and_parse_error(adapter, task_state):
    score = adapter.match_score('not json', 'not json', json.dumps({'object': 'dog'}), task_state)
    assert score.value['overall_accuracy'] == 0.0
    assert score.metadata['parse_error'] is True
```

- [ ] **Step 2: Run the adapter tests to verify they fail**

Run: `pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -q`

Expected: FAIL because the adapter module and registered benchmark do not exist.

- [ ] **Step 3: Implement the minimal registered adapter**

Implement `CustomerMultimodalV1Adapter(VisionLanguageAdapter)` with:

```python
@register_benchmark(BenchmarkMeta(
    name='customer_multimodal_v1',
    pretty_name='Customer Multimodal v1',
    dataset_id='customer_multimodal_v1',
    category='vlm',
    tags=[Tags.CUSTOM, Tags.MULTI_MODAL, Tags.QA],
    metric_list=['overall_accuracy'],
    primary_metric=MetricSelector(name='overall_accuracy', aggregation='mean'),
    few_shot_num=0,
    eval_split='test',
    train_split=None,
    evaluation_version='v1.0',
    description='''
## Overview

Customer Multimodal v1 evaluates business-specific image question answering with deterministic structured outputs.

## Task Description

- **Task Type**: Structured multimodal question answering
- **Input**: Text prompts with one or more local images
- **Output**: JSON object containing scalar business fields
- **Domain**: Customer-defined visual inspection tasks

## Key Features

- Local JSONL fixtures with OpenAI-compatible image messages
- Single-image and multi-image samples
- Per-field exact accuracy plus overall field accuracy
- Explicit parse-error accounting for malformed model output

## Evaluation Notes

- Native EvalScope backend with standard prediction, review, and report persistence
- Deterministic field comparison; no LLM judge in v1
- Scalar strings, numbers, booleans, and null values are supported
- Nested objects and list-specific matching are out of scope for evaluation version v1
''',
))
```

`record_to_sample` must parse `messages` with `chat_messages_from_openai`, serialize `expected` with stable JSON ordering, and preserve record `id`/expected field names in `Sample.metadata`. `match_score` must normalize only scalar values, compare every expected field, set `parse_error` metadata for malformed JSON or non-object responses, and populate `overall_accuracy` without parsing field names from model prose.

- [ ] **Step 4: Run the adapter tests to verify they pass**

Run: `pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -q`

Expected: All fixture, conversion, valid-score, missing-field, and malformed-output tests pass.

- [ ] **Step 5: Commit the adapter**

```powershell
git add evalscope/benchmarks/customer_multimodal_v1 tests/benchmark/test_customer_multimodal_v1_adapter.py
git commit -m "feat: add customer multimodal v1 adapter"
```

### Task 3: Add native registry and persistence smoke coverage

**Files:**
- Create: `tests/cli/test_customer_multimodal_v1.py`
- Modify: `evalscope/benchmarks/_index.json` only through `make docs-update-index` if required by the registry contract

**Interfaces:**
- Consumes: Registered adapter and fixture path from Tasks 1-2.
- Produces: A deterministic `mock_llm` native evaluation proving local image loading, report persistence, field metrics, and registry lookup.

- [ ] **Step 1: Write the failing native smoke test**

```python
def test_customer_multimodal_v1_mock_eval_persists_field_report(tmp_path):
    result = run_task(TaskConfig(
        model='text_generation',
        eval_type='mock_llm',
        datasets=['customer_multimodal_v1'],
        dataset_args={'customer_multimodal_v1': {
            'local_path': 'custom_eval/multimodal/customer_v1',
            'subset_list': ['example'],
        }},
        limit=2,
        work_dir=str(tmp_path / 'outputs'),
        no_timestamp=True,
    ))
    report = json.loads((tmp_path / 'outputs' / 'reports' / 'DummyCustomModel' / 'customer_multimodal_v1.json').read_text())
    assert report['execution_summary']['succeeded'] == 2
    assert 'overall_accuracy' in {metric['identity']['name'] for metric in report['metrics']}
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `pytest tests/cli/test_customer_multimodal_v1.py -q`

Expected: FAIL because the registry index/adapter integration or report path is not complete.

- [ ] **Step 3: Update generated benchmark index**

Run: `make docs-update-index`

Expected: the generated index maps `customer_multimodal_v1` to the adapter module without hand edits.

- [ ] **Step 4: Implement the mock response fixture used by the test**

Patch only the existing `MockLLM.__init__` inside the test, following the repository's established test pattern, and inject one complete `ModelOutput` per fixture record:

```python
from unittest.mock import patch

from evalscope.api.model import ModelOutput
from evalscope.models.mockllm import MockLLM


outputs = [
    ModelOutput.from_content(
        model='customer-mock',
        content='{"object":"dog","color":"black-and-white","count":1}',
    ),
    ModelOutput.from_content(
        model='customer-mock',
        content='{"has_dog":true,"image_count":4}',
    ),
]
original_init = MockLLM.__init__


def patched_init(self, *args, **kwargs):
    kwargs['custom_outputs'] = outputs
    original_init(self, *args, **kwargs)


with patch.object(MockLLM, '__init__', patched_init):
    result = run_task(task_config)
```

The test must assert persisted report values and restore the patch through the context manager. It must not add a production runner or hard-code scores inside the adapter.

- [ ] **Step 5: Run the native smoke test to verify it passes**

Run: `pytest tests/cli/test_customer_multimodal_v1.py -q`

Expected: PASS with two succeeded samples and a persisted report containing field-level metrics.

- [ ] **Step 6: Commit registry/smoke coverage**

```powershell
git add tests/cli/test_customer_multimodal_v1.py evalscope/benchmarks/_index.json
git commit -m "test: cover customer multimodal v1 native evaluation"
```

### Task 4: Verify the branch and optionally run the real MiniMax-M3 check

**Files:**
- Modify: none in production; generated reports go under ignored `outputs/`

**Interfaces:**
- Consumes: completed adapter, generated index, fixtures, and tests from Tasks 1-3.
- Produces: verification evidence and an optional real API report using environment variables only.

- [ ] **Step 1: Run focused and contract checks**

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py tests/cli/test_customer_multimodal_v1.py -q
python -m py_compile evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py
git diff --check
make lint-imports
```

Expected: all focused tests pass, compilation succeeds, diff check is clean, and import direction has no new violation.

- [ ] **Step 2: Run the real API smoke only when credentials are present**

```powershell
$env:EVAL_API_URL='https://api.minimaxi.com/v1'
$env:EVAL_API_KEY='<read from the local secret source; never commit it>'
$env:EVAL_MODEL='MiniMax-M3'
evalscope eval --model $env:EVAL_MODEL --eval-type openai_api --api-url $env:EVAL_API_URL --api-key $env:EVAL_API_KEY --datasets customer_multimodal_v1 --dataset-args '{"customer_multimodal_v1":{"local_path":"custom_eval/multimodal/customer_v1","subset_list":["example"]}}' --limit 2 --work-dir outputs/customer_multimodal_v1
```

Expected: two requests complete, field metrics appear in the report, and no secret appears in tracked files or logs.

- [ ] **Step 3: Verify Core remains unchanged and summarize status**

Run: `git diff --name-only develop...HEAD -- evalscope/api evalscope/evaluator evalscope/models evalscope/metrics evalscope/report`

Expected: no output. Record any missing optional dependency or external API result as `ENVIRONMENT_ISSUE` or `MODEL_API_ISSUE`, not as a code failure.

- [ ] **Step 4: Commit only after verification**

```powershell
git status --short
git log --oneline -4
```

Expected: only intended source, tests, fixtures, generated index, and docs are present; ignored runtime outputs are not staged.
