# Structured Scoring Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three remaining `customer_multimodal_v1` scoring contract gaps so its fixed business metrics are formal quality metrics and invalid customer scoring inputs fail before inference.

**Architecture:** Keep `BusinessComparator` unchanged and close the contract at its two existing boundaries: canonical metric semantics in `catalog.py` and dataset validation in `CustomerMultimodalV1Adapter.record_to_sample()`. Tests pin both boundaries and the native report output; no new scoring algorithm or parallel registry is introduced.

**Tech Stack:** Python 3.10+, Pydantic, pytest, EvalScope metric semantics registry and benchmark adapter APIs.

**Spec:** `docs/superpowers/specs/2026-10-08-mvp-container-evaluation-design.md`

## Global Constraints

- Use the existing registry and adapter mechanisms; do not create a second metric or scoring system.
- Keep `BusinessComparator` behavior unchanged.
- Fixed business scores are bounded 0-1 quality ratios with higher values better.
- Reject invalid customer records before model inference.
- Do not reject unrelated dataset fields that are not scoring configuration.
- Bump `customer_multimodal_v1` from evaluation version `v1.2` to `v1.3` because accepted dataset semantics change.
- Keep comments and docstrings in English, line width 120, LF endings, and UTF-8 without BOM.
- Preserve the pre-existing untracked `docs/reports/` directory and all unrelated worktree changes.

---

## File Structure

- Modify `evalscope/metrics/semantics/catalog.py`: declare the four fixed business metric names.
- Modify `tests/report/semantics/test_catalog.py`: pin the public semantics of those metric names.
- Modify `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py`: reserve collision-prone fields, reject unsupported scoring configuration, and bump the evaluation version.
- Modify `tests/benchmark/test_customer_multimodal_v1_adapter.py`: cover both validation failures and the version bump.
- Modify `tests/cli/test_customer_multimodal_v1.py`: assert fixed business metrics persist with formal quality semantics.

### Task 1: Register Fixed Customer Business Metrics

**Files:**
- Modify: `evalscope/metrics/semantics/catalog.py:214-329`
- Modify: `tests/report/semantics/test_catalog.py`

**Interfaces:**
- Consumes: `METRIC_DEFINITIONS: Dict[str, MetricEntry]` and `MetricEntry.resolve(name: str) -> MetricSemantics`.
- Produces: canonical entries for `overall_accuracy`, `field_accuracy`, `overall_command_correct`, and `schema_valid`.

- [ ] **Step 1: Write the failing catalog test**

Append this test to `tests/report/semantics/test_catalog.py`:

```python
@pytest.mark.parametrize(
    'metric_name',
    ['overall_accuracy', 'field_accuracy', 'overall_command_correct', 'schema_valid'],
)
def test_customer_business_metric_is_a_bounded_quality_ratio(metric_name: str) -> None:
    semantics = METRIC_DEFINITIONS[metric_name].resolve(metric_name)

    assert semantics.semantic_id == 'quality.accuracy.ratio'
    assert semantics.kind is MetricKind.QUALITY
    assert semantics.direction is MetricDirection.HIGHER_IS_BETTER
    assert semantics.value_range is not None
    assert semantics.value_range.min == 0
    assert semantics.value_range.max == 1
    assert semantics.display_multiplier == 100
    assert semantics.display_unit == '%'
```

- [ ] **Step 2: Run the test and verify the missing declarations fail**

Run:

```powershell
pytest tests/report/semantics/test_catalog.py::test_customer_business_metric_is_a_bounded_quality_ratio -v
```

Expected: FAIL with `KeyError` for the first undeclared customer metric.

- [ ] **Step 3: Add the canonical names to the accuracy baseline**

In the `quality.accuracy.ratio` tuple in `evalscope/metrics/semantics/catalog.py`, add the names in alphabetical order:

```python
        'fact_acc',
        'field_accuracy',
        'hard_puzzle_acc',
```

```python
        'overall_a_acc',
        'overall_accuracy',
        'overall_command_correct',
        'overall_f_acc',
```

```python
        'schema_accuracy',
        'schema_valid',
        'tool_calls_match_rate',
```

Do not add the names to `LEGACY_METRIC_MIGRATIONS`; canonical v2 names belong only in `METRIC_DEFINITIONS`.

- [ ] **Step 4: Run the catalog and resolver tests**

Run:

```powershell
pytest tests/report/semantics/test_catalog.py tests/report/semantics/test_resolver.py -v
```

Expected: PASS, including `test_v2_registry_contains_only_canonical_non_dynamic_names`.

- [ ] **Step 5: Commit the metric semantics change**

```powershell
git add -- evalscope/metrics/semantics/catalog.py tests/report/semantics/test_catalog.py
git commit -m "fix: register customer business metrics"
```

### Task 2: Reject Metric Name Collisions

**Files:**
- Modify: `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py:132-151`
- Modify: `tests/benchmark/test_customer_multimodal_v1_adapter.py:194-222`

**Interfaces:**
- Consumes: customer record key `expected: Dict[str, JSON scalar]`.
- Produces: module constant `_RESERVED_EXPECTED_FIELDS = {'field', 'overall'}` and a `ValueError` before `Sample` creation.

- [ ] **Step 1: Replace the single-field test with a parameterized collision test**

Replace `test_customer_record_rejects_reserved_overall_field` with:

```python
@pytest.mark.parametrize('field_name', ['field', 'overall'])
def test_customer_record_rejects_fields_reserved_for_aggregate_metrics(field_name: str) -> None:
    record = load_first_record()
    record['expected'] = {field_name: 'dog', 'count': 1}

    with pytest.raises(ValueError, match=rf'{field_name}.*reserved'):
        _adapter().record_to_sample(record)
```

- [ ] **Step 2: Run the collision test and verify `field` currently passes**

Run:

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py::test_customer_record_rejects_fields_reserved_for_aggregate_metrics -v
```

Expected: one PASS for `overall` and one FAIL for `field` because it is not yet reserved.

- [ ] **Step 3: Centralize and apply the reserved-field contract**

Add near `_reject_json_constant`:

```python
_RESERVED_EXPECTED_FIELDS = {'field', 'overall'}
```

Replace the existing `name == 'overall'` branch with:

```python
            if name in _RESERVED_EXPECTED_FIELDS:
                raise ValueError(f'expected field {name!r} is reserved for an aggregate metric')
```

- [ ] **Step 4: Run the adapter collision and existing record validation tests**

Run:

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -v
```

Expected: PASS; valid customer fixtures continue to convert unchanged.

- [ ] **Step 5: Commit the collision fix**

```powershell
git add -- evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py tests/benchmark/test_customer_multimodal_v1_adapter.py
git commit -m "fix: reject customer metric name collisions"
```

### Task 3: Reject Unsupported Scoring Configuration

**Files:**
- Modify: `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py:132-214`
- Modify: `tests/benchmark/test_customer_multimodal_v1_adapter.py`

**Interfaces:**
- Consumes: flat customer JSONL records.
- Produces: module constant `_UNSUPPORTED_SCORING_FIELDS: Dict[str, str]` and explicit `ValueError` messages.

- [ ] **Step 1: Add failing tests for documented unsupported options**

Append:

```python
@pytest.mark.parametrize(
    ('config_name', 'config_value'),
    [
        ('field_weights', {'confidence': 2}),
        ('relative_tolerance', {'confidence': 0.01}),
        ('fuzzy_matching', True),
        ('normalization', {'trim': True}),
    ],
)
def test_customer_record_rejects_unsupported_scoring_configuration(
    config_name: str,
    config_value: Any,
) -> None:
    record = load_first_record()
    record['expected'] = {'confidence': 0.91}
    record[config_name] = config_value

    with pytest.raises(ValueError, match=rf'{config_name}.*not supported'):
        _adapter().record_to_sample(record)


def test_customer_record_allows_unrelated_dataset_metadata() -> None:
    record = load_first_record()
    record['business_case'] = 'retail-image-check'

    sample = _adapter().record_to_sample(record)

    assert sample.metadata['id'] == record['id']
```

- [ ] **Step 2: Run the new tests and verify unsupported options are silently ignored**

Run:

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -k "unsupported_scoring or unrelated_dataset" -v
```

Expected: four unsupported-option cases FAIL because no error is raised; unrelated metadata PASS.

- [ ] **Step 3: Add explicit unsupported-option validation**

Add near `_RESERVED_EXPECTED_FIELDS`:

```python
_UNSUPPORTED_SCORING_FIELDS = {
    'field_weights': 'field weights',
    'fuzzy_matching': 'fuzzy matching',
    'normalization': 'automatic normalization',
    'relative_tolerance': 'relative tolerance',
}
```

At the start of `record_to_sample()`, after validating `id` and before reading `expected`, add:

```python
        for field_name, feature_name in _UNSUPPORTED_SCORING_FIELDS.items():
            if field_name in record:
                raise ValueError(
                    f'{field_name} is not supported in customer_multimodal_v1; '
                    f'{feature_name} is outside the current evaluation contract'
                )
```

Do not reject all unknown record keys; only the named scoring options are fail-closed.

- [ ] **Step 4: Run all customer adapter and comparator tests**

Run:

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py tests/benchmark/test_customer_multimodal_v1_business_comparator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the unsupported-option validation**

```powershell
git add -- evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py tests/benchmark/test_customer_multimodal_v1_adapter.py
git commit -m "fix: reject unsupported customer scoring options"
```

### Task 4: Publish the New Evaluation Version and Report Contract

**Files:**
- Modify: `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py:34-38`
- Modify: `tests/benchmark/test_customer_multimodal_v1_adapter.py`
- Modify: `tests/cli/test_customer_multimodal_v1.py`

**Interfaces:**
- Consumes: `BenchmarkMeta.evaluation_version` and native customer report JSON.
- Produces: evaluation version `v1.3` and report metrics whose persisted semantics are quality ratios.

- [ ] **Step 1: Add failing version and persisted-semantics assertions**

Add to `tests/benchmark/test_customer_multimodal_v1_adapter.py`:

```python
def test_customer_multimodal_evaluation_version_is_v1_3() -> None:
    assert _adapter().benchmark_meta.evaluation_version == 'v1.3'
```

In `tests/cli/test_customer_multimodal_v1.py`, after building `metrics`, add:

```python
    for name in ['overall_accuracy', 'field_accuracy', 'overall_command_correct']:
        semantics = metrics[name]['semantics']
        assert semantics['semantic_id'] == 'quality.accuracy.ratio'
        assert semantics['direction'] == 'higher_is_better'
        assert semantics['display_multiplier'] == 100
        assert semantics['display_unit'] == '%'
```

- [ ] **Step 2: Run both tests and verify the version assertion fails**

Run:

```powershell
pytest tests/benchmark/test_customer_multimodal_v1_adapter.py::test_customer_multimodal_evaluation_version_is_v1_3 tests/cli/test_customer_multimodal_v1.py -v
```

Expected: the version test FAILS with `v1.2 != v1.3`; report semantics assertions PASS after Task 1.

- [ ] **Step 3: Bump the benchmark evaluation version**

Change:

```python
        evaluation_version='v1.3',
```

Do not edit `BenchmarkMeta.description` or generated benchmark pages in this task. The operator-facing container guide in
the second implementation plan documents runtime input behavior, while this task remains focused on the scoring contract.

- [ ] **Step 4: Run the complete scoring closure suite**

Run:

```powershell
pytest tests/report/semantics/test_catalog.py tests/report/semantics/test_resolver.py tests/benchmark/test_customer_multimodal_v1_adapter.py tests/benchmark/test_customer_multimodal_v1_business_comparator.py tests/cli/test_customer_multimodal_v1.py -v
pytest tests/api/test_benchmark_registry_contract.py -v
```

Expected: PASS with no customer fixed metric degraded to diagnostic semantics.

- [ ] **Step 5: Commit the versioned report contract**

```powershell
git add -- evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py tests/benchmark/test_customer_multimodal_v1_adapter.py tests/cli/test_customer_multimodal_v1.py
git commit -m "test: close customer scoring contract"
```

### Task 5: Run Submission Checks for the Scoring Phase

**Files:**
- Verify only; no expected source changes.

**Interfaces:**
- Consumes: all Task 1-4 commits.
- Produces: evidence that the scoring phase is independently mergeable before container work starts.

- [ ] **Step 1: Run formatting and import checks on the changed Python files**

```powershell
pre-commit run --files evalscope/metrics/semantics/catalog.py evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py tests/report/semantics/test_catalog.py tests/benchmark/test_customer_multimodal_v1_adapter.py tests/cli/test_customer_multimodal_v1.py
make lint-imports
```

Expected: PASS. If a formatter changes a file, inspect the diff and rerun only the affected targeted tests.

- [ ] **Step 2: Run the project CI smoke test**

```powershell
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings
```

Expected: PASS.

- [ ] **Step 3: Confirm the worktree contains no accidental files**

```powershell
git status --short
git diff --check
```

Expected: only the pre-existing untracked `docs/reports/` directory remains; no scoring implementation changes are
uncommitted.
