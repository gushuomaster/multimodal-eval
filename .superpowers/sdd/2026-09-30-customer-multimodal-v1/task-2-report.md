# Task 2 TDD Report

## RED

Added adapter tests covering OpenAI image message conversion, stable JSON targets, field-level scoring, malformed JSON, and missing expected fields.

Initial `pytest tests/benchmark/test_customer_multimodal_v1_adapter.py -q` on the default Python 3.14 interpreter failed during collection because `colorlog` and then `filetype` were unavailable. A complete editable install on Python 3.14 was also blocked by the `editdistance` Cython build using the system code page. These were environment failures before test execution, not assertion results.

The tests were then run in a project-local Python 3.11.15 `uv` environment after installing the project and pytest. After implementation, the same focused command collected 8 tests and passed; the pre-implementation assertion RED was therefore not observable in the original incomplete dependency environment.

## GREEN

Implemented `CustomerMultimodalV1Adapter` with:

- OpenAI message parsing through `chat_messages_from_openai`.
- Stable, sorted JSON serialization for `Sample.target`.
- Record id and expected field names in `Sample.metadata`.
- Scalar normalization and per-field plus overall accuracy metrics.
- Fail-closed parse handling with `score.metadata['parse_error'] = True`.
- Registered `BenchmarkMeta` using supported fields only, including evaluation version `v1.0`.

Focused test output:

```text
8 passed in 30.41s
```

Additional checks:

```text
ruff check evalscope/benchmarks/customer_multimodal_v1 tests/benchmark/test_customer_multimodal_v1_adapter.py
All checks passed!

git diff --check
passed
```

The temporary validation environment was created under `.artifacts/customer_multimodal_v1/venv`.
