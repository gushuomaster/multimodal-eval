# Customer Multimodal v1 Design

## Goal

Build the first business-specific multimodal benchmark on top of EvalScope without modifying EvalScope Core, proving that local single-image and multi-image samples can be evaluated against structured field-level expectations.

## Scope

This phase covers one native benchmark named `customer_multimodal_v1` and its deterministic structured-output scorer. It does not cover free-text LLM judging, robustness generation, stability testing, security runners, web dashboard changes, VLMEvalKit, RAGEval, or the GJB control layer.

## Data Contract

The local dataset uses JSONL records with this shape:

```json
{
  "id": "sample-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "请识别图片中的对象及其属性。"},
        {"type": "image_url", "image_url": {"url": "image.jpg"}}
      ]
    }
  ],
  "expected": {
    "object": "安全帽",
    "color": "黄色",
    "count": 1
  }
}
```

`messages` follows the existing OpenAI-compatible message format and may contain one or more image parts. `expected` is a JSON object with scalar field values used for deterministic comparison. The adapter stores the serialized expectation as the EvalScope `Sample.target` string and preserves the original field names in sample metadata.

## Evaluation Flow

```text
JSONL record
  -> customer_multimodal_v1 adapter
  -> Sample(input=ChatMessage list, target=JSON string)
  -> existing ModelAPI/OpenAICompatibleAPI
  -> model response
  -> JSON response parser
  -> per-field exact comparison
  -> field metrics and aggregate report
```

The adapter reuses `VisionLanguageAdapter` for media parsing and request construction. It registers through `register_benchmark` and uses the standard native evaluator/output structure.

## Scoring Contract

For each expected field, parse the model answer as a JSON object and compare the field value with the expected value. A field is correct only when the field exists and its normalized value equals the expected value. The first version supports scalar strings, numbers, booleans, and null values; nested objects and list-specific matching are out of scope.

The report exposes:

- one accuracy value per expected field;
- `overall_accuracy`, defined as correct fields divided by expected fields across the sample;
- sample counts and parse-error counts in metadata.

Malformed or non-JSON model output is recorded as a parse error and receives zero correct fields; it is not silently treated as full credit.

## Files and Boundaries

- `evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py`: adapter, record conversion, JSON parsing, deterministic field scoring, and benchmark metadata.
- `custom_eval/multimodal/customer_v1/`: minimal local JSONL fixtures and referenced image files for smoke tests.
- `tests/benchmark/test_customer_multimodal_v1_adapter.py`: adapter and scoring tests, including single-image, multi-image, valid JSON, missing field, and malformed output cases.
- `tests/cli/test_customer_multimodal_v1.py`: native mock-model smoke test proving registry discovery, local loading, evaluation, and report persistence.
- `docs/superpowers/plans/2026-09-30-customer-multimodal-v1.md`: implementation plan generated after this design is approved.

No file under `evalscope/api`, `evalscope/evaluator`, `evalscope/models`, `evalscope/metrics`, or `evalscope/report` is modified by this phase.

## Verification

The implementation must pass focused adapter tests, the native mock-model smoke test, Python compilation, line-width checks, and the benchmark registry contract. A real MiniMax-M3 run is an optional external verification after deterministic local tests pass; its API key must come from the environment and never be committed.

## Non-Goals and Constraints

- Do not change `pyproject.toml` or optional dependency groups.
- Do not install all optional extras.
- Do not hand-edit generated benchmark documentation or `_index.json`; run the repository generation command if the new adapter requires generated registry metadata.
- Do not add a parallel runner, metric framework, or API dispatch system.
