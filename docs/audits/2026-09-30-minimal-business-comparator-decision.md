# Minimal Business Comparator — Implementation Decision

日期：2026-09-30
范围：`customer_multimodal_v1` Phase 3B 结构化业务比较能力。

## Decision

采用 benchmark-local reusable helper：

```text
customer_multimodal_v1 strict JSON
        → optional Draft 2020-12 JSON Schema
        → BusinessComparator
        → EvalScope Score / aggregation / report
```

`BusinessComparator` 位于
`evalscope/benchmarks/customer_multimodal_v1/business_comparator.py`。当前只有一个明确消费者，因此不提升为
Generic Core abstraction，也不复用与 tool envelope、function name 和 `finish_reason='tool_calls'` 耦合的
General-FC validator。

## Business Contract

- 字符串、整数、浮点数、布尔值和 null 使用严格类型及 exact value comparison。
- `True != 1`，`False != 0`。
- 字符串不执行大小写、空白、标点或语义归一化。
- 可选 `tolerance.<field>.absolute` 只适用于 numeric expected field；边界值包含在容差内。
- 可选 `critical_fields` 决定 `overall_command_correct`；未声明时所有 expected fields 都是关键字段。
- 可选 `schema` 直接交给 `jsonschema.Draft202012Validator`，不重写 JSON Schema engine。
- schema failure 不抹掉仍可计算的字段业务准确率，但 command verdict 必为 0。
- strict JSON failure 将字段、schema（如配置）和 command 指标置 0，并记录 `parse_error`。

## Metrics

- `accuracy`：canonical primary metric，保持现有报告入口。
- `overall_accuracy`：兼容别名，值等于无权重字段均值。
- `field_accuracy`：显式业务字段均值，值等于正确字段数除以 expected 字段数。
- `overall_command_correct`：schema gate 与关键字段业务规则共同决定的二值指标。
- `schema_valid`：仅 schema-enabled case 输出。

字段级错误继续存储在 `Score.metadata`；不增加 evidence store、数据库或第二套报告。

## Version Decision

`evaluation_version` 从 `v1.1` 提升到 `v1.2`。理由不是内部重构，而是正式增加了 schema、absolute tolerance、
critical fields、`field_accuracy` 和 `overall_command_correct` 可用评分语义，同时字符串比较从历史兼容归一化改为
严格 exact match。

## Explicit Deferrals

以下能力为 `NOT_IMPLEMENTED`：

- field weights
- relative tolerance
- semantic or fuzzy normalization
- LLM-based normalization
- generic structured-evaluation Core framework

## Core Boundary

Phase 3B 不修改 `evalscope/run.py`、`evalscope/config.py`、`evalscope/api/`、`evalscope/models/`、
`evalscope/evaluator/` 或 `evalscope/metrics/`。EvalScope 现有 Metric、Aggregation、Report、Benchmark、Dataset、
Registry 和 persistence 能力保持复用。
