# Phase 3A — EvalScope Structured Evaluation Capability Audit

审计基线：`7b9b7e1526f3bef9281133164c6695e9b4b96885`

范围：当前 HEAD 的 EvalScope structured-output evaluation、`general_fc`、`response_schema`、Score/Metric/Report，
以及 `customer_multimodal_v1`。本阶段只审计和添加 characterization tests，不实施 Phase 3B，不修改 Generic Core。

## A. Current EvalScope Capabilities

| Capability | Status | Evidence and boundary |
|---|---|---|
| JSON | PARTIAL | `FunctionCallAdapter.validate_tool_call()` 和 `ToolFunction` 使用普通 `json.loads()`，可解析 tool arguments，但没有普通结构化响应的公共 strict parser，也会接受 `NaN`/`Infinity`。见 `evalscope/api/benchmark/adapters/function_call_adapter.py:59-90`、`evalscope/api/tool/tool_call.py:18-25`。 |
| Schema | SUPPORTED | `general_fc` 对保留在 `Sample.metadata` 中的原始 schema 调用 `jsonschema.validate()`。公共 `validate_tool_arguments()` 也调用同一依赖，但经过受限 `ToolInfo`/`JSONSchema` 模型。见 `evalscope/benchmarks/general_fc/general_fc_adapter.py:67-87`、`evalscope/api/tool/utils.py:70-93`。 |
| Required | SUPPORTED | 原始 schema 的 `required` 由 `jsonschema` 执行；characterization test 已覆盖 missing required。 |
| Type | SUPPORTED | 原始 schema 的 scalar/object/array type 由 `jsonschema` 执行；characterization test 已覆盖 wrong type。 |
| Enum | SUPPORTED | 原始 schema 的 `enum` 由 `jsonschema` 执行。 |
| Range | PARTIAL | `general_fc` 验证时使用的原始 schema 支持 `minimum`/`maximum`；但 EvalScope 的 `JSONSchema` 模型没有这些字段，因此 `ResponseSchema` 和发送给模型的 `ToolInfo` typed path 会丢弃它们。见 `evalscope/utils/json_schema.py:30-61`。 |
| Nested | SUPPORTED | 原始 JSON Schema 支持 nested object；characterization test 已覆盖 nested required。 |
| Array | SUPPORTED | 原始 JSON Schema 支持 array/items；characterization test 已覆盖 item type。 |
| Aggregation | SUPPORTED | `Score.value` 可含多个数值 key；`Mean` 自动逐 key 聚合，某 key 缺失时该 sample 不计入该 key 的 `num`。见 `evalscope/api/metric/scorer.py:46-80`、`evalscope/metrics/aggregators/aggregators.py:55-94`。 |
| Diagnostics | PARTIAL | `Score.metadata`、`SampleScore.sample_metadata` 会进入 review；prediction 可保存 sample metadata。`general_fc` 只保存第一个人类可读 `error_reason`。标准 report flatten 时只取 score/num/identity，不保留 `AggScore.metadata`。见 `evalscope/api/metric/scorer.py:72-73,120-136`、`evalscope/report/generator.py:103-129`、`evalscope/api/evaluator/cache.py:263-287,497-523`。 |
| Report | SUPPORTED | 任意 `Score.value` key 可聚合并进入 report。未声明指标会保留值但降级为 diagnostic，并记录 `[metric-semantics]` warning；diagnostic 不能作为 primary。见 `evalscope/report/generator.py:185-225`、`evalscope/metrics/semantics/resolver.py:126-130,183-193`。 |

### JSON parsing and strictness

- EvalScope 当前没有面向普通 structured response 的公共 strict JSON parser。
- Python `json.loads()` 默认接受非标准常量 `NaN`、`Infinity`、`-Infinity`；当前 tool paths 没有传入
  `parse_constant`。characterization test 证明 `NaN` 会进入 arguments，并可通过仅要求 `type: number` 的 schema。
- `parse_tool_call()` 会把 malformed object arguments 变成 `{}`，同时记录 `ToolCall.parse_error`；
  `FunctionCallAdapter.validate_tool_call()` 不读取 `parse_error`，所以没有 required 字段的 permissive schema
  仍可能验证成功。见 `evalscope/api/tool/utils.py:15-63`。
- `customer_multimodal_v1` 自己通过 `json.loads(..., parse_constant=...)` 拒绝非有限常量，这是 benchmark-local
  行为，不是 EvalScope 通用能力。

### `response_schema`: generation constraint, not evaluation validation

- `GenerateConfig.response_schema` 的源码注释明确写明“output should still be validated”，所以它只是
  **Generation Constraint**，不是 **Evaluation Validation**。见 `evalscope/api/model/generate_config.py:155-156`。
- OpenAI Chat Completions path 将其发送为 `response_format.type=json_schema`；OpenAI Responses path 将其发送为
  `text.format`。见 `evalscope/models/utils/openai.py:283-292`、
  `evalscope/models/utils/openai_responses.py:141-151`。
- 它不是所有 `ModelAPI` 的统一能力。参数文档声明 OpenAI、Google、Mistral 支持；当前直接源码引用集中在
  OpenAI-compatible/Responses request builders。
- provider 拒绝 schema 时走普通 provider request error 路径；没有自动把失败转换成 schema score，也没有在
  返回后再次校验内容。
- `ResponseSchema.json_schema` 使用受限 `JSONSchema` 模型；当前模型支持 type/format/default/enum/items/
  properties/additionalProperties/anyOf/required，但不建模 minimum/maximum/pattern 等关键字。

### Score, metrics, primary metric, and persistence

以下是有效的 per-sample 表达：

```python
Score(
    value={
        'json_valid': 1.0,
        'schema_valid': 1.0,
        'field_accuracy': 0.8,
        'overall_command_correct': 0.0,
    },
    metadata={'schema_errors': []},
)
```

- 不要求每个 `Score.value` key 都有独立 metric scorer registry entry；adapter 可以直接产生这些 key。
- 默认 `mean` aggregator 自动发现并逐 key 聚合。
- report semantics 仍需声明：未声明 key 会 warning 并降级为无方向、无单位的 diagnostic。
- `Score.main_score_name` 只选择单个 sample 内的主值；report primary 必须由
  `BenchmarkMeta.primary_metric` 指定，并且必须解析为 quality metric，不能只是 diagnostic。
- 详细 parse/schema/value diagnostics 应放 `Score.metadata`，因为它随 review 保存；输入侧业务上下文放
  `Sample.metadata`/`SampleScore.sample_metadata`。标准 report 适合保存聚合指标，不适合保存逐样本错误列表。

## B. General-FC Reuse Analysis

完整调用链：

```text
Dataset record
→ GeneralFCAdapter.record_to_sample()
→ Sample.tools + raw tools in Sample.metadata
→ FunctionCallAdapter._on_inference(model.generate(..., tools=...))
→ ModelOutput.message.tool_calls
→ GeneralFCAdapter.match_score()
→ FunctionCallAdapter.validate_tool_call()
→ Score(value + error_reason)
→ GeneralFCAdapter.aggregate_scores()
→ EvalScope Report
```

关键源码：`evalscope/benchmarks/general_fc/general_fc_adapter.py:67-132,136-197`、
`evalscope/api/benchmark/adapters/function_call_adapter.py:41-95`。

| General-FC capability | Status | Notes |
|---|---|---|
| JSON parse | PARTIAL | 使用普通 `json.loads()`；非严格 JSON。`ToolFunction` 正常建模后 arguments 已是 dict，使 validator 内的 string branch 通常不是主要路径。 |
| required field | SUPPORTED | 原始 schema + `jsonschema.validate()`。 |
| type validation | SUPPORTED | 原始 schema + `jsonschema.validate()`。 |
| enum | SUPPORTED | 原始 schema + `jsonschema.validate()`。 |
| range | SUPPORTED | validation 使用的原始 schema 保留 `minimum`/`maximum`；但 `record_to_sample()` 构造的 `Sample.tools` 经过 `ToolInfo`，发送给模型的 typed tool schema 会丢弃这些未知关键字。这是请求契约与验证契约可能不一致的边界。 |
| nested object | SUPPORTED | 原始 schema 递归验证。 |
| array | SUPPORTED | 原始 schema 验证 `items`。 |
| additionalProperties | SUPPORTED | 原始 schema 可禁止未知字段。 |
| schema_accuracy | SUPPORTED | `valid attempted samples / attempted samples`；attempt 由 `finish_reason == 'tool_calls'` 判定，是 sample-level，不是每个 tool call 各计一次。 |
| diagnostics | PARTIAL | 只返回第一个错误的 `error_reason` string；没有结构化 missing/type/range error list，且不消费 `ToolCall.parse_error`。 |
| aggregation | SUPPORTED | adapter 自定义 tool-call counts、sample-level schema accuracy、tool-call decision F1。 |

Reusable components:

- 已安装并实际使用的 `jsonschema` 依赖及其完整 raw-schema validation 行为。
- `Score`、`SampleScore`、官方 aggregation、metric semantics、review/report persistence。
- 公共 `evalscope.api.tool.validate_tool_arguments()` 可用于真正的 tool execution 参数校验。

Coupled components:

- `FunctionCallAdapter.validate_tool_call()` 要求 `ToolCall` list、函数名查找、OpenAI tool schema envelope。
- `GeneralFCAdapter.match_score()` 依赖 `ModelOutput.message.tool_calls` 和 `finish_reason == 'tool_calls'`。
- `GeneralFCAdapter.aggregate_scores()` 混合 schema validity 与“是否应该调用工具”的 precision/recall/F1 语义。

Direct reuse: **NO**。普通 JSON response 没有 function name、tool-call list 或 tool finish reason；伪造这些对象会把业务评测
绑到错误的协议语义。

Thin adapter required: **YES**，结论为 `REUSE_WITH_THIN_ADAPTER`。普通 structured response 应在 benchmark-local
helper 中 strict-parse JSON，再以 raw dict schema 调 `jsonschema`，最后把标准数值指标和诊断写入 `Score`。

Not reusable:

- tool-call decision F1、tool-call count、tool name dispatch。
- 以 `finish_reason` 定义 schema attempt 的 aggregation。
- 只返回第一个 string error 的诊断形状（若客户需要结构化错误分类）。

### Validator options and recommendation

**推荐方案：C — benchmark-local 薄适配层直接使用现有 `jsonschema` dependency。**

理由：可保留完整 raw schema（包括 range 等关键字），不需要伪造 function-call 对象，不修改 Core，并能把
strict parsing 与客户业务 comparator 明确分层。它不是新 Schema Engine，只是一次直接 library call。

其他方案：

| Option | Pros | Cons | Suitable when |
|---|---|---|---|
| A. 直接调用 `FunctionCallAdapter.validate_tool_call()` | 最少表面代码；复用现有错误字符串 | 强耦合 `ToolCall`、函数名和 tool envelope；忽略 `parse_error`；普通响应需要伪造对象 | 只有输入本来就是真实 tool calls 时 |
| B. 使用公共 `validate_tool_arguments()` | 公共 API；错误 path 比 GeneralFC 更清晰 | 仍要求 `ToolCall`/`ToolInfo`；受限 typed schema 会丢 range；invalid schema 被视为 unconstrained | 原生 agent/tool execution 参数校验 |
| D. 抽取 Core 公共 validator | 多 benchmark 可共享统一 helper | 违反本阶段 Core 约束；只有一个明确新消费者，抽象时机过早；需要更广回归 | 至少两个非 tool benchmark 出现相同稳定契约，并单独批准 Core 变更后 |

## C. customer_multimodal_v1 Duplication

| Current implementation | EvalScope equivalent | Decision |
|---|---|---|
| `_parse_object()` strict parses object and rejects `NaN`/`Infinity` | 仅有非严格、tool-specific parse；无普通响应 strict helper | 保留为 `Refactor Candidate`，未来移到 benchmark-local validation helper；`REUSE_WITH_THIN_ADAPTER` |
| Dataset `expected` non-empty object validation | Data adapter lifecycle 可承载验证，但字段契约由 benchmark 定义 | `BUSINESS_EXTENSION` |
| Expected scalar-only、finite number、reserved `overall` rules | 无通用等价；这些是客户数据契约 | `BUSINESS_EXTENSION` |
| String trim/lower normalization | 无通用等价，且属于评分语义 | `BUSINESS_EXTENSION`；未获客户规则前不扩展 hyphen/punctuation/semantic normalization |
| bool-number distinction and exact expected-value comparison | 无通用 structured business comparator | `BUSINESS_EXTENSION` |
| 动态 `<field>_accuracy` + `overall_accuracy` + canonical `accuracy` | `Score.value`、Mean aggregation、report 已支持动态 key | 表达与聚合 `REUSE_OFFICIAL`；字段判定逻辑仍为 `BUSINESS_EXTENSION` |
| `parse_error` in `Score.metadata` | review 原生持久化 `Score.metadata` | `REUSE_OFFICIAL`，不是重复存储 |
| `aggregate_scores()` 增加 `parse_error_count` metadata | 官方 aggregation 可聚合数值 key，但标准 report 丢弃 `AggScore.metadata` | 只为当前 sidecar 服务；Phase 3B 可评估改为显式 diagnostic metric，但会改变 v1.1 artifact contract |
| `generate_report()` diagnostics JSONL sidecar | review 已保存详细 per-sample metadata；标准 report 保存聚合数值 | `Refactor Candidate`，不在 Phase 3A 删除；若删除/替换需单独版本化 |
| `load_from_disk(use_local_loader=True)` | EvalScope 官方 local loader | `REUSE_OFFICIAL`，已正确复用 |
| Vision-language message conversion/model invocation | `VisionLanguageAdapter` + OpenAI message conversion | `REUSE_OFFICIAL`，已正确复用 |

当前相关资产：

- Adapter：`evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py`
- Fixture：`custom_eval/multimodal/customer_v1/example.jsonl`
- Adapter tests：`tests/benchmark/test_customer_multimodal_v1_adapter.py`
- CLI/report tests：`tests/cli/test_customer_multimodal_v1.py`
- Generated docs：`docs/{en,zh}/benchmarks/customer_multimodal_v1.md`（不可手工编辑）

## D. Capability Matrix

Decision 仅使用任务指定枚举。

| Capability | EvalScope Support | Existing Location | customer_v1 Implementation | Decision |
|---|---|---|---|---|
| Strict JSON | NOT_SUPPORTED as generic API | Tool paths use ordinary `json.loads()` | `_parse_object()` rejects constants and non-object JSON | REUSE_WITH_THIN_ADAPTER |
| JSON Schema | PARTIAL as reusable surface; SUPPORTED in raw GeneralFC path | `FunctionCallAdapter.validate_tool_call()` / `validate_tool_arguments()` / `jsonschema` dependency | None | REUSE_WITH_THIN_ADAPTER |
| Required Fields | SUPPORTED for raw schema | `jsonschema.validate()` | Only validates expected object itself, not predicted schema | REUSE_WITH_THIN_ADAPTER |
| Type Validation | SUPPORTED for raw schema | `jsonschema.validate()` | Exact scalar comparison implicitly rejects mismatched values | REUSE_WITH_THIN_ADAPTER |
| Enum | SUPPORTED for raw schema | `jsonschema.validate()` | None | REUSE_WITH_THIN_ADAPTER |
| Numeric Range | PARTIAL; raw GeneralFC supports, typed schema drops keywords | Raw metadata schema + `jsonschema`; not `JSONSchema` model | Only finite-number rule, no min/max | REUSE_WITH_THIN_ADAPTER |
| Additional Properties | SUPPORTED for raw schema | `jsonschema.validate()` | Extra predicted fields are ignored | REUSE_WITH_THIN_ADAPTER |
| Nested Object | SUPPORTED for raw schema | `jsonschema.validate()` | Expected nested values are prohibited | NOT_NEEDED |
| Array | SUPPORTED for raw schema | `jsonschema.validate()` | Expected arrays are prohibited | NOT_NEEDED |
| Exact Value Compare | No generic structured comparator | Benchmark-specific scorers only | `_scalars_equal()` | BUSINESS_EXTENSION |
| Float Tolerance | No generic reusable policy; only benchmark-specific implementations found | Examples such as OfficeQA/MeasureBench are task-coupled | Exact numeric equality | BUSINESS_EXTENSION |
| Field Accuracy | `Score.value`/Mean/report can express and aggregate it | Score + Mean | Emits dynamic `<field>_accuracy` | BUSINESS_EXTENSION |
| Overall Command Pass | Can be represented, but decision rule is not generic | `Score.value` | Uses mean field accuracy, not all-or-nothing command pass | BUSINESS_EXTENSION |
| Diagnostics | `Score.metadata`/review/sample metadata supported; report aggregate metadata omitted | Score, cache, report generator | parse error flag + aggregate sidecar | REUSE_OFFICIAL |
| Aggregation | SUPPORTED | Mean aggregator | Delegates to superclass, then annotates aggregates | REUSE_OFFICIAL |
| Primary Metric | SUPPORTED | `BenchmarkMeta.primary_metric`, semantics resolver | Canonical `accuracy` is primary | REUSE_OFFICIAL |
| Report Persistence | SUPPORTED for metrics and review metadata | cache + report generator | Standard report plus custom sidecar | REUSE_OFFICIAL |

Notes:

- `REUSE_WITH_THIN_ADAPTER` does not mean复用 GeneralFC 的 tool semantics；它表示复用现有 adapter extension point、
  `jsonschema` dependency、Score/aggregation/report，并只在 benchmark-local 层连接普通 JSON response。
- Nested object/array 对当前 v1.1 scalar-only 客户契约是 `NOT_NEEDED`；若未来客户 schema 允许它们，可直接转为
  `REUSE_WITH_THIN_ADAPTER`，无需新 engine。

## E. Actual Business Gaps

真正需要我们维护的范围只有客户评分策略：

```text
BusinessComparator
├── expected-value comparison
├── optional numeric tolerance policy
├── optional normalization policy
├── field weights / critical-field rules
├── overall command PASS/FAIL decision
└── business diagnostics (mismatch classification and context)
```

具体边界：

- Expected value comparator：客户定义 predicted vs expected 的判定，不属于 JSON Schema。
- Float tolerance：没有可直接复用的通用 EvalScope comparator；现有 tolerance 都是 benchmark-specific，故为
  `BUSINESS_EXTENSION` 候选。必须由客户定义 absolute/relative tolerance、单位、边界包含性。
- Normalization：`black and white` vs `black-and-white` 只能证明存在候选规则，不能证明应默认等价。当前不改语义。
- Field weights、critical field veto、overall all-or-nothing pass 都需要客户规则。
- 业务 diagnostics 的字段名和分类属于扩展；存储和持久化复用 `Score.metadata`/review。

不需要自研：parser framework、schema engine、metric engine、aggregation engine、report engine、runner、benchmark engine。

## F. Core Gap

```text
Core modification required: NO
```

现有 `Benchmark Adapter → Score → aggregation → report/review` 扩展链足以承载普通 structured response 的解析、
schema validity metrics 和业务 comparator。`response_schema` 的 typed schema 不完整、report 不保存
`AggScore.metadata` 都是明确限制，但均有 benchmark-local/标准 review 路径可绕过，不构成本任务的不可绕过 Core gap。

本阶段 Generic EvalScope Core Modified = **NO**。

## G. Minimal Phase 3B Proposal

推荐：只有在客户确认要引入 schema validation/tolerance/overall command semantics 后，实施一个
benchmark-local `BusinessComparator`；如果目标只是保持当前 v1.1 exact-scalar 行为，Phase 3B 的最小新增生产代码为 **0 行**。

```text
Files:
  NEW    evalscope/benchmarks/customer_multimodal_v1/structured_validation.py
  MODIFY evalscope/benchmarks/customer_multimodal_v1/customer_multimodal_v1_adapter.py
  MODIFY tests/benchmark/test_customer_multimodal_v1_adapter.py
  MODIFY tests/cli/test_customer_multimodal_v1.py (only if persisted metrics/artifacts change)
  REGENERATE benchmark docs/meta only if BenchmarkMeta.description or evaluation semantics change

Reuse:
  VisionLanguageAdapter lifecycle
  existing jsonschema dependency (raw dict schema)
  Score / Score.metadata / SampleScore
  Mean aggregation
  BenchmarkMeta.primary_metric and metric semantics
  standard prediction, review, and report persistence

New code:
  strict JSON object parsing wrapper
  raw-schema validation adapter
  BusinessComparator policy object/functions
  structured mismatch diagnostics written to Score.metadata

Refactor:
  move _parse_object() and comparison helpers out of the adapter only when new policy is approved
  keep current behavior until replacement tests prove semantic equivalence
  do not reuse or subclass GeneralFC scoring/aggregation
  do not remove diagnostics sidecar unless its replacement artifact and version change are approved

Tests:
  valid/invalid strict JSON, including NaN/Infinity
  required/type/enum/range/additionalProperties/nested/array schema cases as applicable
  exact comparison, bool-vs-number, null, case/trim behavior
  customer-approved tolerance and normalization boundary cases
  field metrics, overall pass, diagnostics persistence, primary metric, cache/report regression
  current mock CLI and registry contracts

Estimated production change:
  80–140 LOC when schema + configurable comparator policy are both approved
Estimated test change:
  100–160 LOC
Estimated complexity:
  Medium
```

Semantic impact:

- 仅重排 helper 且输出完全相同：不影响 v1.1，但仍需 equivalence tests。
- 新增 schema-invalid 判定、numeric tolerance、normalization、field weights、overall command pass 或改变 sidecar：
  会改变公开评分/产物语义，应将 `evaluation_version` 至少升为 v1.2，并在独立 Phase 3B 获得确认。

## H. Regression

```text
customer_multimodal_v1: PASS (40 tests)
Registry:                PASS (27 tests)
Import contracts:        PASS (5 kept, 0 broken)
Characterization tests:  PASS (11 tests)
Combined pytest:         PASS (78 tests)
Ruff:                    PASS (changed files)
Format:                  PASS (changed files)
Encoding:                PASS (UTF-8 without BOM, LF)
git diff --check:        PASS
Core boundary:           PASS (only docs/audits and tests/benchmark changed)
Secret scan:             PASS (changed files)
```

## I. Git

```text
Branch:       feature/structured-eval-audit
HEAD:         audit commit at branch tip; exact hash reported in final handoff
Base:         7b9b7e1526f3bef9281133164c6695e9b4b96885
Files changed: 2 (audit document + characterization tests)
Working tree: clean after audit commit; verified in final handoff
Merge status: NOT MERGED (awaiting user confirmation)
```

## Audit conclusion

EvalScope 已经具备 adapter lifecycle、raw-schema validation 依赖、multi-metric Score、aggregation、primary metric、
review/report persistence。需要自研的只有客户业务 comparator 与客户定义的诊断内容；普通 structured response 只需
benchmark-local 薄适配，不需要新框架，也不需要修改 Core。

Final status after fresh verification: `STRUCTURED_EVALUATION_CAPABILITY_AUDIT_COMPLETE`
