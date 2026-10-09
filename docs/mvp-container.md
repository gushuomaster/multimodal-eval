# MVP 容器评测操作指南

后端调用宿主机脚本 `scripts/mvp/run_container.py`，传入数据目录、输出目录和模型参数。
脚本生成完整运行配置，前台启动 Docker，透传日志和退出码；后端从输出目录读取状态、报告和逐条结果。
推荐每个任务使用独立输出目录，保留 runtime 和全部评测产物，便于核对实际参数和失败原因。
安全加固、并发任务管理、后端 HTTP 服务，以及输出产物和 runtime 配置的自动清理均留待 MVP 后续阶段实现。

## 1. 构建镜像并运行任务

宿主机需要可用的 Docker Linux 容器环境、Python 3.10 及以上版本，以及安装了当前 checkout 的 Python 环境。
以下命令均在仓库根目录执行；`python` 应指向该环境。首次准备环境可执行 `python -m pip install -e .`。
镜像使用 Python 3.11，只安装项目基础依赖，不安装可选 extras；日志不经过 Python 输出缓冲。

```powershell
docker build -f docker/mvp/Dockerfile -t multimodal-eval:mvp .
```

### PowerShell：先用 mock 跑通执行链路

```powershell
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$dataRoot = (Get-Location).Path
$jobId = 'mock-' + [guid]::NewGuid().ToString('N')
$jobOutput = "D:/project/agent_eval/multimodal-eval-artifacts/artifacts/mvp-container/$jobId"

python scripts/mvp/run_container.py `
  --data-dir "$dataRoot" `
  --dataset-local-path custom_eval/multimodal/customer_v1 `
  --output-dir "$jobOutput"
$jobExitCode = $LASTEXITCODE
Write-Output "任务退出码：$jobExitCode"
if (Test-Path -LiteralPath "$jobOutput/run_status.json") {
  Get-Content -LiteralPath "$jobOutput/run_status.json" -Encoding UTF8
}
```

默认使用 `mock_llm`、本地 `example.jsonl`、最多 2 条样本，不调用模型 API。
mock 默认响应不是业务要求的 JSON，因此报告中的业务得分可能全为 0；退出码 0 表示执行链路完成。

### PowerShell：OpenAI 兼容模型参数示例

以下地址、模型和 key 都是占位符，使用前替换为实际服务参数；`--model` 是服务识别的模型名，
`--model-id` 是报告目录使用的标识。示例使用 PowerShell 7.3 及以上的原生参数传递方式，保留 JSON 双引号。

```powershell
$PSNativeCommandArgumentPassing = 'Standard'
$generationJson = '{"temperature":0,"max_tokens":512}'
$jobId = 'api-' + [guid]::NewGuid().ToString('N')
$jobOutput = "D:/project/agent_eval/multimodal-eval-artifacts/artifacts/mvp-container/$jobId"

python scripts/mvp/run_container.py `
  --data-dir "$dataRoot" `
  --dataset-local-path custom_eval/multimodal/customer_v1 `
  --output-dir "$jobOutput" `
  --eval-type openai_api `
  --model YOUR_VISION_MODEL `
  --model-id customer-api-demo `
  --api-url https://api.example.com/v1 `
  --api-key YOUR_API_KEY `
  --limit 2 `
  --eval-batch-size 1 `
  --generation-config $generationJson
$jobExitCode = $LASTEXITCODE
```

Windows PowerShell 5.1 和旧版 PowerShell 使用旧的原生命令参数传递规则；运行上述调用前，改用下面两行准备参数，
不要只给 JSON 加外层单引号后直接传给 `python.exe`。新的 `Standard` 模式不需要这一步反斜杠转义。

```powershell
$generationJson = '{"temperature":0,"max_tokens":512}'
$generationJson = $generationJson.Replace('"', '\"')
```

后端通过进程参数数组调用脚本时，直接传入 JSON 字符串 `{"temperature":0,"max_tokens":512}`，
不添加 shell 引号或反斜杠转义。

### Linux：同一个启动器

```bash
export PYTHONUTF8=1
data_root="$(pwd)"
job_id="mock-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
artifact_root="${HOME}/multimodal-eval-artifacts/artifacts/mvp-container"
job_output="${artifact_root}/${job_id}"
python scripts/mvp/run_container.py \
  --data-dir "$data_root" \
  --dataset-local-path custom_eval/multimodal/customer_v1 \
  --output-dir "$job_output" \
  --generation-config '{"temperature":0,"max_tokens":512}'
job_exit_code=$?
printf '任务退出码：%s\n' "$job_exit_code"
```

## 2. 启动参数与默认值

未提供的评测参数取自模板。下表列出随仓库发布的模板默认值；选择其他模板后，以该模板为准。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--image` | `multimodal-eval:mvp` | 已构建的镜像名。 |
| `--config-template` | `configs/mvp/customer_multimodal_v1.yaml` | 宿主机完整 TaskConfig YAML，相对路径按调用目录解析。 |
| `--data-dir` | 必填 | 已存在的宿主机数据根，挂载到 `/eval/data`。 |
| `--dataset-local-path` | `custom_eval/multimodal/customer_v1` | 相对数据根的现有目录或 `.jsonl` 文件；不能使用绝对路径、空组件、`.` 或 `..`。 |
| `--output-dir` | 必填 | 宿主机任务输出目录；自动创建，挂载到 `/eval/output`。 |
| `--model` | `text_generation` | mock 模型名；API 模式改为服务的模型名。 |
| `--model-id` | `null` | 未指定时由 TaskConfig 根据模型名推导，用于报告等产物目录。 |
| `--eval-type` | `mock_llm` | 本指南使用 `mock_llm` 或 `openai_api`。 |
| `--api-url` | `null` | OpenAI 兼容服务的 API base URL，通常以 `/v1` 结尾。 |
| `--api-key` | `EMPTY` | 传给模型服务的 key。 |
| `--limit` | `2` | 非负整数为每个子集的样本上限，`0` 表示不限；`0` 与 `1` 之间的小数为比例，例如 `0.5`。 |
| `--eval-batch-size` | `1` | 评测请求的批量设置。 |
| `--generation-config` | `{}` | JSON 对象，例如 temperature、max_tokens；整体替换模板中的对象，缺省字段再由框架补全。 |

默认数据集为 `customer_multimodal_v1`，`subset_list` 为 `['example']`。更换数据文件名或子集时，
另存一份完整模板并调整 `dataset_args.customer_multimodal_v1.subset_list`，通过 `--config-template` 选择它。

## 3. 默认配置与每次运行的配置

仓库中的默认 YAML 保持不变。启动器每次重新读取模板，只覆盖本次明确传入的参数，
用 TaskConfig 验证后，原子写入 `<output-dir>/runtime/task_config.yaml`，不会读取或继承上次 runtime。
运行配置保留传入的原始参数，包括 API key；框架另存的配置快照沿用现有序列化行为。

启动器固定写入以下值，并将数据集路径转换为 `/eval/data/<dataset-local-path>`：

```yaml
work_dir: /eval/output
no_timestamp: true
```

容器的入口是 `evalscope`，工作目录是 `/eval/data`。启动器执行的容器内命令为：

```bash
evalscope eval --config /eval/output/runtime/task_config.yaml \
  --status-file /eval/output/run_status.json
```

直接使用 CLI 时，`--config` 接受完整 YAML 或 JSON，`--status-file` 可省略；提供状态文件时必须同时提供配置文件。
`--config` 不能与 `--model`、`--datasets`、`--limit` 等评测 flags 混用，覆盖值应在生成配置时应用。
宿主启动器始终传入状态文件路径。

重复使用同一输出目录会替换 runtime、状态和部分结果；脚本不清理已有文件。
推荐为每个新任务分配新目录，并在任务结束后保留 runtime、日志和结果。

## 4. 挂载与媒体路径

| 宿主机位置 | 容器位置 | 用途 |
| --- | --- | --- |
| `--data-dir` | `/eval/data`，只读 | 数据 JSONL 和图片等媒体。 |
| `--output-dir` | `/eval/output`，可写 | 配置、状态、日志和结果。 |

媒体相对路径从 `/eval/data` 解析，不是从 JSONL 所在目录解析。例如仓库样例引用
`custom_eval/multimodal/images/dog.jpg`，因此 `--data-dir` 应指向仓库根，
不能只挂载 `custom_eval/multimodal/customer_v1`。自有数据也应保持 JSONL 中的媒体路径与挂载树一致，
不要把宿主机的 `D:/...` 绝对路径作为容器媒体路径。

`.dockerignore` 过滤本地 worktree、环境、缓存、构建产物和 `docs/reports`，保留 `custom_eval` 样例及媒体。
这些过滤仅影响镜像构建上下文，不会删除宿主机文件。运行时仍使用调用者指定的数据挂载。

## 5. 后端读取状态、报告与逐条结果

Docker 前台运行，stdout/stderr 直接留给调用方；后端可在启动进程时接收这些流。
输出目录采用固定层级，不额外嵌套时间戳：

```text
<output-dir>/
  runtime/task_config.yaml
  run_status.json
  logs/eval_log.log
  configs/...
  reports/<model_id>/customer_multimodal_v1.json
  reviews/<model_id>/customer_multimodal_v1_example.jsonl
  predictions/<model_id>/customer_multimodal_v1_example.jsonl
```

后端应先等待进程结束，再读取 UTF-8 `run_status.json`，按其中的路径数组定位本次产物，避免硬编码模型目录名。
`reports` 是汇总 JSON，`reviews` 是逐样本评分 JSONL，`predictions` 是模型输出 JSONL。
数组路径使用 `/`，相对于任务输出目录；只列本次新建或发生变化的结果，失败时可能为空或包含部分结果。
运行配置路径在正常容器任务中是 `runtime/task_config.yaml`；配置解析失败前无法确定输出根时也可能为容器绝对路径。

以下是成功状态的结构示例，时间和文件名仅供说明：

```json
{
  "status": "succeeded",
  "exit_code": 0,
  "started_at": "2026-10-09T02:00:00Z",
  "finished_at": "2026-10-09T02:00:05Z",
  "config_path": "runtime/task_config.yaml",
  "reports": ["reports/text_generation/customer_multimodal_v1.json"],
  "reviews": ["reviews/text_generation/customer_multimodal_v1_example.jsonl"],
  "predictions": ["predictions/text_generation/customer_multimodal_v1_example.jsonl"],
  "error": null
}
```

状态使用 UTC 时间，在成功结束或捕获评测异常后原子写入；没有 `running` 状态。
评测异常的状态为 `failed`，`exit_code` 非零，`error` 包含 `type` 和 `message`。

| 情况 | 进程结果与处理 |
| --- | --- |
| 正常完成 | 启动器返回 Docker 的退出码 0，状态为 `succeeded`。从报告和 reviews 读取模型得分。 |
| 配置、路径或 generation JSON 在宿主机校验失败 | argparse 退出码 2，不启动 Docker，不保证新状态文件；读取 stderr。 |
| 宿主机无法启动 Docker 可执行程序 | 启动器返回 1；输出目录可写时落盘宿主侧失败状态。 |
| Docker 返回非零，例如镜像或 daemon 问题 | 启动器透传该退出码；保留本次 CLI 写出的合法失败状态，否则写宿主侧失败状态。 |
| 评测抛出异常 | CLI 非零退出并尝试写失败状态；后端结合 error、日志和部分产物定位原因。 |

文件不可写、进程被强制终止等情况可能没有本次状态；不要把旧状态文件当作当前任务完成凭据。
退出码和状态描述执行情况，不表示模型合格。模型答错、返回无效业务 JSON，甚至请求错误被框架记录为样本结果时，
任务也可能完成；最终判断应结合汇总指标、逐条评分和日志。

## 6. 当前业务评分约束

每条 JSONL 记录需要非空 `id`、OpenAI 格式 `messages` 和非空对象 `expected`。
`expected` 的值为 JSON 标量，数字必须有限；当前支持严格标量比较、逐字段绝对数值容差、关键字段，
以及可选的 Draft 2020-12 JSON Schema 校验。

绝对容差写为 `"tolerance": {"count": {"absolute": 1}}`，引用数值型 expected 字段；
每个容差策略只能含 `absolute`，值须为非负有限数。`critical_fields` 是引用 expected 字段的非空、无重复列表。

以下记录顶层评分键尚不支持，出现即在样本转换阶段报错，即使值是 `false` 或 `null` 也不例外：

| 键 | 尚不支持的能力 |
| --- | --- |
| `field_weights` | 字段加权。 |
| `relative_tolerance` | 相对容差；`tolerance.<field>.relative` 也不支持。 |
| `fuzzy_matching` | 模糊匹配。 |
| `normalization` | 自动去空格、大小写转换等值归一化。 |

`expected` 中的 `field`、`overall` 为保留业务字段名。校验针对生成指标名的 canonical 形式，
因此 `Field`、`FIELD`、`field-`、`Overall` 等归一后与 `field_accuracy`、`overall_accuracy` 冲突的写法同样被拒绝。
这里的指标名称归一化不表示业务字符串会自动归一化后评分。其他未消费的普通数据字段不会仅因未使用而被拒绝。
