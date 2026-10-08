# MVP Container Evaluation Design

Date: 2026-10-08

## 1. Goal

Complete the minimum runnable version of the multimodal evaluation platform in two phases:

1. Finish the existing `customer_multimodal_v1` structured business scoring contract.
2. Provide a configuration-driven Docker execution path that a backend service can invoke and observe.

The MVP proves this complete flow:

```text
backend parameters
  -> runtime task configuration
  -> Docker container
  -> EvalScope evaluation
  -> stdout/stderr logs
  -> mounted JSON reports and sample reviews
  -> machine-readable run status
```

The MVP does not include a frontend, a backend HTTP API, a task queue, concurrency isolation, permissions, secret
management, automatic task cleanup, model management, dataset management, comparison dashboards, or new evaluation
capabilities beyond the existing customer benchmark.

## 2. Design Decisions

### 2.1 Use a default configuration plus a generated runtime configuration

The repository provides a default YAML configuration for the customer multimodal task. The host launcher never edits
that default file. On every invocation it:

1. Reads the default configuration.
2. Applies only the values supplied by the caller.
3. Writes a complete runtime configuration under the caller-selected output directory.
4. Starts Docker with that runtime configuration.

The runtime configuration is regenerated from the default on every run. Previous overrides therefore cannot leak into
the next run, even though runtime configurations and output directories are retained indefinitely.

The MVP deliberately does not add concurrent-task isolation. A caller that reuses the same output directory accepts
that its existing runtime configuration and results may be replaced. The expected integration uses a distinct output
directory for each task.

### 2.2 Use the existing `TaskConfig` schema

The runtime YAML is a normal EvalScope `TaskConfig`, not a second platform-specific configuration schema. The container
adds file-based CLI execution:

```text
evalscope eval --config /eval/output/runtime/task_config.yaml \
  --status-file /eval/output/run_status.json
```

When `--config` is present, the CLI loads the YAML or JSON file through the existing `parse_task_config()` path. Normal
flag-based `evalscope eval` behavior remains unchanged. CLI evaluation flags are not merged on top of `--config`; host
overrides must be applied while creating the runtime configuration so there is only one source of truth inside the
container.

### 2.3 Keep the launcher small and portable

The host launcher is a Python script instead of a PowerShell-only or Bash-only script. It accepts the paths required to
run Docker plus the small set of values the backend is expected to change in the MVP:

- Docker image name.
- Default configuration path.
- Input dataset directory.
- Dataset JSONL path relative to the mounted input root.
- Output directory.
- Model name and optional report model ID.
- API URL and API key.
- Sample limit and evaluation batch size.
- Generation configuration as a JSON object.

Values not supplied by the caller come from the default YAML. The launcher serializes Docker arguments as a list and
runs Docker attached, so container stdout and stderr remain directly available to the invoking backend process.

The API key may be written to the retained runtime configuration for this MVP. No new access controls, cleanup, or
redaction behavior are added. Existing EvalScope masking behavior is left intact because removing it is unrelated to
making the execution path runnable.

## 3. Container Contract

### 3.1 Filesystem mounts

The launcher creates the output directory before starting Docker and uses these stable container paths:

| Host resource | Container path | Purpose |
| --- | --- | --- |
| Input dataset directory | `/eval/data` | JSONL data and referenced media |
| Output directory | `/eval/output` | Runtime config, logs, predictions, reviews, reports, status |

The generated runtime configuration always uses:

```yaml
work_dir: /eval/output
no_timestamp: true
```

The mounted input root is `/eval/data`. The launcher converts the caller's relative dataset path into a child such as
`/eval/data/custom_eval/multimodal/customer_v1` and stores that path in `dataset_args`. The input root is mounted
read-only; the output directory is writable. The container working directory is `/eval/data`, so media paths relative
to that mounted tree resolve consistently.

### 3.2 Container process

The image installs this repository as the `evalscope` package and uses Python 3.11. Its default executable is the
EvalScope CLI. The launcher runs the evaluation in the foreground and returns the same exit code as Docker.

The image only needs the dependencies required by the native OpenAI-compatible API path and
`customer_multimodal_v1`. Optional backends and the EvalScope web application are outside the MVP image.

### 3.3 Logs

EvalScope already emits progress and evaluation logs through the process streams and writes `logs/eval_log.log` under
the work directory. The container does not intercept or transform these streams. A backend can therefore forward the
Docker process output to the frontend while the retained log file remains available in the mounted output directory.

## 4. Run Status Contract

The CLI writes `/eval/output/run_status.json` whether evaluation succeeds or raises an exception after CLI startup. The
file is UTF-8 JSON with this shape:

```json
{
  "status": "succeeded",
  "exit_code": 0,
  "started_at": "2026-10-08T10:00:00Z",
  "finished_at": "2026-10-08T10:00:05Z",
  "config_path": "runtime/task_config.yaml",
  "reports": ["reports/model/customer_multimodal_v1.json"],
  "reviews": ["reviews/model/customer_multimodal_v1_example.jsonl"],
  "predictions": ["predictions/model/customer_multimodal_v1_example.jsonl"],
  "error": null
}
```

On failure, `status` is `failed`, `exit_code` is nonzero, discovered artifact lists contain any files that were written,
and `error` contains the exception type and message. Artifact paths are relative to the mounted output directory so the
host does not need to translate container paths.

If Docker itself cannot start, the host launcher returns Docker's nonzero exit code. It may write a host-side failed
status only when the output directory is available; the container cannot guarantee a status file before its process
starts.

The launcher and status writer use atomic replacement for their own YAML and JSON files so a caller does not read a
partially written document. No broader durability or locking guarantee is part of the MVP.

## 5. Structured Scoring Closure

The existing business comparator remains the scoring implementation. This phase closes three contract gaps without
adding new scoring algorithms.

### 5.1 Register fixed business metrics

The fixed aggregate metrics emitted by `customer_multimodal_v1` are declared in the canonical semantics catalog with
bounded 0-1, higher-is-better semantics. This includes the field-level mean, the compatibility overall accuracy, command
correctness, and schema validity metrics. Dynamic `<business_field>_accuracy` metrics remain field-specific report
details and continue through the existing resolver behavior.

Report tests verify that the fixed metrics resolve as quality metrics, render as percentages, and do not produce
undeclared-metric diagnostics.

### 5.2 Reject score-name collisions

The adapter rejects an expected business field named `field`, because it would emit `field_accuracy` and collide with
the fixed aggregate metric of the same name. The existing rejection of `overall` remains because it would collide with
`overall_accuracy`.

The validation happens while converting the dataset record, before model inference, and returns a clear error naming
the reserved field.

### 5.3 Reject documented unsupported scoring configuration

The current evaluation version supports only:

- strict scalar equality;
- per-field absolute numeric tolerance;
- critical fields;
- optional Draft 2020-12 JSON Schema validation.

Configuration that requests documented but unsupported behavior, including field weights, relative tolerance, fuzzy
matching, or automatic normalization, fails during record conversion instead of being ignored. Unrelated dataset fields
are not rejected merely because the adapter does not consume them.

## 6. Error Handling

- Invalid default or runtime YAML fails before Docker evaluation starts.
- Invalid override JSON fails in the host launcher before the default configuration is replaced by a runtime file.
- Missing input directories, missing configuration files, and unusable output paths fail before `docker run`.
- Invalid task configuration and dataset contracts produce a nonzero container exit code and a failed status file.
- Model API failures follow existing EvalScope behavior and remain visible in stdout/stderr, logs, reviews, and status.
- The launcher does not retry Docker or model requests. Retry policy belongs to a later platform phase.

## 7. Verification

### 7.1 Structured scoring

- Add failing-first tests for all fixed metric semantics.
- Add a dataset-contract test for the `field` collision.
- Add tests for each explicitly unsupported scoring option.
- Run the customer adapter, comparator, semantics, report, and native customer CLI tests.

### 7.2 Configuration CLI and status

- Test YAML and JSON config execution without Docker by using the mock model.
- Test that normal flag-based CLI execution is unchanged.
- Test successful and failed `run_status.json` content and relative artifact discovery.
- Test that runtime configuration generation always starts from the default and does not retain a previous override.
- Test launcher argument validation and Docker command construction without invoking a real daemon.

### 7.3 Image smoke test

- Build the MVP image.
- Run the bundled customer fixture through the image with `mock_llm`.
- Confirm live stdout/stderr, exit code 0, retained log, report JSON, review JSONL, prediction JSONL, runtime config, and
  successful status JSON in the mounted output directory.
- Run one deliberately invalid configuration and confirm a nonzero exit code and failed status JSON.

The local Docker client is installed, but the Docker Desktop Linux daemon was not running during design. Image build and
container smoke verification require the daemon to be available during implementation verification.

## 8. Deliverables

- Canonical metric registrations and customer dataset validation fixes.
- Targeted regression tests for the scoring closure.
- `evalscope eval --config` and status-file support.
- A default MVP task configuration.
- A portable host launcher.
- An MVP Dockerfile and Docker build context exclusions where needed.
- A concise operator guide with parameter descriptions, examples, mounts, outputs, and failure behavior.
- Automated unit/integration tests plus Docker smoke evidence when the local daemon is available.
