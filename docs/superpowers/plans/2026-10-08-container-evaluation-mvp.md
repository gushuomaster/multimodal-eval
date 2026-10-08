# Container Evaluation MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a configuration-driven Docker execution path that exposes live EvalScope logs and writes stable evaluation artifacts plus `run_status.json` to a caller-selected host directory.

**Architecture:** Extend the existing `evalscope eval` command to accept a complete YAML/JSON `TaskConfig` and an optional status-file path. A small Python host launcher regenerates a runtime config from a default template, mounts input and output directories, and runs an image whose entrypoint is the existing EvalScope CLI.

**Tech Stack:** Python 3.10+, argparse, Pydantic, PyYAML, subprocess, pytest, Docker with a Python 3.11 slim base image.

**Spec:** `docs/superpowers/specs/2026-10-08-mvp-container-evaluation-design.md`

## Global Constraints

- The runtime configuration is regenerated from the unchanged default template on every launch.
- The MVP does not add concurrency isolation, permissions, secret management, automatic cleanup, retries, or a backend HTTP API.
- Runtime configurations and evaluation outputs remain on disk.
- Existing EvalScope secret masking remains unchanged; do not add or remove redaction behavior.
- Docker runs in the foreground with inherited stdout and stderr.
- Container input root is `/eval/data`; container output root is `/eval/output`.
- The generated `TaskConfig` always uses `work_dir: /eval/output` and `no_timestamp: true`.
- Artifact paths in `run_status.json` are POSIX-style paths relative to the output root.
- Keep comments and docstrings in English, line width 120, LF endings, and UTF-8 without BOM.
- Preserve the pre-existing untracked `docs/reports/` directory and unrelated changes.

---

## File Structure

- Create `evalscope/cli/eval_status.py`: typed run-status contract, artifact discovery, and atomic JSON writing.
- Modify `evalscope/cli/start_eval.py`: add `--config` and `--status-file`, dispatch file-based execution, preserve flag mode.
- Create `tests/cli/test_eval_config.py`: config execution and status integration tests.
- Create `evalscope/mvp/__init__.py`: MVP launcher package marker.
- Create `evalscope/mvp/container_launcher.py`: runtime YAML generation, Docker command construction, argument parsing, attached execution.
- Create `scripts/mvp/run_container.py`: thin executable entrypoint for the launcher.
- Create `tests/mvp/test_container_launcher.py`: launcher unit tests without a Docker daemon.
- Create `configs/mvp/customer_multimodal_v1.yaml`: immutable default `TaskConfig` template.
- Create `docker/mvp/Dockerfile`: native API/customer benchmark image.
- Create `.dockerignore`: exclude VCS, worktrees, caches, and existing outputs from the build context.
- Create `docs/mvp-container.md`: Chinese operator contract and examples.
- Create `tests/mvp/test_container_assets.py`: static contract tests for template and Dockerfile.

### Task 1: Define and Persist the Run Status Contract

**Files:**
- Create: `evalscope/cli/eval_status.py`
- Create: `tests/cli/test_eval_config.py`

**Interfaces:**
- Produces: `RunError`, `EvalRunStatus`, `discover_artifacts(output_dir: Path) -> Dict[str, List[str]]`, and `write_run_status(path: Path, status: EvalRunStatus) -> None`.
- Consumed by: Task 2 file-based CLI execution.

- [ ] **Step 1: Write failing status serialization and discovery tests**

Create `tests/cli/test_eval_config.py` with:

```python
import json
from pathlib import Path

from evalscope.cli.eval_status import EvalRunStatus, discover_artifacts, write_run_status


def test_discover_artifacts_returns_sorted_relative_paths(tmp_path: Path) -> None:
    for relative_path in [
        'reports/model/customer_multimodal_v1.json',
        'reviews/model/customer_multimodal_v1_example.jsonl',
        'predictions/model/customer_multimodal_v1_example.jsonl',
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')

    artifacts = discover_artifacts(tmp_path)

    assert artifacts == {
        'reports': ['reports/model/customer_multimodal_v1.json'],
        'reviews': ['reviews/model/customer_multimodal_v1_example.jsonl'],
        'predictions': ['predictions/model/customer_multimodal_v1_example.jsonl'],
    }


def test_write_run_status_atomically_persists_utf8_json(tmp_path: Path) -> None:
    status_path = tmp_path / 'run_status.json'
    status = EvalRunStatus(
        status='failed',
        exit_code=1,
        started_at='2026-10-08T02:00:00Z',
        finished_at='2026-10-08T02:00:01Z',
        config_path='runtime/task_config.yaml',
        error={'type': 'ValueError', 'message': '配置无效'},
    )

    write_run_status(status_path, status)

    assert json.loads(status_path.read_text(encoding='utf-8'))['error']['message'] == '配置无效'
    assert list(tmp_path.glob('.run_status.json.*.tmp')) == []
```

- [ ] **Step 2: Run the tests and verify the module is missing**

```powershell
pytest tests/cli/test_eval_config.py -v
```

Expected: collection ERROR with `ModuleNotFoundError: evalscope.cli.eval_status`.

- [ ] **Step 3: Implement the typed status contract and atomic writer**

Create `evalscope/cli/eval_status.py` with these public shapes:

```python
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunError(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: str
    message: str


class EvalRunStatus(BaseModel):
    model_config = ConfigDict(extra='forbid')

    status: Literal['succeeded', 'failed']
    exit_code: int
    started_at: str
    finished_at: str
    config_path: str
    reports: List[str] = Field(default_factory=list)
    reviews: List[str] = Field(default_factory=list)
    predictions: List[str] = Field(default_factory=list)
    error: Optional[RunError] = None


def discover_artifacts(output_dir: Path) -> Dict[str, List[str]]:
    patterns = {
        'reports': 'reports/**/*.json',
        'reviews': 'reviews/**/*.jsonl',
        'predictions': 'predictions/**/*.jsonl',
    }
    return {
        name: sorted(path.relative_to(output_dir).as_posix() for path in output_dir.glob(pattern))
        for name, pattern in patterns.items()
    }


def write_run_status(path: Path, status: EvalRunStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(status.model_dump(mode='json'), ensure_ascii=False, indent=2) + '\n'
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        newline='\n',
        prefix=f'.{path.name}.',
        suffix='.tmp',
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        temporary_file.write(payload)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)
```

Add public docstrings to both models and functions before running lint.

- [ ] **Step 4: Run the status tests**

```powershell
pytest tests/cli/test_eval_config.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit the status contract**

```powershell
git add -- evalscope/cli/eval_status.py tests/cli/test_eval_config.py
git commit -m "feat: add evaluation run status contract"
```

### Task 2: Execute Complete Task Config Files from the CLI

**Files:**
- Modify: `evalscope/cli/start_eval.py`
- Modify: `tests/cli/test_eval_config.py`

**Interfaces:**
- Consumes: `run_task(TaskConfig)`, `parse_task_config(path: str)`, and Task 1 status helpers.
- Produces: `run_config_file(config_path: str, status_file: str) -> None`, CLI options `--config` and `--status-file`.

- [ ] **Step 1: Add failing success, failure, and legacy dispatch tests**

Append tests that create a minimal YAML config and call `EvalCMD.execute()`:

```python
from argparse import Namespace

import pytest
import yaml

from evalscope.cli.start_eval import EvalCMD


def write_mock_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        yaml.safe_dump({
            'model': 'text_generation',
            'eval_type': 'mock_llm',
            'datasets': ['customer_multimodal_v1'],
            'dataset_args': {
                'customer_multimodal_v1': {
                    'local_path': 'custom_eval/multimodal/customer_v1',
                    'subset_list': ['example'],
                },
            },
            'limit': 2,
            'eval_batch_size': 1,
            'work_dir': str(output_dir),
            'no_timestamp': True,
        }, sort_keys=False),
        encoding='utf-8',
    )


def test_eval_command_runs_yaml_config_and_writes_success_status(tmp_path: Path) -> None:
    config_path = tmp_path / 'runtime' / 'task_config.yaml'
    config_path.parent.mkdir()
    output_dir = tmp_path / 'output'
    status_path = output_dir / 'run_status.json'
    write_mock_config(config_path, output_dir)

    EvalCMD(Namespace(config=str(config_path), status_file=str(status_path))).execute()

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'succeeded'
    assert status['exit_code'] == 0
    assert status['reports'] == ['reports/text_generation/customer_multimodal_v1.json']
    assert status['reviews'] == ['reviews/text_generation/customer_multimodal_v1_example.jsonl']
    assert status['predictions'] == ['predictions/text_generation/customer_multimodal_v1_example.jsonl']


def test_eval_command_writes_failed_status_and_reraises(tmp_path: Path) -> None:
    config_path = tmp_path / 'broken.yaml'
    status_path = tmp_path / 'output' / 'run_status.json'
    config_path.write_text('datasets: [customer_multimodal_v1]\n', encoding='utf-8')

    with pytest.raises(Exception):
        EvalCMD(Namespace(config=str(config_path), status_file=str(status_path))).execute()

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['exit_code'] == 1
    assert status['error']['type']
    assert status['error']['message']


def test_eval_command_without_config_preserves_namespace_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    args = Namespace(config=None, status_file=None, model='mock')
    monkeypatch.setattr('evalscope.run.run_task', lambda task: captured.setdefault('task', task))

    EvalCMD(args).execute()

    assert captured['task'] is args
```

- [ ] **Step 2: Run the new tests and verify file mode is not implemented**

```powershell
pytest tests/cli/test_eval_config.py -v
```

Expected: file-mode tests FAIL because `EvalCMD.execute()` still passes the `Namespace` directly to `run_task`.

- [ ] **Step 3: Implement `--config`, `--status-file`, and status-aware execution**

In `EvalCMD.define_args()`, add before `add_argument(parser)`:

```python
        parser.add_argument('--config', type=str, help='Complete TaskConfig YAML or JSON file.')
        parser.add_argument('--status-file', type=str, help='Write a machine-readable run status JSON file.')
```

Add an internal UTC formatter and public execution helper to `start_eval.py`:

```python
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def run_config_file(config_path: str, status_file: str) -> None:
    started_at = _utc_now()
    status_path = Path(status_file).resolve()
    output_dir = status_path.parent
    task_config = None
    try:
        task_config = parse_task_config(config_path)
        run_task(task_config)
        artifacts = discover_artifacts(Path(task_config.work_dir))
        status = EvalRunStatus(
            status='succeeded',
            exit_code=0,
            started_at=started_at,
            finished_at=_utc_now(),
            config_path=_relative_path(Path(config_path), output_dir),
            **artifacts,
        )
    except Exception as error:
        artifacts = discover_artifacts(Path(task_config.work_dir)) if task_config is not None else discover_artifacts(output_dir)
        status = EvalRunStatus(
            status='failed',
            exit_code=1,
            started_at=started_at,
            finished_at=_utc_now(),
            config_path=_relative_path(Path(config_path), output_dir),
            error={'type': type(error).__name__, 'message': str(error)},
            **artifacts,
        )
        write_run_status(status_path, status)
        raise
    write_run_status(status_path, status)
```

Implement `_relative_path(path: Path, output_dir: Path) -> str` using `Path.resolve().relative_to()` and fall back to
the resolved POSIX path when the config is outside the output root. In `execute()`, require `--status-file` with
`--config`, reject `--status-file` without `--config`, call `run_config_file()` for file mode, and keep the existing
`run_task(self.args)` branch unchanged for flag mode.

- [ ] **Step 4: Run CLI config and existing CLI argument tests**

```powershell
pytest tests/cli/test_eval_config.py tests/api/test_task_config_validation.py tests/cli/test_type_hints.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit file-based CLI execution**

```powershell
git add -- evalscope/cli/start_eval.py tests/cli/test_eval_config.py
git commit -m "feat: run evaluations from config files"
```

### Task 3: Generate Runtime Configurations from an Immutable Default

**Files:**
- Create: `evalscope/mvp/__init__.py`
- Create: `evalscope/mvp/container_launcher.py`
- Create: `scripts/mvp/run_container.py`
- Create: `configs/mvp/customer_multimodal_v1.yaml`
- Create: `tests/mvp/test_container_launcher.py`

**Interfaces:**
- Produces: `build_runtime_config(template_path: Path, output_dir: Path, dataset_local_path: str, overrides: Mapping[str, Any]) -> Path`.
- Produces: `build_docker_command(image: str, data_dir: Path, output_dir: Path) -> List[str]`.
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int`.
- Consumed by: Task 4 Docker runtime and operator guide.

- [ ] **Step 1: Write failing runtime config regeneration tests**

Create `tests/mvp/test_container_launcher.py` with:

```python
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalscope.mvp.container_launcher import build_docker_command, build_runtime_config, main


def write_template(path: Path) -> None:
    path.write_text(
        yaml.safe_dump({
            'model': 'default-model',
            'eval_type': 'mock_llm',
            'api_key': 'default-key',
            'datasets': ['customer_multimodal_v1'],
            'dataset_args': {'customer_multimodal_v1': {'local_path': '/old', 'subset_list': ['example']}},
            'limit': 2,
        }, sort_keys=False),
        encoding='utf-8',
    )


def test_runtime_config_is_rebuilt_from_default_on_every_run(tmp_path: Path) -> None:
    template = tmp_path / 'default.yaml'
    output_dir = tmp_path / 'output'
    write_template(template)

    runtime_path = build_runtime_config(template, output_dir, 'fixtures/customer_v1', {'model': 'first-model'})
    first = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))
    runtime_path = build_runtime_config(template, output_dir, 'fixtures/customer_v1', {'api_url': 'https://api.test/v1'})
    second = yaml.safe_load(runtime_path.read_text(encoding='utf-8'))

    assert first['model'] == 'first-model'
    assert second['model'] == 'default-model'
    assert second['api_url'] == 'https://api.test/v1'
    assert second['work_dir'] == '/eval/output'
    assert second['no_timestamp'] is True
    assert second['dataset_args']['customer_multimodal_v1']['local_path'] == '/eval/data/fixtures/customer_v1'


def test_docker_command_mounts_data_read_only_and_output_writable(tmp_path: Path) -> None:
    data_dir = (tmp_path / 'data').resolve()
    output_dir = (tmp_path / 'output').resolve()

    command = build_docker_command('multimodal-eval:mvp', data_dir, output_dir)

    assert command == [
        'docker', 'run', '--rm',
        '--volume', f'{data_dir}:/eval/data:ro',
        '--volume', f'{output_dir}:/eval/output',
        'multimodal-eval:mvp',
        'eval', '--config', '/eval/output/runtime/task_config.yaml',
        '--status-file', '/eval/output/run_status.json',
    ]
```

- [ ] **Step 2: Run the tests and verify the launcher package is missing**

```powershell
pytest tests/mvp/test_container_launcher.py -v
```

Expected: collection ERROR with `ModuleNotFoundError: evalscope.mvp`.

- [ ] **Step 3: Implement atomic runtime config generation**

Create `evalscope/mvp/container_launcher.py` with typed functions. `build_runtime_config()` must:

```python
def build_runtime_config(
    template_path: Path,
    output_dir: Path,
    dataset_local_path: str,
    overrides: Mapping[str, Any],
) -> Path:
    config = yaml.safe_load(template_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('default task configuration must be a YAML mapping')
    config.update({key: value for key, value in overrides.items() if value is not None})
    container_dataset_path = PurePosixPath('/eval/data') / _validated_relative_path(dataset_local_path)
    config['dataset_args']['customer_multimodal_v1']['local_path'] = container_dataset_path.as_posix()
    config['work_dir'] = '/eval/output'
    config['no_timestamp'] = True
    runtime_path = output_dir / 'runtime' / 'task_config.yaml'
    _write_yaml_atomically(runtime_path, config)
    return runtime_path
```

`_validated_relative_path()` rejects absolute paths, `.` and `..` segments, and an empty value because they would not map
to a deterministic child of `/eval/data`. Implement the writer directly:

```python
def _write_yaml_atomically(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        newline='\n',
        prefix=f'.{path.name}.',
        suffix='.tmp',
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        yaml.safe_dump(dict(config), temporary_file, allow_unicode=True, sort_keys=False)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)
```

- [ ] **Step 4: Implement Docker command construction and the launcher CLI**

`build_docker_command()` returns exactly the list asserted above. `main()` parses:

```text
--image
--config-template
--data-dir
--dataset-local-path
--output-dir
--model
--model-id
--eval-type
--api-url
--api-key
--limit
--eval-batch-size
--generation-config
```

`--data-dir` and `--output-dir` are required. Defaults are:

```text
--image multimodal-eval:mvp
--config-template configs/mvp/customer_multimodal_v1.yaml
--dataset-local-path custom_eval/multimodal/customer_v1
```

Parse `--generation-config` with `json.loads()` and require a JSON object. Validate the template and data directories,
create the output directory, build the runtime config, then call `subprocess.run(command, check=False)` without capture
so logs remain attached. Return `completed.returncode`.

Create `scripts/mvp/run_container.py` as a thin entrypoint:

```python
from evalscope.mvp.container_launcher import main


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 5: Create the immutable default customer task configuration**

Create `configs/mvp/customer_multimodal_v1.yaml`:

```yaml
model: text_generation
model_id: text_generation
eval_type: mock_llm
api_url: null
api_key: EMPTY
datasets:
  - customer_multimodal_v1
dataset_args:
  customer_multimodal_v1:
    local_path: /eval/data/custom_eval/multimodal/customer_v1
    subset_list:
      - example
limit: 2
eval_batch_size: 1
generation_config: {}
work_dir: /eval/output
no_timestamp: true
```

- [ ] **Step 6: Add argument and attached-execution tests**

Append this attached-execution test to `tests/mvp/test_container_launcher.py`:

```python
def test_main_writes_overrides_and_returns_docker_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / 'default.yaml'
    data_dir = tmp_path / 'data'
    output_dir = tmp_path / 'output'
    data_dir.mkdir()
    write_template(template)
    captured = {}

    def fake_run(command: list[str], check: bool) -> Any:
        captured['command'] = command
        captured['check'] = check
        return type('Completed', (), {'returncode': 7})()

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)

    exit_code = main([
        '--config-template', str(template),
        '--data-dir', str(data_dir),
        '--dataset-local-path', 'fixtures/customer_v1',
        '--output-dir', str(output_dir),
        '--api-key', 'raw-mvp-key',
    ])

    runtime = yaml.safe_load((output_dir / 'runtime' / 'task_config.yaml').read_text(encoding='utf-8'))
    assert runtime['api_key'] == 'raw-mvp-key'
    assert isinstance(captured['command'], list)
    assert captured['check'] is False
    assert exit_code == 7
```

Add a second test calling `main()` with `--generation-config []` and assert `SystemExit` plus an error message stating that
generation configuration must be a JSON object.

- [ ] **Step 7: Run all launcher tests**

```powershell
pytest tests/mvp/test_container_launcher.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit the launcher and default configuration**

```powershell
git add -- evalscope/mvp/__init__.py evalscope/mvp/container_launcher.py scripts/mvp/run_container.py configs/mvp/customer_multimodal_v1.yaml tests/mvp/test_container_launcher.py
git commit -m "feat: add MVP container launcher"
```

### Task 4: Package the MVP Image and Document the Contract

**Files:**
- Create: `docker/mvp/Dockerfile`
- Create: `.dockerignore`
- Create: `docs/mvp-container.md`
- Create: `tests/mvp/test_container_assets.py`

**Interfaces:**
- Consumes: Task 2 CLI and Task 3 launcher paths.
- Produces: image `multimodal-eval:mvp` with `ENTRYPOINT ["evalscope"]` and a documented backend invocation contract.

- [ ] **Step 1: Add failing static asset contract tests**

Create `tests/mvp/test_container_assets.py`:

```python
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_mvp_dockerfile_uses_python_311_and_evalscope_entrypoint() -> None:
    dockerfile = (ROOT / 'docker' / 'mvp' / 'Dockerfile').read_text(encoding='utf-8')

    assert 'FROM python:3.11-slim' in dockerfile
    assert 'PYTHONUNBUFFERED=1' in dockerfile
    assert 'pip install --no-cache-dir .' in dockerfile
    assert 'WORKDIR /eval/data' in dockerfile
    assert 'ENTRYPOINT ["evalscope"]' in dockerfile


def test_default_mvp_config_matches_container_mount_contract() -> None:
    config = yaml.safe_load((ROOT / 'configs' / 'mvp' / 'customer_multimodal_v1.yaml').read_text(encoding='utf-8'))

    assert config['work_dir'] == '/eval/output'
    assert config['no_timestamp'] is True
    assert config['dataset_args']['customer_multimodal_v1']['local_path'].startswith('/eval/data/')
```

- [ ] **Step 2: Run the static asset tests and verify Dockerfile is missing**

```powershell
pytest tests/mvp/test_container_assets.py -v
```

Expected: Dockerfile test FAILS with `FileNotFoundError`; config contract test PASS.

- [ ] **Step 3: Create the MVP Dockerfile**

Create `docker/mvp/Dockerfile`:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/evalscope
COPY . .
RUN python -m pip install --no-cache-dir .

WORKDIR /eval/data
ENTRYPOINT ["evalscope"]
CMD ["--help"]
```

- [ ] **Step 4: Create root build-context exclusions**

Create `.dockerignore` with:

```text
.git
.worktrees
.pytest_cache
.ruff_cache
**/__pycache__
**/*.pyc
docs/_build
evalscope/web/node_modules
evalscope/web/dist
outputs
```

Do not exclude `custom_eval`, because it is used as the repository fixture source during smoke verification.

- [ ] **Step 5: Write the Chinese operator guide**

Create `docs/mvp-container.md` covering these exact commands:

```powershell
docker build -f docker/mvp/Dockerfile -t multimodal-eval:mvp .

python scripts/mvp/run_container.py `
  --data-dir D:\project\agent_eval\multimodal-eval `
  --dataset-local-path custom_eval/multimodal/customer_v1 `
  --output-dir D:\project\agent_eval\multimodal-eval\outputs\mvp\demo
```

Add the real API example with `--eval-type openai_api`, `--model`, `--model-id`, `--api-url`, and `--api-key`. Explain that
the data mount is the root used to resolve relative media URLs, the runtime YAML is rebuilt on every invocation, outputs
are retained, logs stay attached, and the backend should read `run_status.json` plus its relative artifact paths.

- [ ] **Step 6: Run asset and launcher tests**

```powershell
pytest tests/mvp/test_container_assets.py tests/mvp/test_container_launcher.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit image assets and documentation**

```powershell
git add -- docker/mvp/Dockerfile .dockerignore docs/mvp-container.md tests/mvp/test_container_assets.py
git commit -m "build: add MVP evaluation image"
```

### Task 5: Verify the Image End to End

**Files:**
- Modify only if verification exposes a defect in the files introduced by Tasks 1-4.
- Write retained evidence under the project artifact parent, not the repository root.

**Interfaces:**
- Consumes: all container MVP deliverables.
- Produces: mock success evidence and invalid-config failure evidence from a real Docker container.

- [ ] **Step 1: Verify the Docker daemon before building**

```powershell
docker version --format '{{json .Server}}'
```

Expected: exit code 0 and a Linux server description. If the daemon is unavailable, report this external blocker and
continue with all non-Docker tests; do not claim the image smoke test passed.

- [ ] **Step 2: Create the unified artifact directory**

Use PowerShell with explicit absolute paths:

```powershell
$mvpArtifacts = 'D:\project\agent_eval\multimodal-eval-artifacts\artifacts\mvp-container'
New-Item -ItemType Directory -Force -Path $mvpArtifacts | Out-Null
```

This path follows the project auxiliary-directory rule and is not deleted automatically.

- [ ] **Step 3: Build the image**

```powershell
docker build -f docker/mvp/Dockerfile -t multimodal-eval:mvp . 2>&1 | Tee-Object -FilePath "$mvpArtifacts\docker-build.log"
```

Expected: exit code 0 and image `multimodal-eval:mvp` present.

- [ ] **Step 4: Run the bundled mock evaluation through the host launcher**

```powershell
$smokeOutput = "$mvpArtifacts\mock-success"
python scripts/mvp/run_container.py --data-dir (Get-Location).Path --dataset-local-path custom_eval/multimodal/customer_v1 --output-dir $smokeOutput
```

Expected: exit code 0 with progress logs printed live.

- [ ] **Step 5: Verify all mounted success artifacts**

```powershell
$status = Get-Content -LiteralPath "$smokeOutput\run_status.json" -Encoding UTF8 | ConvertFrom-Json
$status.status
$status.exit_code
$status.reports
$status.reviews
$status.predictions
Get-Item -LiteralPath "$smokeOutput\logs\eval_log.log"
Get-Item -LiteralPath "$smokeOutput\runtime\task_config.yaml"
```

Expected: `succeeded`, exit code `0`, one customer report, one review JSONL, one prediction JSONL, and both retained files.

- [ ] **Step 6: Run an invalid runtime config through the image**

Copy the successful runtime YAML to `$mvpArtifacts\invalid\runtime\task_config.yaml`, replace `datasets` with
`[missing_mvp_benchmark]`, mount the repository root as `/eval/data`, mount `$mvpArtifacts\invalid` as `/eval/output`, and
run:

```powershell
docker run --rm --volume "${PWD}:/eval/data:ro" --volume "$mvpArtifacts\invalid:/eval/output" multimodal-eval:mvp eval --config /eval/output/runtime/task_config.yaml --status-file /eval/output/run_status.json
```

Expected: nonzero exit code and a retained status with `status: failed`, `exit_code: 1`, and a nonempty error.

- [ ] **Step 7: Run all targeted non-Docker tests**

```powershell
pytest tests/cli/test_eval_config.py tests/mvp/test_container_launcher.py tests/mvp/test_container_assets.py tests/cli/test_customer_multimodal_v1.py -v
```

Expected: PASS.

- [ ] **Step 8: Run submission checks**

```powershell
pre-commit run --files evalscope/cli/eval_status.py evalscope/cli/start_eval.py evalscope/mvp/__init__.py evalscope/mvp/container_launcher.py scripts/mvp/run_container.py tests/cli/test_eval_config.py tests/mvp/test_container_launcher.py tests/mvp/test_container_assets.py
make lint-imports
pytest tests/cli/test_all.py::TestRun::test_ci_lite -v -s -p no:warnings
git diff --check
git status --short
```

Expected: all checks PASS; only the pre-existing untracked `docs/reports/` directory remains.

- [ ] **Step 9: Commit any verification-only fixes**

If Tasks 5.1-5.8 required an in-scope fix, stage only those exact files, rerun the affected verification, and commit:

```powershell
git commit -m "fix: close MVP container verification gaps"
```

If no fixes were needed, do not create an empty commit.
