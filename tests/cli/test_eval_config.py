import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from evalscope.cli.start_eval import EvalCMD
from evalscope.config import TaskConfig


def _eval_args(*arguments: str) -> Namespace:
    parser = ArgumentParser()
    EvalCMD.define_args(parser.add_subparsers())
    return parser.parse_args(['eval', *arguments])


def _execute_eval(*arguments: str) -> None:
    args = _eval_args(*arguments)
    args.func(args).execute()


def _write_mock_config(path: Path, output_dir: Path, **overrides: object) -> None:
    config = {
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
        'api_key': 'task2-secret-do-not-display',
        **overrides,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config) if path.suffix == '.json' else yaml.safe_dump(config, sort_keys=False)
    path.write_text(payload, encoding='utf-8')


@pytest.mark.parametrize('extension', ['yaml', 'json'])
@pytest.mark.parametrize('with_status', [True, False])
def test_eval_command_runs_config_file(tmp_path: Path, extension: str, with_status: bool) -> None:
    output_dir = tmp_path / 'output'
    config_path = output_dir / f'runtime/task_config.{extension}'
    status_path = tmp_path / 'external/run_status.json'
    _write_mock_config(config_path, output_dir)

    _execute_eval('--config', str(config_path), *(['--status-file', str(status_path)] if with_status else []))

    report_path = output_dir / 'reports/text_generation/customer_multimodal_v1.json'
    assert json.loads(report_path.read_text(encoding='utf-8'))['dataset_name'] == 'customer_multimodal_v1'
    assert 'task2-secret-do-not-display' not in (output_dir / 'logs/eval_log.log').read_text(encoding='utf-8')
    assert status_path.exists() is with_status
    if with_status:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        assert status['status'] == 'succeeded'
        assert status['exit_code'] == 0
        assert status['error'] is None
        assert status['config_path'] == f'runtime/task_config.{extension}'
        assert status['reports'] == ['reports/text_generation/customer_multimodal_v1.json']
        assert status['reviews'] == ['reviews/text_generation/customer_multimodal_v1_example.jsonl']
        assert status['predictions'] == ['predictions/text_generation/customer_multimodal_v1_example.jsonl']
        assert status['started_at'].endswith('Z') and status['finished_at'].endswith('Z')


def test_eval_command_tracks_actual_timestamp_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / 'output'
    actual_dir = output_dir / '20261008_120000'
    historical = actual_dir / 'reports/old/previous.json'
    historical.parent.mkdir(parents=True)
    historical.write_text('{}', encoding='utf-8')
    config_path = tmp_path / 'config.yaml'
    status_path = tmp_path / 'external/status.json'
    _write_mock_config(config_path, output_dir, no_timestamp=False)
    monkeypatch.setattr('evalscope.run.current_time', lambda: datetime(2026, 10, 8, 12))

    _execute_eval('--config', str(config_path), '--status-file', str(status_path))

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['config_path'] == config_path.resolve().as_posix()
    assert status['reports'] == ['reports/text_generation/customer_multimodal_v1.json']
    assert status['reviews'] == ['reviews/text_generation/customer_multimodal_v1_example.jsonl']
    assert status['predictions'] == ['predictions/text_generation/customer_multimodal_v1_example.jsonl']
    assert (actual_dir / status['reports'][0]).is_file()


@pytest.mark.parametrize('failure', ['invalid-type', 'malformed-yaml', 'missing-file', 'unknown-benchmark'])
def test_eval_command_failed_status_excludes_history(tmp_path: Path, failure: str) -> None:
    config_path = tmp_path / 'broken.yaml'
    output_dir = tmp_path / 'output'
    status_path = output_dir / 'run_status.json'
    for category, extension in [('reports', 'json'), ('reviews', 'jsonl'), ('predictions', 'jsonl')]:
        historical = output_dir / category / f'old/previous.{extension}'
        historical.parent.mkdir(parents=True)
        historical.write_text('{}', encoding='utf-8')
    status_path.write_text('{"status":"succeeded"}', encoding='utf-8')
    if failure == 'invalid-type':
        _write_mock_config(config_path, output_dir, eval_batch_size='invalid')
    elif failure == 'malformed-yaml':
        config_path.write_text('datasets: [', encoding='utf-8')
    elif failure == 'unknown-benchmark':
        _write_mock_config(config_path, output_dir, datasets=['task2_nonexistent_benchmark'])

    with pytest.raises(Exception):
        _execute_eval('--config', str(config_path), '--status-file', str(status_path))

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['exit_code'] == 1
    assert status['error']['type'] and status['error']['message']
    assert status['reports'] == status['reviews'] == status['predictions'] == []


@pytest.mark.parametrize('use_cache', [True, False])
def test_eval_command_failure_before_directory_setup_excludes_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_cache: bool
) -> None:
    output_dir = tmp_path / 'output'
    cache_dir = tmp_path / 'cache'
    config_path = tmp_path / 'config.yaml'
    status_path = tmp_path / 'external/status.json'
    _write_mock_config(config_path, output_dir, no_timestamp=False, use_cache=str(cache_dir) if use_cache else None)
    for root in [output_dir, cache_dir]:
        historical = root / 'reports/old/previous.json'
        historical.parent.mkdir(parents=True)
        historical.write_text('{}', encoding='utf-8')

    def fail_seed(seed: int) -> None:
        raise RuntimeError('Failed before output directory setup')

    monkeypatch.setattr('evalscope.run.seed_everything', fail_seed)
    with pytest.raises(RuntimeError, match='Failed before output directory setup'):
        _execute_eval('--config', str(config_path), '--status-file', str(status_path))

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['exit_code'] == 1
    assert status['error']['type'] == 'RuntimeError'
    assert status['reports'] == status['reviews'] == status['predictions'] == []


@pytest.mark.parametrize('use_cache', [True, False])
@pytest.mark.parametrize('fails', [True, False])
def test_eval_command_lists_only_created_or_changed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_cache: bool, fails: bool
) -> None:
    output_dir = tmp_path / 'output'
    actual_dir = tmp_path / 'cache' if use_cache else output_dir
    config_path = tmp_path / 'config.json'
    status_path = tmp_path / 'external/status.json'
    _write_mock_config(config_path, output_dir, use_cache=str(actual_dir) if use_cache else None)
    report_dir = actual_dir / 'reports/model'
    report_dir.mkdir(parents=True)
    (report_dir / 'old.json').write_text('{}', encoding='utf-8')
    (report_dir / 'changed.json').write_text('{}', encoding='utf-8')

    def run_with_partial_artifacts(task: TaskConfig) -> None:
        assert isinstance(task, TaskConfig)
        task.work_dir = str(actual_dir)
        (report_dir / 'changed.json').write_text('{"changed":true}', encoding='utf-8')
        (report_dir / 'created.json').write_text('{}', encoding='utf-8')
        if fails:
            raise RuntimeError('Partial run failed')

    monkeypatch.setattr('evalscope.run.run_task', run_with_partial_artifacts)
    if fails:
        with pytest.raises(RuntimeError, match='Partial run failed'):
            _execute_eval('--config', str(config_path), '--status-file', str(status_path))
    else:
        _execute_eval('--config', str(config_path), '--status-file', str(status_path))

    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == ('failed' if fails else 'succeeded')
    assert status['reports'] == ['reports/model/changed.json', 'reports/model/created.json']


@pytest.mark.parametrize(
    'flags',
    [
        ['--model', 'mock'],
        ['--seed', '42'],
        ['--repeats=1'],
        ['--dataset-args', '{}'],
        ['--debug'],
        ['--collect-perf'],
        ['--no-collect-perf'],
        ['--generation-config', 'temperature=0'],
    ],
)
@pytest.mark.parametrize('config_first', [True, False])
def test_eval_command_rejects_config_mixed_with_explicit_flags(flags: list[str], config_first: bool) -> None:
    config = ['--config', 'does-not-need-to-exist.yaml']
    with pytest.raises((ValueError, SystemExit), match='--config'):
        _execute_eval(*(config + flags if config_first else flags + config))


def test_eval_command_rejects_status_without_config() -> None:
    with pytest.raises((ValueError, SystemExit), match='--status-file'):
        _execute_eval('--status-file', 'unused-status.json')


@pytest.mark.parametrize('flags', [[], ['--model', 'mock', '--no-collect-perf', '--generation-config', 'temperature=0']])
def test_eval_command_preserves_legacy_namespace_dispatch(monkeypatch: pytest.MonkeyPatch, flags: list[str]) -> None:
    from evalscope.arguments import add_argument

    legacy_parser = ArgumentParser()
    add_argument(legacy_parser)
    expected = vars(legacy_parser.parse_args(flags))
    args = _eval_args(*flags)
    for name, value in expected.items():
        assert getattr(args, name) == value
    captured = []
    monkeypatch.setattr('evalscope.run.run_task', captured.append)

    args.func(args).execute()

    assert len(captured) == 1 and captured[0] is args


def test_eval_command_runs_legacy_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / 'output'
    _execute_eval(
        '--model',
        'text_generation',
        '--eval-type',
        'mock_llm',
        '--datasets',
        'customer_multimodal_v1',
        '--dataset-args',
        json.dumps({
            'customer_multimodal_v1': {
                'local_path': 'custom_eval/multimodal/customer_v1',
                'subset_list': ['example'],
            }
        }),
        '--limit',
        '1',
        '--work-dir',
        str(output_dir),
        '--no-timestamp',
        '--no-collect-perf',
    )

    report_path = output_dir / 'reports/text_generation/customer_multimodal_v1.json'
    assert json.loads(report_path.read_text(encoding='utf-8'))['dataset_name'] == 'customer_multimodal_v1'


def test_eval_command_config_without_model_uses_framework_default(tmp_path: Path) -> None:
    output_dir = tmp_path / 'output'
    config_path = tmp_path / 'default-model.yaml'
    _write_mock_config(config_path, output_dir)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    del config['model']
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    status_path = tmp_path / 'status.json'

    _execute_eval('--config', str(config_path), '--status-file', str(status_path))

    assert json.loads(status_path.read_text(encoding='utf-8'))['status'] == 'succeeded'


def test_discover_artifacts_returns_sorted_relative_paths(tmp_path: Path) -> None:
    from evalscope.cli.eval_status import discover_artifacts

    for relative_path in [
        'reports/model/z.json',
        'reports/model/a.json',
        'reports/model/not-json.txt',
        'reviews/model/customer_multimodal_v1_example.jsonl',
        'predictions/model/customer_multimodal_v1_example.jsonl',
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')

    (tmp_path / 'reports/model/subdir.json').mkdir()

    artifacts = discover_artifacts(tmp_path)

    assert artifacts == {
        'reports': ['reports/model/a.json', 'reports/model/z.json'],
        'reviews': ['reviews/model/customer_multimodal_v1_example.jsonl'],
        'predictions': ['predictions/model/customer_multimodal_v1_example.jsonl'],
    }


def test_write_run_status_atomically_persists_utf8_json(tmp_path: Path) -> None:
    from evalscope.cli.eval_status import EvalRunStatus, write_run_status

    status_path = tmp_path / 'nested' / 'run_status.json'
    status = EvalRunStatus(
        status='failed',
        exit_code=1,
        started_at='2026-10-08T02:00:00Z',
        finished_at='2026-10-08T02:00:01Z',
        config_path='runtime/task_config.yaml',
        error={'type': 'ValueError', 'message': '配置无效'},
    )

    write_run_status(status_path, status)

    payload = status_path.read_bytes()
    assert '配置无效'.encode('utf-8') in payload
    assert payload.endswith(b'\n') and b'\r\n' not in payload
    assert json.loads(payload) == {
        'status': 'failed',
        'exit_code': 1,
        'started_at': '2026-10-08T02:00:00Z',
        'finished_at': '2026-10-08T02:00:01Z',
        'config_path': 'runtime/task_config.yaml',
        'reports': [],
        'reviews': [],
        'predictions': [],
        'error': {'type': 'ValueError', 'message': '配置无效'},
    }
    status.status = 'succeeded'
    status.exit_code = 0
    status.error = None
    write_run_status(status_path, status)
    assert json.loads(status_path.read_text(encoding='utf-8'))['status'] == 'succeeded'
    assert list(status_path.parent.iterdir()) == [status_path]


def test_discover_artifacts_missing_directory_returns_empty_lists(tmp_path: Path) -> None:
    from evalscope.cli.eval_status import discover_artifacts

    assert discover_artifacts(tmp_path / 'missing') == {'reports': [], 'reviews': [], 'predictions': []}


@pytest.mark.parametrize('failure', ['replace', 'write'])
def test_write_run_status_failure_preserves_existing_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from evalscope.cli import eval_status

    status_path = tmp_path / 'run_status.json'
    original = b'{"status": "succeeded"}\n'
    status_path.write_bytes(original)
    status = eval_status.EvalRunStatus(
        status='failed',
        exit_code=1,
        started_at='2026-10-08T02:00:00Z',
        finished_at='2026-10-08T02:00:01Z',
        config_path='runtime/task_config.yaml',
        error={'type': 'ValueError', 'message': '\ud800' if failure == 'write' else '配置无效'},
    )

    def fail_replace(source: Path, destination: Path) -> None:
        assert Path(source).parent == destination.parent
        assert json.loads(Path(source).read_text(encoding='utf-8'))['status'] == 'failed'
        assert destination.read_bytes() == original
        raise PermissionError('Replacement denied')

    if failure == 'replace':
        monkeypatch.setattr(eval_status.os, 'replace', fail_replace)

    with pytest.raises(PermissionError if failure == 'replace' else UnicodeEncodeError):
        eval_status.write_run_status(status_path, status)

    assert status_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [status_path]


@pytest.mark.parametrize(
    'overrides',
    [
        {'status': 'running'},
        {'unexpected': True},
        {'error': {'type': 'ValueError', 'message': 'Invalid', 'unexpected': True}},
    ],
)
def test_run_status_rejects_invalid_contract_fields(overrides: dict) -> None:
    from pydantic import ValidationError

    from evalscope.cli.eval_status import EvalRunStatus

    fields = {
        'status': 'succeeded',
        'exit_code': 0,
        'started_at': '2026-10-08T02:00:00Z',
        'finished_at': '2026-10-08T02:00:01Z',
        'config_path': 'runtime/task_config.yaml',
    }
    with pytest.raises(ValidationError):
        EvalRunStatus(**(fields | overrides))


@pytest.mark.parametrize(
    'overrides',
    [
        {'exit_code': 0, 'error': {'type': 'ValueError', 'message': 'Invalid'}},
        {},
        {'error': None},
    ],
)
def test_failed_run_status_requires_nonzero_exit_and_error(overrides: dict) -> None:
    from pydantic import ValidationError

    from evalscope.cli.eval_status import EvalRunStatus

    fields = {
        'status': 'failed',
        'exit_code': 1,
        'started_at': '2026-10-08T02:00:00Z',
        'finished_at': '2026-10-08T02:00:01Z',
        'config_path': 'runtime/task_config.yaml',
    }
    with pytest.raises(ValidationError):
        EvalRunStatus(**(fields | overrides))


@pytest.mark.parametrize('exit_code', [1, -1])
@pytest.mark.parametrize('message', ['Invalid', str(ValueError())])
def test_failed_run_status_accepts_nonzero_exit_and_error(exit_code: int, message: str) -> None:
    from evalscope.cli.eval_status import EvalRunStatus, RunError

    error = RunError(type='ValueError', message=message)
    status = EvalRunStatus(
        status='failed',
        exit_code=exit_code,
        started_at='2026-10-08T02:00:00Z',
        finished_at='2026-10-08T02:00:01Z',
        config_path='runtime/task_config.yaml',
        error=error,
    )

    assert status.exit_code == exit_code
    assert status.error == error
