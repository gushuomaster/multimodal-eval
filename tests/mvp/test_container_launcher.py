import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalscope.cli.eval_status import EvalRunStatus, write_run_status
from evalscope.config import TaskConfig
from evalscope.mvp.container_launcher import build_docker_command, build_runtime_config, main


def write_template(path: Path, **changes: Any) -> None:
    config = {
        'model': 'default-model',
        'model_id': None,
        'eval_type': 'mock_llm',
        'api_key': 'default-key',
        'datasets': ['customer_multimodal_v1'],
        'dataset_args': {'customer_multimodal_v1': {'local_path': '/old', 'subset_list': ['example']}},
        'limit': 2,
        'generation_config': {'temperature': 0.7, 'max_tokens': 99},
    }
    config.update(changes)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')


def read_runtime(output_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((output_dir / 'runtime' / 'task_config.yaml').read_text(encoding='utf-8'))


@pytest.fixture
def launch_paths(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    template = tmp_path / 'default.yaml'
    data_dir = tmp_path / 'data'
    output_dir = tmp_path / 'output'
    write_template(template)
    (data_dir / 'fixtures' / 'customer_v1').mkdir(parents=True)
    args = [
        '--config-template', str(template), '--data-dir', str(data_dir),
        '--output-dir', str(output_dir), '--dataset-local-path', 'fixtures/customer_v1',
    ]
    return template, data_dir, output_dir, args


def test_runtime_config_is_rebuilt_from_immutable_default(launch_paths: tuple) -> None:
    template, _, output_dir, _ = launch_paths
    original = template.read_bytes()
    runtime = build_runtime_config(template, output_dir, 'fixtures/customer_v1', {'model': 'first-model'})
    assert read_runtime(output_dir)['model'] == 'first-model'
    runtime.write_text('invalid: [old runtime', encoding='utf-8')

    build_runtime_config(template, output_dir, 'fixtures/customer_v1', {'model': None, 'api_url': 'https://api.test/v1'})

    second = read_runtime(output_dir)
    assert second['model'] == 'default-model'
    assert second['api_url'] == 'https://api.test/v1'
    assert second['work_dir'] == '/eval/output'
    assert second['no_timestamp'] is True
    assert second['dataset_args']['customer_multimodal_v1'] == {
        'local_path': '/eval/data/fixtures/customer_v1', 'subset_list': ['example'],
    }
    assert template.read_bytes() == original


def test_raw_keys_and_generation_replacement_survive_validation(launch_paths: tuple) -> None:
    template, _, output_dir, _ = launch_paths
    overrides = {'api_key': 'raw-测试-key', 'generation_config': {'temperature': 0}, 'limit': 0}
    build_runtime_config(template, output_dir, 'fixtures/customer_v1', overrides)
    config = read_runtime(output_dir)
    assert config['api_key'] == 'raw-测试-key'
    assert config['generation_config'] == {'temperature': 0}
    assert config['limit'] == 0
    assert overrides == {'api_key': 'raw-测试-key', 'generation_config': {'temperature': 0}, 'limit': 0}


@pytest.mark.parametrize(
    'changes, message',
    [
        ({'dataset_args': None}, 'dataset_args.*mapping'),
        ({'dataset_args': []}, 'dataset_args.*mapping'),
        ({'dataset_args': {}}, 'customer_multimodal_v1.*mapping'),
        ({'dataset_args': {'customer_multimodal_v1': []}}, 'customer_multimodal_v1.*mapping'),
        ({'generation_config': []}, 'generation_config.*object'),
        ({'generation_config': None}, 'generation_config.*object'),
        ({'limit': -1}, 'limit'),
        ({'generation_config': {'temperature': 'invalid'}}, 'temperature'),
        ({'unknown_task_option': True}, 'unknown_task_option'),
    ],
)
def test_invalid_config_is_rejected_before_persistence(launch_paths: tuple, changes: dict, message: str) -> None:
    template, _, output_dir, _ = launch_paths
    old_runtime = build_runtime_config(template, output_dir, 'fixtures/customer_v1', {})
    before = old_runtime.read_bytes()
    write_template(template, **changes)

    with pytest.raises(ValueError, match=message):
        build_runtime_config(template, output_dir, 'fixtures/customer_v1', {})

    assert old_runtime.read_bytes() == before


@pytest.mark.parametrize('payload', ['[]', 'null', 'model'])
def test_template_requires_yaml_mapping(tmp_path: Path, payload: str) -> None:
    template = tmp_path / 'default.yaml'
    template.write_text(payload, encoding='utf-8')
    with pytest.raises(ValueError, match='YAML mapping'):
        build_runtime_config(template, tmp_path / 'output', 'fixtures/customer_v1', {})
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize(
    'relative', ['', '.', '..', './data', 'data/../file', 'data/./file', '/absolute',
                 'C:/absolute', r'C:\absolute', 'C:relative', r'\\server\share', r'\root', 'data//file', 'data/'],
)
def test_rejects_paths_without_deterministic_child_mapping(launch_paths: tuple, relative: str) -> None:
    template, _, output_dir, _ = launch_paths
    with pytest.raises(ValueError, match='dataset-local-path'):
        build_runtime_config(template, output_dir, relative, {})
    assert not output_dir.exists()


@pytest.mark.parametrize('relative', ['样例/customer.jsonl', r'样例\customer.jsonl'])
def test_normalizes_dataset_slashes_without_losing_unicode(launch_paths: tuple, relative: str) -> None:
    template, _, output_dir, _ = launch_paths
    build_runtime_config(template, output_dir, relative, {})
    assert read_runtime(output_dir)['dataset_args']['customer_multimodal_v1']['local_path'] == (
        '/eval/data/样例/customer.jsonl'
    )


def test_atomic_replace_keeps_prior_runtime_on_failure(launch_paths: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    template, _, output_dir, _ = launch_paths
    runtime = build_runtime_config(template, output_dir, 'fixtures/customer_v1', {})
    original = runtime.read_bytes()

    def failed_replace(source: Path, destination: Path) -> None:
        assert Path(destination) == runtime
        assert runtime.read_bytes() == original
        assert yaml.safe_load(Path(source).read_text(encoding='utf-8'))['model'] == 'replacement'
        raise OSError('replace failed')

    monkeypatch.setattr('evalscope.mvp.container_launcher.os.replace', failed_replace)
    with pytest.raises(OSError, match='replace failed'):
        build_runtime_config(template, output_dir, 'fixtures/customer_v1', {'model': 'replacement'})
    assert runtime.read_bytes() == original
    assert list(runtime.parent.iterdir()) == [runtime]


def test_docker_command_mounts_data_read_only_and_output_writable(tmp_path: Path) -> None:
    data_dir = (tmp_path / 'data with spaces').resolve()
    output_dir = (tmp_path / 'output with spaces').resolve()
    assert build_docker_command('multimodal-eval:mvp', data_dir, output_dir) == [
        'docker', 'run', '--rm',
        '--volume', f'{data_dir}:/eval/data:ro',
        '--volume', f'{output_dir}:/eval/output',
        'multimodal-eval:mvp',
        'eval', '--config', '/eval/output/runtime/task_config.yaml',
        '--status-file', '/eval/output/run_status.json',
    ]


def test_main_applies_all_overrides_and_returns_attached_docker_exit(
    launch_paths: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, data_dir, output_dir, args = launch_paths

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        assert command[0:3] == ['docker', 'run', '--rm']
        assert f'{data_dir.resolve()}:/eval/data:ro' in command
        assert f'{output_dir.resolve()}:/eval/output' in command
        assert 'custom-image:tag' in command
        assert check is False
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)
    result = main(args + [
        '--image', 'custom-image:tag', '--model', 'custom-model', '--model-id', 'custom-id',
        '--eval-type', 'openai_api', '--api-url', 'https://api.test/v1', '--api-key', 'raw-mvp-key',
        '--limit', '0.5', '--eval-batch-size', '3', '--generation-config', '{"temperature": 0}',
    ])
    runtime = read_runtime(output_dir)
    assert {key: runtime[key] for key in (
        'model', 'model_id', 'eval_type', 'api_url', 'api_key', 'limit', 'eval_batch_size', 'generation_config',
    )} == {
        'model': 'custom-model', 'model_id': 'custom-id', 'eval_type': 'openai_api',
        'api_url': 'https://api.test/v1', 'api_key': 'raw-mvp-key', 'limit': 0.5,
        'eval_batch_size': 3, 'generation_config': {'temperature': 0},
    }
    assert result == 7


def test_main_defaults_support_jsonl_and_derive_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / 'data'
    dataset = data_dir / 'custom_eval' / 'multimodal' / 'customer_v1'
    dataset.mkdir(parents=True)
    output_dir = tmp_path / 'output'

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        assert 'multimodal-eval:mvp' in command
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)
    args = ['--data-dir', str(data_dir), '--output-dir', str(output_dir)]
    assert main(args) == 0
    defaults = read_runtime(output_dir)
    assert TaskConfig.model_validate(defaults).eval_type == 'mock_llm'
    assert defaults['model_id'] is None
    assert defaults['limit'] == 2
    assert defaults['eval_batch_size'] == 1
    assert defaults['generation_config'] == {}

    (data_dir / '样例.jsonl').write_text('{}\n', encoding='utf-8')
    assert main(args + ['--dataset-local-path', '样例.jsonl', '--model', 'changed-model', '--limit', '3']) == 0
    runtime = read_runtime(output_dir)
    assert runtime['model_id'] is None
    assert TaskConfig.model_validate(runtime).model_id == 'changed-model'
    assert runtime['dataset_args']['customer_multimodal_v1']['local_path'] == '/eval/data/样例.jsonl'
    assert TaskConfig.model_validate(runtime).limit == 3


@pytest.mark.parametrize('value', ['[]', 'null', '42', '"text"', '{'])
def test_main_rejects_invalid_generation_json(launch_paths: tuple, value: str, capsys: pytest.CaptureFixture) -> None:
    _, _, output_dir, args = launch_paths
    with pytest.raises(SystemExit) as error:
        main(args + ['--generation-config', value])
    assert error.value.code == 2
    assert 'generation configuration must be a JSON object' in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    'option, value, message',
    [
        ('--config-template', 'missing.yaml', 'config-template'),
        ('--data-dir', 'missing-data', 'data-dir'),
        ('--dataset-local-path', 'missing.jsonl', 'dataset-local-path'),
        ('--dataset-local-path', 'existing.txt', 'dataset-local-path'),
        ('--limit', '-1', 'limit'),
        ('--generation-config', '{"temperature": "invalid"}', 'temperature'),
    ],
)
def test_main_validates_before_docker(
    launch_paths: tuple, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    option: str, value: str, message: str,
) -> None:
    _, data_dir, _, args = launch_paths
    (data_dir / 'existing.txt').write_text('not a dataset', encoding='utf-8')

    def unexpected_run(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Docker must not run for invalid input')

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', unexpected_run)
    with pytest.raises(SystemExit) as error:
        main(args + [option, value])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def make_status(*, failed: bool = False) -> EvalRunStatus:
    return EvalRunStatus(
        status='failed' if failed else 'succeeded', exit_code=1 if failed else 0,
        started_at='2026-10-09T00:00:00Z', finished_at='2026-10-09T00:00:01Z',
        config_path='runtime/task_config.yaml', reports=['reports/current.json'],
        error={'type': 'ValueError', 'message': 'current evaluation failed'} if failed else None,
    )


@pytest.mark.parametrize('old_status', [None, 'success', 'failed', 'malformed'])
@pytest.mark.parametrize('failure', [7, 125, FileNotFoundError('docker missing')])
def test_failed_launch_replaces_stale_status_without_claiming_artifacts(
    launch_paths: tuple, monkeypatch: pytest.MonkeyPatch, old_status: str | None, failure: Any,
) -> None:
    _, _, output_dir, args = launch_paths
    (output_dir / 'reports').mkdir(parents=True)
    old_report = output_dir / 'reports' / 'old.json'
    old_report.write_text('{}', encoding='utf-8')
    status_file = output_dir / 'run_status.json'
    if old_status == 'malformed':
        status_file.write_text('{old invalid JSON', encoding='utf-8')
    elif old_status is not None:
        write_run_status(status_file, make_status(failed=old_status == 'failed'))

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        if isinstance(failure, OSError):
            raise failure
        return subprocess.CompletedProcess(command, failure)

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)
    exit_code = main(args)
    status = EvalRunStatus.model_validate_json(status_file.read_text(encoding='utf-8'))
    assert status.status == 'failed'
    assert status.exit_code == exit_code != 0
    if isinstance(failure, int):
        assert exit_code == failure
        assert str(failure) in status.error.message
    else:
        assert status.error.type == 'FileNotFoundError'
        assert status.error.message == 'docker missing'
    assert status.config_path == 'runtime/task_config.yaml'
    assert status.reports == status.reviews == status.predictions == []
    assert status.started_at != '2026-10-09T00:00:00Z'
    assert old_report.read_text(encoding='utf-8') == '{}'


@pytest.mark.parametrize('previous_failed, failed', [(True, False), (False, True), (True, True)])
def test_main_preserves_new_cli_status(
    launch_paths: tuple, monkeypatch: pytest.MonkeyPatch, previous_failed: bool, failed: bool,
) -> None:
    _, _, output_dir, args = launch_paths
    status_file = output_dir / 'run_status.json'
    write_run_status(status_file, make_status(failed=previous_failed))
    current_status = make_status(failed=failed)

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        write_run_status(status_file, current_status)
        return subprocess.CompletedProcess(command, 1 if failed else 0)

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)
    assert main(args) == (1 if failed else 0)
    assert json.loads(status_file.read_text(encoding='utf-8')) == current_status.model_dump(mode='json')


def test_nonzero_docker_exit_cannot_leave_new_success(launch_paths: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, output_dir, args = launch_paths
    status_file = output_dir / 'run_status.json'

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        write_run_status(status_file, make_status())
        return subprocess.CompletedProcess(command, 137)

    monkeypatch.setattr('evalscope.mvp.container_launcher.subprocess.run', fake_run)
    assert main(args) == 137
    status = EvalRunStatus.model_validate_json(status_file.read_text(encoding='utf-8'))
    assert status.status == 'failed'
    assert status.exit_code == 137
    assert status.reports == []
