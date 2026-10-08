import json
from pathlib import Path

import pytest


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
