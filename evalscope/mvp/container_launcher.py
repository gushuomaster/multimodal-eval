import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, List, Mapping, Optional, Sequence

import yaml

from evalscope.cli.eval_status import EvalRunStatus, write_run_status
from evalscope.config import TaskConfig


def _validated_relative_path(value: str) -> PurePosixPath:
    normalized = value.replace('\\', '/')
    if (
        PureWindowsPath(value).drive
        or normalized.startswith('/')
        or any(part in {'', '.', '..'} for part in normalized.split('/'))
    ):
        raise ValueError('--dataset-local-path must be a nonempty relative path without empty, . or .. components')
    return PurePosixPath(normalized)


def _write_yaml_atomically(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='\n',
            prefix=f'.{path.name}.',
            suffix='.tmp',
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yaml.safe_dump(dict(config), temporary_file, allow_unicode=True, sort_keys=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_runtime_config(
    template_path: Path,
    output_dir: Path,
    dataset_local_path: str,
    overrides: Mapping[str, Any],
) -> Path:
    """Validate fresh template values and atomically write the raw container configuration."""
    runtime_path = output_dir / 'runtime' / 'task_config.yaml'
    if template_path.resolve() == runtime_path.resolve():
        raise ValueError(
            '--config-template must differ from the runtime target; choose another template or --output-dir'
        )
    config = yaml.safe_load(template_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('default task configuration must be a YAML mapping')
    config.update({key: value for key, value in overrides.items() if value is not None})
    dataset_args = config.get('dataset_args')
    if not isinstance(dataset_args, dict):
        raise ValueError('dataset_args must be a mapping')
    customer_args = dataset_args.get('customer_multimodal_v1')
    if not isinstance(customer_args, dict):
        raise ValueError('dataset_args.customer_multimodal_v1 must be a mapping')
    if not isinstance(config.get('generation_config', {}), dict):
        raise ValueError('generation_config must be a JSON object')

    container_dataset_path = PurePosixPath('/eval/data') / _validated_relative_path(dataset_local_path)
    customer_args['local_path'] = container_dataset_path.as_posix()
    config['work_dir'] = '/eval/output'
    config['no_timestamp'] = True
    # Validation may normalize values and mask secrets on serialization; persist the raw input instead.
    TaskConfig.model_validate(copy.deepcopy(config))
    _write_yaml_atomically(runtime_path, config)
    return runtime_path


def build_docker_command(image: str, data_dir: Path, output_dir: Path) -> List[str]:
    """Build the attached Docker invocation with deterministic input and output mounts."""
    return [
        'docker',
        'run',
        '--rm',
        '--volume',
        f'{data_dir.resolve()}:/eval/data:ro',
        '--volume',
        f'{output_dir.resolve()}:/eval/output',
        image,
        'eval',
        '--config',
        '/eval/output/runtime/task_config.yaml',
        '--status-file',
        '/eval/output/run_status.json',
    ]


def _generation_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('generation configuration must be a JSON object') from error
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError('generation configuration must be a JSON object')
    return config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _has_current_failure(status_path: Path, previous: Optional[os.stat_result]) -> bool:
    try:
        # The CLI replaces status atomically, even when two runs produce identical JSON.
        return (
            status_path.stat() != previous
            and EvalRunStatus.model_validate_json(status_path.read_bytes()).status == 'failed'
        )
    except (OSError, ValueError):
        return False


def _write_launch_failure(status_path: Path, started_at: str, exit_code: int, error: Exception) -> None:
    try:
        write_run_status(
            status_path,
            EvalRunStatus(
                status='failed',
                exit_code=exit_code,
                started_at=started_at,
                finished_at=_utc_now(),
                config_path='runtime/task_config.yaml',
                error={'type': type(error).__name__, 'message': str(error)},
            ),
        )
    except OSError as status_error:
        print(
            f'Could not write host failure status to {status_path} ({type(status_error).__name__}); '
            f'returning Docker exit code {exit_code}. Check the output directory permissions and available space.',
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate a runtime config, run Docker with attached logs, and return its exit code."""
    parser = argparse.ArgumentParser(description='Run the customer multimodal evaluation container.')
    parser.add_argument('--image', default='multimodal-eval:mvp')
    parser.add_argument('--config-template', type=Path, default=Path('configs/mvp/customer_multimodal_v1.yaml'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--dataset-local-path', default='custom_eval/multimodal/customer_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    for name in ('model', 'model-id', 'eval-type', 'api-url', 'api-key'):
        parser.add_argument(f'--{name}')
    parser.add_argument('--limit', type=float)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--generation-config', type=_generation_config)
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not args.config_template.is_file():
        parser.error('--config-template must be an existing YAML file')
    if not data_dir.is_dir():
        parser.error('--data-dir must be an existing directory')
    try:
        relative_path = _validated_relative_path(args.dataset_local_path)
        host_dataset = data_dir.joinpath(*relative_path.parts)
        if not (host_dataset.is_dir() or (host_dataset.is_file() and host_dataset.suffix.lower() == '.jsonl')):
            parser.error('--dataset-local-path must name an existing directory or JSONL file under --data-dir')
        overrides = {
            key: getattr(args, key)
            for key in (
                'model',
                'model_id',
                'eval_type',
                'api_url',
                'api_key',
                'limit',
                'eval_batch_size',
                'generation_config',
            )
        }
        build_runtime_config(args.config_template, output_dir, args.dataset_local_path, overrides)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    command = build_docker_command(args.image, data_dir, output_dir)
    status_path = output_dir / 'run_status.json'
    previous = status_path.stat() if status_path.is_file() else None
    started_at = _utc_now()
    try:
        completed = subprocess.run(command, check=False)
    except OSError as error:
        _write_launch_failure(status_path, started_at, 1, error)
        print(f'Docker launch failed: {error}', file=sys.stderr)
        return 1
    if completed.returncode != 0 and not _has_current_failure(status_path, previous):
        _write_launch_failure(
            status_path,
            started_at,
            completed.returncode,
            subprocess.CalledProcessError(completed.returncode, command),
        )
    return completed.returncode
