# Copyright (c) Alibaba, Inc. and its affiliates.
from argparse import Action, ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from evalscope.cli.base import ArgumentParserWithSubParsers, CLICommand
from evalscope.cli.eval_status import EvalRunStatus, discover_artifacts, write_run_status
from evalscope.config import TaskConfig, parse_task_config


class _EvaluationArgumentParser(ArgumentParser):
    def add_argument(self, *args: Any, **kwargs: Any) -> Action:
        # Track presence independently of values, including explicit defaults and booleans.
        action = kwargs.get('action', 'store')
        action_type = self._registry_get('action', action, action)

        class ExplicitArgument(action_type):
            def __call__(
                self, parser: ArgumentParser, namespace: Namespace, values: Any, option_string: Optional[str] = None
            ) -> None:
                explicit = getattr(namespace, '_explicit_eval_options', ())
                namespace._explicit_eval_options = (*explicit, option_string)
                super().__call__(parser, namespace, values, option_string)

        kwargs['action'] = ExplicitArgument
        return super().add_argument(*args, **kwargs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _relative_path(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _snapshot_artifacts(output_dir: Path) -> dict[Path, tuple[int, ...]]:
    # Include timestamp subdirectories before run_task chooses or reuses one.
    return {
        path.resolve(): _file_identity(path)
        for path in output_dir.rglob('*')
        if path.suffix in {'.json', '.jsonl'} and path.is_file()
    }


def _write_config_status(
    config_path: str,
    status_file: str,
    started_at: str,
    task_config: Optional[TaskConfig],
    before: Optional[dict[Path, tuple[int, ...]]],
    error: Optional[Exception] = None,
) -> None:
    artifacts = {'reports': [], 'reviews': [], 'predictions': []}
    reported_config_path = Path(config_path).resolve().as_posix()
    if task_config is not None:
        output_dir = Path(task_config.work_dir).resolve()
        reported_config_path = _relative_path(Path(config_path), output_dir)
        if before is not None:
            artifacts = {
                category: [
                    relative
                    for relative in paths
                    if before.get((output_dir / relative).resolve()) != _file_identity(output_dir / relative)
                ]
                for category, paths in discover_artifacts(output_dir).items()
            }
    status = EvalRunStatus(
        status='failed' if error is not None else 'succeeded',
        exit_code=1 if error is not None else 0,
        started_at=started_at,
        finished_at=_utc_now(),
        config_path=reported_config_path,
        error={'type': type(error).__name__, 'message': str(error)} if error is not None else None,
        **artifacts,
    )
    write_run_status(Path(status_file).resolve(), status)


def run_config_file(config_path: str, status_file: Optional[str] = None) -> None:
    """Run a complete YAML or JSON TaskConfig and optionally write an atomic status file."""
    from evalscope.run import run_task

    started_at = _utc_now()
    task_config = None
    before = None
    try:
        task_config = parse_task_config(config_path)
        if status_file is not None:
            before = _snapshot_artifacts(Path(task_config.work_dir))
            if task_config.use_cache:
                before.update(_snapshot_artifacts(Path(task_config.use_cache)))
        run_task(task_config)
    except Exception as error:
        if status_file is not None:
            _write_config_status(config_path, status_file, started_at, task_config, before, error)
        raise
    if status_file is not None:
        _write_config_status(config_path, status_file, started_at, task_config, before)


def subparser_func(args: Namespace) -> 'EvalCMD':
    """Function which will be called for a specific sub parser."""
    return EvalCMD(args)


class EvalCMD(CLICommand):
    name = 'eval'

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @staticmethod
    def define_args(parsers: ArgumentParserWithSubParsers) -> None:
        """define args for create pipeline template command."""
        from evalscope.arguments import add_argument

        evaluation_parser = _EvaluationArgumentParser(add_help=False)
        add_argument(evaluation_parser)
        parser = parsers.add_parser(EvalCMD.name, parents=[evaluation_parser])
        parser.add_argument('--config', type=str, help='Complete TaskConfig YAML or JSON file.')
        parser.add_argument('--status-file', type=str, help='Write a machine-readable run status JSON file.')
        parser.set_defaults(func=subparser_func)

    def execute(self) -> None:
        """Execute either a complete config file or the legacy evaluation flags."""
        from evalscope.run import run_task

        config_path = getattr(self.args, 'config', None)
        status_file = getattr(self.args, 'status_file', None)
        if status_file is not None and config_path is None:
            raise ValueError('--status-file requires --config')
        if config_path is not None:
            explicit = getattr(self.args, '_explicit_eval_options', ())
            if explicit:
                raise ValueError(f'--config cannot be combined with evaluation flags: {", ".join(explicit)}')
            run_config_file(config_path, status_file)
            return

        vars(self.args).pop('_explicit_eval_options', None)
        run_task(self.args)
