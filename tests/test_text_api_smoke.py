import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from evalscope.config import TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / 'configs' / 'smoke' / 'text_api_smoke.py'


def load_smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('text_api_smoke', SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load smoke module from {SMOKE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTextApiSmoke(unittest.TestCase):
    def test_missing_api_environment_is_reported_as_external_input_blocker(self) -> None:
        env = os.environ.copy()
        for name in ('EVAL_API_URL', 'EVAL_API_KEY', 'EVAL_MODEL'):
            env.pop(name, None)
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, str(SMOKE_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, '')
        self.assertEqual(
            result.stderr,
            'BLOCKED_EXTERNAL_INPUT: missing EVAL_API_URL, EVAL_API_KEY, EVAL_MODEL; '
            'set them before running the text API smoke test.\n',
        )

    def test_build_task_config_uses_environment_for_native_text_smoke(self) -> None:
        module = load_smoke_module()
        self.assertTrue(callable(getattr(module, 'build_task_config', None)), 'build_task_config is not implemented')
        env = {
            'EVAL_API_URL': 'https://example.invalid/v1/chat/completions',
            'EVAL_API_KEY': 'test-secret',
            'EVAL_MODEL': 'test-model',
        }

        with patch.dict(os.environ, env, clear=False):
            task_config = module.build_task_config()

        self.assertIsInstance(task_config, TaskConfig)
        self.assertEqual(task_config.model, 'test-model')
        self.assertEqual(task_config.model_id, 'test-model')
        self.assertEqual(task_config.eval_backend, 'Native')
        self.assertEqual(task_config.eval_type, 'openai_api')
        self.assertEqual(task_config.api_url, 'https://example.invalid/v1/chat/completions')
        self.assertEqual(task_config.api_key.get_secret_value(), 'test-secret')
        self.assertEqual(task_config.datasets, ['gsm8k'])
        self.assertEqual(task_config.limit, 3)
        self.assertEqual(task_config.eval_batch_size, 1)
        self.assertEqual(task_config.generation_config.temperature, 0.0)
        self.assertEqual(task_config.generation_config.max_tokens, 512)
        self.assertEqual(task_config.work_dir, 'outputs/smoke/text_api')

    def test_run_smoke_executes_the_configured_task_once(self) -> None:
        module = load_smoke_module()
        self.assertTrue(callable(getattr(module, 'run_smoke', None)), 'run_smoke is not implemented')
        received = []

        def task_runner(task_config: TaskConfig) -> dict:
            received.append(task_config)
            return {}

        env = {
            'EVAL_API_URL': 'https://example.invalid/v1/chat/completions',
            'EVAL_API_KEY': 'test-secret',
            'EVAL_MODEL': 'test-model',
        }
        with patch.dict(os.environ, env, clear=False):
            exit_code = module.run_smoke(task_runner)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], TaskConfig)
        self.assertEqual(received[0].model, 'test-model')


if __name__ == '__main__':
    unittest.main()
