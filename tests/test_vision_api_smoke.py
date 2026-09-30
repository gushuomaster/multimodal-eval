import importlib.util
import os
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from evalscope.config import TaskConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / 'configs' / 'smoke' / 'vision_api_smoke.py'


def load_smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('vision_api_smoke', SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load smoke module from {SMOKE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVisionApiSmoke(unittest.TestCase):
    def test_single_mode_builds_general_vqa_task(self) -> None:
        module = load_smoke_module()
        env = {
            'EVAL_API_URL': 'https://example.invalid/v1',
            'EVAL_API_KEY': 'test-secret',
            'EVAL_MODEL': 'MiniMax-M3',
            'VISION_SMOKE_MODE': 'single',
        }

        with patch.dict(os.environ, env, clear=False):
            task_config = module.build_task_config()

        self.assertIsInstance(task_config, TaskConfig)
        self.assertEqual(task_config.model, 'MiniMax-M3')
        self.assertEqual(task_config.eval_type, 'openai_api')
        self.assertEqual(task_config.datasets, ['general_vqa'])
        self.assertEqual(task_config.dataset_args['general_vqa']['local_path'], 'custom_eval/multimodal/vqa')
        self.assertEqual(task_config.dataset_args['general_vqa']['subset_list'], ['example_openai'])
        self.assertEqual(task_config.limit, 1)
        self.assertEqual(task_config.work_dir, 'outputs/smoke/vision_api/single')

    def test_multi_mode_builds_general_vmcq_task(self) -> None:
        module = load_smoke_module()
        env = {
            'EVAL_API_URL': 'https://example.invalid/v1',
            'EVAL_API_KEY': 'test-secret',
            'EVAL_MODEL': 'MiniMax-M3',
            'VISION_SMOKE_MODE': 'multi',
        }

        with patch.dict(os.environ, env, clear=False):
            task_config = module.build_task_config()

        self.assertEqual(task_config.datasets, ['general_vmcq'])
        self.assertEqual(task_config.dataset_args['general_vmcq']['local_path'], 'custom_eval/multimodal/mcq')
        self.assertEqual(task_config.dataset_args['general_vmcq']['subset_list'], ['example'])
        self.assertEqual(task_config.limit, 1)
        self.assertEqual(task_config.work_dir, 'outputs/smoke/vision_api/multi')


if __name__ == '__main__':
    unittest.main()
