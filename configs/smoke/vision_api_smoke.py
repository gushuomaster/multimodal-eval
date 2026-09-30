import os
import sys
from typing import TYPE_CHECKING, Callable, Dict, List

if TYPE_CHECKING:
    from evalscope.config import TaskConfig


REQUIRED_ENV = ('EVAL_API_URL', 'EVAL_API_KEY', 'EVAL_MODEL')
MODES = ('single', 'multi')


def missing_environment() -> List[str]:
    """Return required vision smoke-test environment variables that are blank."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name, '').strip()]


def build_task_config() -> 'TaskConfig':
    """Build the selected local single-image or multi-image vision task."""
    from evalscope.config import TaskConfig

    mode = os.environ.get('VISION_SMOKE_MODE', 'single').strip().lower()
    if mode not in MODES:
        raise ValueError(f'VISION_SMOKE_MODE must be one of: {", ".join(MODES)}')

    if mode == 'single':
        dataset = 'general_vqa'
        dataset_args: Dict[str, Dict[str, object]] = {
            dataset: {
                'local_path': 'custom_eval/multimodal/vqa',
                'subset_list': ['example_openai'],
            }
        }
    else:
        dataset = 'general_vmcq'
        dataset_args = {
            dataset: {
                'local_path': 'custom_eval/multimodal/mcq',
                'subset_list': ['example'],
            }
        }

    return TaskConfig(
        model=os.environ['EVAL_MODEL'].strip(),
        eval_type='openai_api',
        api_url=os.environ['EVAL_API_URL'].strip(),
        api_key=os.environ['EVAL_API_KEY'].strip(),
        datasets=[dataset],
        dataset_args=dataset_args,
        limit=1,
        eval_batch_size=1,
        generation_config={
            'temperature': 0.0,
            'max_tokens': 256,
        },
        work_dir=f'outputs/smoke/vision_api/{mode}',
    )


def run_smoke(task_runner: Callable[['TaskConfig'], object]) -> int:
    """Validate external inputs and execute the selected vision task."""
    missing = missing_environment()
    if missing:
        names = ', '.join(missing)
        print(
            f'BLOCKED_EXTERNAL_INPUT: missing {names}; set them before running the vision API smoke test.',
            file=sys.stderr,
        )
        return 2

    try:
        task_runner(build_task_config())
    except ValueError as exc:
        print(f'INVALID_CONFIGURATION: {exc}', file=sys.stderr)
        return 2
    return 0


def main() -> int:
    """Run the vision smoke test through EvalScope's public entry point."""
    from evalscope import run_task

    return run_smoke(run_task)


if __name__ == '__main__':
    raise SystemExit(main())
