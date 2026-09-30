import os
import sys
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from evalscope.config import TaskConfig


REQUIRED_ENV = ('EVAL_API_URL', 'EVAL_API_KEY', 'EVAL_MODEL')


def missing_environment() -> List[str]:
    """Return required smoke-test environment variables that are unset or blank."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name, '').strip()]


def build_task_config() -> 'TaskConfig':
    """Build the Native EvalScope task used by the text API smoke test."""
    from evalscope.config import TaskConfig

    return TaskConfig(
        model=os.environ['EVAL_MODEL'].strip(),
        eval_type='openai_api',
        api_url=os.environ['EVAL_API_URL'].strip(),
        api_key=os.environ['EVAL_API_KEY'].strip(),
        datasets=['gsm8k'],
        limit=3,
        eval_batch_size=1,
        generation_config={
            'temperature': 0.0,
            'max_tokens': 512,
        },
        work_dir='outputs/smoke/text_api',
    )


def run_smoke(task_runner: Callable[['TaskConfig'], object]) -> int:
    """Validate external inputs and execute the configured EvalScope task."""
    missing = missing_environment()
    if missing:
        names = ', '.join(missing)
        print(
            f'BLOCKED_EXTERNAL_INPUT: missing {names}; set them before running the text API smoke test.',
            file=sys.stderr,
        )
        return 2

    task_runner(build_task_config())
    return 0


def main() -> int:
    """Run the text API smoke test through EvalScope's public entry point."""
    from evalscope import run_task

    return run_smoke(run_task)


if __name__ == '__main__':
    raise SystemExit(main())
