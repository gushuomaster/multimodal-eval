import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunError(BaseModel):
    """Describe the exception that caused an evaluation run to fail."""

    model_config = ConfigDict(extra='forbid')

    type: str
    message: str


class EvalRunStatus(BaseModel):
    """Record a completed evaluation run and its relative artifact paths."""

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
    """Return sorted POSIX paths to artifact files relative to the output directory."""
    patterns = {
        'reports': 'reports/**/*.json',
        'reviews': 'reviews/**/*.jsonl',
        'predictions': 'predictions/**/*.jsonl',
    }
    return {
        name: sorted(path.relative_to(output_dir).as_posix() for path in output_dir.glob(pattern) if path.is_file())
        for name, pattern in patterns.items()
    }


def write_run_status(path: Path, status: EvalRunStatus) -> None:
    """Atomically persist UTF-8 JSON, preserving existing status on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(status.model_dump(mode='json'), ensure_ascii=False, indent=2) + '\n'
    temporary_path: Optional[Path] = None
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
            temporary_file.write(payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
