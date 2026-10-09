import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evalscope.mvp.container_launcher import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
