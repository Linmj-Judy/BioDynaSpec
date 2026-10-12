"""Make the repo importable the way the training entry point does.

`scripts/pretrain.sh` runs `src/trainer_far.py` with

    PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/src

so both `import src.*` (fully qualified) and `import openfold_local.*` /
`import model.*` / `import featureextractor` (top-level, relative to `src/`)
resolve. Tests should see the same two roots, otherwise modules that the real
run imports happily fail to import under pytest and tests silently skip.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

for _extra in (_REPO_ROOT, _REPO_ROOT / "src"):
    _p = str(_extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)
