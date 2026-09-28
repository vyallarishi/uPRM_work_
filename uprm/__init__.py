"""uPRM: step-level scoring of reasoning chains (see README.md).

This package imports the ICM repository (`icm`). If `icm` is not installed
(`pip install -e ../ICM_work_`), the sibling checkout next to this repo is
used, so everything runs from a plain clone of the three repositories side
by side.
"""
import sys as _sys
from pathlib import Path as _Path

for _pkg, _folder in (("icm", "ICM_work_"),):
    try:
        __import__(_pkg)
    except ImportError:
        _sibling = _Path(__file__).resolve().parents[2] / _folder
        if _sibling.is_dir():
            _sys.path.append(str(_sibling))
