"""Where scripts read caches and write outputs.

In ~/phase0 every script used `Path(__file__).parent`, i.e. the phase0
directory itself, and cluster jobs ran with that directory as the current
directory. Here that single directory is ~/multilingual_icm/data/uPRM_work_/runs
(outside git), and jobs `cd` into it. Override with the UPRM_RUN_DIR environment variable.
"""
import os
from pathlib import Path

RUN_DIR = Path(os.environ.get("UPRM_RUN_DIR", Path.home() / "multilingual_icm" / "data" / "uPRM_work_" / "runs"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
