"""Put the artifact root on ``sys.path`` so ``benchlib`` imports from a clone, uninstalled.

The experiments are directories of scripts, not packages, and there is no install step: a reader
clones the repository and runs ``reproduce.sh``. pytest puts each test file's own directory on the
path, which is what lets a test import the sibling script it exercises; this adds the one directory
above the experiments, which is what lets both import the shared engine.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
