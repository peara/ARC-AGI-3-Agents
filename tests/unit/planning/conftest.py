"""Conftest for planning sub-directory tests.

Adds the project root to sys.path so that top-level packages like
``planning`` and ``effects`` are importable from this sub-directory.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)