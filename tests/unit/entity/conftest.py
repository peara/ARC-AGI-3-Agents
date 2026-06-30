"""Conftest for entity ID stability tests.

Adds the project root to sys.path so that top-level packages like
``entity`` and ``perception`` are importable from this sub-directory.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so that ``entity`` resolves
# to the top-level package, not this test directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)