from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TEST_DIR = Path(__file__).resolve().parent
__path__ = [str(_TEST_DIR), str(_PROJECT_ROOT / "replay")]
