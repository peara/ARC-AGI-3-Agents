import os
import shutil

import pytest
from arcengine import FrameData, GameState


def make_mock_llm():
    """Return a mock LLM callable that rejects all grouping proposals.

    Use this in tests that need an EntityBuilder/CombinedEngine but don't
    care about compound formation. It returns a valid JSON verdict list
    with a single reject verdict for proposal_id 0.
    """
    import json

    def _mock(messages):
        return json.dumps([
            {
                "proposal_id": 0,
                "verdict": "reject",
                "relation": "none",
                "members": [],
                "reason": "mock reject",
            }
        ])

    return _mock


def make_mock_combined_engine():
    """Return a CombinedEngine whose update() returns no confirmed groups.

    Equivalent to the old classical-only fallback where no compounds form.
    The heuristic engine is bypassed entirely to avoid crashes on minimal
    test registries with empty shape_keys.
    """
    from grouping.combined_engine import CombinedEngine

    engine = CombinedEngine(llm_call=make_mock_llm())
    engine.update = lambda *args, **kwargs: []
    return engine


def make_confirming_combined_engine():
    """Return a CombinedEngine that confirms all merge proposals via LLM.

    Use in tests that need compounds to actually form. The LLM mock confirms
    every proposal as a merge group. Heuristics still run normally, but the
    LLM adjudication always confirms. Uses a low readiness threshold so
    co-movement fires with minimal observations.
    """
    import json

    from grouping.combined_engine import CombinedEngine
    from grouping.readiness import ReadinessConfig

    call_count = [0]

    def _confirming_llm(messages):
        call_count[0] += 1
        return json.dumps([
            {
                "proposal_id": 0,
                "verdict": "confirm",
                "relation": "merge",
                "members": [],
                "reason": "test confirm",
            }
        ])

    return CombinedEngine(
        llm_call=_confirming_llm,
        config=ReadinessConfig(co_movement_min_actions=1),
    )


@pytest.fixture
def mock_combined_engine():
    return make_mock_combined_engine()


def get_test_recordings_dir():
    conftest_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(conftest_dir, "recordings")


@pytest.fixture(scope="session", autouse=True)
def clean_test_recordings():
    test_recordings_dir = get_test_recordings_dir()

    os.environ["RECORDINGS_DIR"] = test_recordings_dir

    if os.path.exists(test_recordings_dir):
        shutil.rmtree(test_recordings_dir)
    os.makedirs(test_recordings_dir, exist_ok=True)

    yield test_recordings_dir


@pytest.fixture
def temp_recordings_dir(clean_test_recordings):
    test_recordings_dir = get_test_recordings_dir()

    os.makedirs(test_recordings_dir, exist_ok=True)

    original_dir = os.environ.get("RECORDINGS_DIR")
    os.environ["RECORDINGS_DIR"] = test_recordings_dir

    yield test_recordings_dir

    if original_dir:
        os.environ["RECORDINGS_DIR"] = original_dir
    else:
        os.environ.pop("RECORDINGS_DIR", None)


@pytest.fixture
def sample_frame():
    return FrameData(
        game_id="test-game",
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        levels_completed=5,
    )


@pytest.fixture
def use_env_vars(monkeypatch):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("ARC_API_KEY"):
        monkeypatch.setenv("ARC_API_KEY", "test-key")
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    if not os.environ.get("SCHEME"):
        monkeypatch.setenv("SCHEME", "https")
    if not os.environ.get("HOST"):
        monkeypatch.setenv("HOST", "three.arcprize.org")
    if not os.environ.get("PORT"):
        monkeypatch.setenv("PORT", "443")
