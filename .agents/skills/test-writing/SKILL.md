---
name: test-writing
description: "House test-writing conventions for the ARC-AGI-3-Agents repo. Covers pytest placement, real-path testing over mocking, incident-anchored regressions, contract pinning with inspect.getsource(), failure-list discipline, house gates, and fixture hygiene. Use when the user asks to write tests, add tests for a module, refactor tests, build a test plan, run pytest, or grow the test suite. Triggers: 'write tests', 'add tests for', 'refactor tests', 'test plan', 'pytest', 'test suite'."
---

# Test Writing — House Conventions

Guidelines for authoring pytest-based unit tests in this repo. All conventions
are grounded in existing test files, not aspirational rules.

## 1. Placement

Tests live under `tests/unit/<agent>/` with a folder-level `conftest.py` for
shared factories. Cross-cutting modules (planning, effects, entity) use flat
`tests/unit/` when they have no agent-specific sub-directory.

Exemplars:
- `tests/unit/simulator_agent/conftest.py` — shared sandbox factories
  (`make_seeded_live_sandbox`, `make_plain_sandbox`, `make_win_callback`).
- `tests/unit/langgraph_vision_agent/conftest.py` — `make_frame`, `make_grid`,
  `_mock_services` for vision-node tests.
- `tests/unit/entity/conftest.py` — sys.path bootstrap only; no `__init__.py`
  in any test folder (pytest path-based discovery works without it).

`pytest.ini` sets `python_files = test_*.py`, so renaming a test file needs no
config change.

## 2. Test real paths, not mocks-of-the-thing-under-test

Avoid mocking the method you are trying to verify. Instead, construct the object
via `__new__` + explicit attribute seeding when `__init__` needs network or LLM
credentials.

Exemplars:
- `tests/unit/simulator_agent/conftest.py:82-128` — `make_seeded_live_sandbox`
  seeds every attribute `__init__` would set, including `_exec_budget`, then
  builds the real namespace so `action()` closure is bound to the callback.
- `tests/unit/simulator_agent/conftest.py:26-35` — `make_mock_step_env_callback`
  returns a dict state (grid, valid_actions, last_action_result) that the real
  sandbox consumes.
- `tests/unit/simulator_agent/test_agent.py:1002-1063` — `_make_budget_exhausted_agent`
  injects `agent._llm_chat = fake_llm_chat` to script LLM responses and drives
  the full `agent.run()` path.
- `tests/unit/simulator_agent/test_simulate_diagnosis.py:79-132` — `test_predict_diff_sets_regions`
   exercises the real `predict_and_compare()` on a real `SimulatorSandbox`.

## 3. Incident-anchored regression tests

Every bugfix pairs with a regression test. The docstring cites the run GUID and
root cause. The test name states the failure mode.

Exemplar:
- `tests/unit/simulator_agent/test_sandbox_safety.py:43-60` — `test_ls20_incident_pattern_recovers`.
  Anchor: 2026-09-04 run `fd3925f7`. The LLM's `while ar>8:` loop over a
  non-converging `simulate()` hung `run_code()` forever. The test reproduces
  that exact pattern and asserts the tracer budget fires.

## 4. No tautologies

Assert production behavior, not mocks echoing themselves. A test that mocks
`predict_and_compare` to return a fixed dict, then asserts the dict equals that
fixed dict, teaches nothing.

Precedent: commit message pattern `test(simulator): rewrite N tautological tests
to exercise production paths`. When you see a test doing nothing but mock
verification, replace it with a real-object construction from section 2.

## 5. inspect.getsource() contract tests

Use `inspect.getsource()` to pin cross-method contracts that are invisible at
runtime (budget guards, no-random-fallback policies, frame-index derivation).

Exemplars:
- `tests/unit/simulator_agent/test_agent.py:953-994` — `TestActionBudget` joins
  `getsource(run)` + `getsource(_run_tool_loop)` and asserts
  `"action_counter >= self.MAX_ACTIONS"` appears in the combined source.
- `tests/unit/simulator_agent/test_frame_index.py:45-52` — `test_no_frame_index_attribute_remains`
  inspects `__init__` and `run` source to prove `_frame_index` was removed.

Maintenance trap: when you split a method whose source is pinned, the
`getsource()` scope must widen in the same commit. This broke twice during the
2026-09-04 refactor. Keep these tests next to the contract they guard.

## 6. Failure-list baseline discipline

The repo currently carries 25 documented pre-existing failures (see
`/tmp/opencode/failures_head.txt`). Never claim "all tests pass" without
diffing the failure list against HEAD.

Technique:
```bash
uv run pytest tests/unit/ -q > /tmp/before.txt 2>&1
git stash
uv run pytest tests/unit/ -q > /tmp/after.txt 2>&1
git stash pop
diff /tmp/before.txt /tmp/after.txt
```

Never silently fix pre-existing failures in an unrelated PR. If a fix is
intentional, it belongs in its own commit with a clear message.

## 7. House gates

Marker discipline:
- Use `@pytest.mark.unit` on every fast, offline test. `pytest.ini` has
  `--strict-markers`, so undeclared markers raise.

Run scope:
- Per-area: `uv run pytest tests/unit/simulator_agent/ -q`
- Full suite only when touching shared modules (entity, effects, planning).
- Long files run solo (`uv run pytest tests/unit/simulator_agent/test_agent.py -q`)
   because combined runs exceed tool output windows.

Static gates:
- `ruff` on touched files.
- `mypy --strict` on non-test code (tests are excluded from strict typing).

## 8. Fixture hygiene

Factories are single-sourced in the folder `conftest.py`. Do not copy-paste
`__new__` seeding into individual test files.

Mirror-`__init__` seeding belongs in exactly one place. The `_exec_budget`
omission bug in an early conftest draft caused three tests to fail because the
diagnosis helper checked `s._exec_budget` and it was never seeded.

Thread-spawning tests (sandbox `run_code()` uses a worker thread) must assert
bounded runtime and clean teardown. The zombie-test lesson from the sandbox work:
a hung worker must be quarantined and the next `run_code()` must still work.
Exemplar: `tests/unit/simulator_agent/test_sandbox_safety.py:105-121` — asserts elapsed < 4s and
sandbox usable after containment.

## 9. Renaming and moving test files

When a test area outgrows a flat file, reorganize into `tests/unit/<agent>/`.
Always use `git mv` so history follows the file; verify with `git log --follow`.

Keep test class and function names stable so failure lists stay comparable.
Keep `@pytest.mark.unit` markers unchanged. Update internal self-references
(docstrings, comments, cross-file path strings) in the same commit.

Exemplar: the 2026-09-04 simulator-agent reorg moved six flat files into
`tests/unit/simulator_agent/`:
- `test_simulator_first_agent.py` -> `test_agent.py`
- `test_sandbox_budget.py` -> `test_sandbox_safety.py`
- `test_simulator_frame_index.py` -> `test_frame_index.py`
- `test_simulator_prompt.py` -> `test_prompts.py`
- `test_simulate_diagnosis.py` and `test_workflow_controller.py` kept their names.

After the move, run the per-area suite to confirm discovery and markers:
`uv run pytest tests/unit/simulator_agent/ -q`
