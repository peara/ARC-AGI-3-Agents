from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

from planning.mechanics_prompt import build_messages

log = logging.getLogger(__name__)

# Field caps
MAX_OBJECTIVE_CHARS = 200
MAX_MECHANICS_ITEMS = 5
MAX_PROGRESS_ITEMS = 5
MAX_CHANGES_CHARS = 500
MAX_NEXT_STEPS_CHARS = 300

# Notepad timing
COLD_START_FRAMES = 5
NOTEPAD_COOLDOWN_FRAMES = 8

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class MechanicsHypothesis:
    """
    A symbolic hypothesis about the game mechanics, updated iteratively by the LLM.
    """

    objective: str
    key_mechanics: tuple[str, ...]
    progress_signals: tuple[str, ...]
    entity_roles: dict[str, str]
    next_steps: str
    confidence: float
    status: Literal["initial", "confirmed", "refined", "refuted"]
    changes: str
    frame_index: int

    @classmethod
    def from_llm_response(cls, raw: dict[str, object], frame_index: int) -> MechanicsHypothesis:
        """
        Parses an LLM JSON response into a MechanicsHypothesis instance.
        Applies strict field caps and provides defaults for missing fields.
        """
        # 1. Extract and default basic strings
        obj = raw.get("objective", "")
        if not isinstance(obj, str):
            obj = str(obj)
        objective = obj[:MAX_OBJECTIVE_CHARS]

        changes = raw.get("changes", "")
        if not isinstance(changes, str):
            changes = str(changes)
        changes = changes[:MAX_CHANGES_CHARS]

        next_steps = raw.get("next_steps", "")
        if not isinstance(next_steps, str):
            next_steps = str(next_steps)
        next_steps = next_steps[:MAX_NEXT_STEPS_CHARS]

        # 2. Extract and cap lists/tuples
        def cap_list(key: str, limit: int) -> tuple[str, ...]:
            val = raw.get(key, [])
            if not isinstance(val, list):
                val = [str(v) for v in val] if hasattr(val, "__iter__") and not isinstance(val, (str, dict)) else [str(val)]
            return tuple(str(v) for v in val[:limit])

        key_mechanics = cap_list("key_mechanics", MAX_MECHANICS_ITEMS)
        progress_signals = cap_list("progress_signals", MAX_PROGRESS_ITEMS)

        # 3. Entity roles (dict)
        roles_raw = raw.get("entity_roles", {})
        if not isinstance(roles_raw, dict):
            entity_roles = {}
        else:
            entity_roles = {str(k): str(v) for k, v in roles_raw.items()}

        # 4. Confidence (float clamped [0, 1])
        conf_raw = raw.get("confidence", 0.0)
        try:
            confidence = float(cast(str | float | int, conf_raw))
        except (ValueError, TypeError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # 5. Status (Literal)
        status_raw = raw.get("status", "initial")
        valid_statuses = {"initial", "confirmed", "refined", "refuted"}
        status = cast(
            Literal["initial", "confirmed", "refined", "refuted"],
            status_raw if isinstance(status_raw, str) and status_raw in valid_statuses else "initial"
        )

        return cls(
            objective=objective,
            key_mechanics=key_mechanics,
            progress_signals=progress_signals,
            entity_roles=entity_roles,
            next_steps=next_steps,
            confidence=confidence,
            status=status,
            changes=changes,
            frame_index=frame_index,
        )

    def to_bundle_dict(self) -> dict[str, object]:
        """
        Returns a compact representation of the hypothesis for the LLM planner context.
        """
        return {
            "objective": self.objective,
            "next_steps": self.next_steps,
            "confidence": self.confidence,
            "status": self.status,
        }


class MechanicsNotepad:
    """Persistent mechanics hypothesis manager: trigger checks, LLM calls, parse/validate."""

    def __init__(self, llm_call: Callable[[list[dict[str, Any]]], str], vision_enabled: bool = False) -> None:
        self._llm_call = llm_call
        self._vision_enabled = vision_enabled
        self._hypothesis: MechanicsHypothesis | None = None
        self._last_update_frame: int = -NOTEPAD_COOLDOWN_FRAMES  # ensures cold start can fire immediately

    @property
    def hypothesis(self) -> MechanicsHypothesis | None:
        return self._hypothesis

    def should_trigger(
        self,
        frame_index: int,
        levels_completed: int,
        prev_levels_completed: int,
        new_confirmed_rules: list[object],
        diverged: bool,
        n_entities: int,
    ) -> bool:
        """Check if the notepad should update at this frame.

        Four triggers:
        1. Cold start: frame_index >= COLD_START_FRAMES and self._hypothesis is None and n_entities >= 2
        2. levels_completed change: levels_completed != prev_levels_completed
        3. New rule confirmed: len(new_confirmed_rules) > 0
        4. Divergence: diverged == True

        Cooldown: NOTEPAD_COOLDOWN_FRAMES between updates (except cold start which bypasses cooldown).
        Empty-scene guard: cold start trigger returns False if n_entities < 2.
        """
        # Cold start trigger
        if self._hypothesis is None:
            if frame_index >= COLD_START_FRAMES and n_entities >= 2:
                return True
            return False

        # Non-cold-start triggers: check cooldown first
        if frame_index - self._last_update_frame < NOTEPAD_COOLDOWN_FRAMES:
            return False

        # Check other triggers
        if levels_completed != prev_levels_completed:
            return True
        if len(new_confirmed_rules) > 0:
            return True
        if diverged:
            return True

        return False

    def update(
        self,
        frames: list[list[list[int]]],
        scene_summaries: list[dict[str, Any]],
        action_legend: dict[int, str],
        frame_index: int,
        levels_completed_delta: int = 0,
    ) -> MechanicsHypothesis | None:
        """Build prompt, call LLM, parse response, apply confidence monotonicity.

        Returns the new hypothesis, or None if the LLM call or parsing fails.
        """
        # Build full hypothesis dict for refinement prompt
        prev_dict: dict[str, Any] | None = None
        if self._hypothesis is not None:
            prev_dict = {
                "objective": self._hypothesis.objective,
                "key_mechanics": list(self._hypothesis.key_mechanics),
                "progress_signals": list(self._hypothesis.progress_signals),
                "entity_roles": self._hypothesis.entity_roles,
                "next_steps": self._hypothesis.next_steps,
                "confidence": self._hypothesis.confidence,
                "status": self._hypothesis.status,
                "changes": self._hypothesis.changes,
            }

        messages = build_messages(
            frames=frames,
            scene_summaries=scene_summaries,
            action_legend=action_legend,
            prev_hypothesis=prev_dict,
            levels_completed_delta=levels_completed_delta,
            vision_enabled=self._vision_enabled,
        )

        try:
            raw_response = self._llm_call(messages)
        except Exception as exc:
            log.warning("mechanics_notepad: LLM call failed: %s", exc)
            return None

        from agents.llm_client import ChatResponse as _CR
        raw_str = raw_response.content if isinstance(raw_response, _CR) else raw_response
        parsed = self._parse_json_response(raw_str)
        if parsed is None:
            log.warning("mechanics_notepad: could not parse LLM response as JSON")
            return None

        new_hypothesis = MechanicsHypothesis.from_llm_response(parsed, frame_index=frame_index)

        # Confidence monotonicity: if both prev and new are confirmed/refined,
        # new confidence = max(new, prev)
        if self._hypothesis is not None:
            if (
                self._hypothesis.status in ("confirmed", "refined")
                and new_hypothesis.status in ("confirmed", "refined")
            ):
                new_confidence = max(new_hypothesis.confidence, self._hypothesis.confidence)
                # Since frozen, create new instance with max confidence
                new_hypothesis = MechanicsHypothesis(
                    objective=new_hypothesis.objective,
                    key_mechanics=new_hypothesis.key_mechanics,
                    progress_signals=new_hypothesis.progress_signals,
                    entity_roles=new_hypothesis.entity_roles,
                    next_steps=new_hypothesis.next_steps,
                    confidence=new_confidence,
                    status=new_hypothesis.status,
                    changes=new_hypothesis.changes,
                    frame_index=new_hypothesis.frame_index,
                )

        self._hypothesis = new_hypothesis
        self._last_update_frame = frame_index
        log.info(
            "mechanics_notepad: updated hypothesis at frame %d: status=%s confidence=%.2f objective='%.80s'",
            frame_index, new_hypothesis.status, new_hypothesis.confidence, new_hypothesis.objective,
        )
        return self._hypothesis

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any] | None:
        """Extract a JSON dict from an LLM response string.

        Tries markdown ```json ... ``` blocks first, then falls back to
        parsing the raw string directly.  Returns None on any parse failure.
        """
        # Try fenced JSON blocks
        for match in _JSON_BLOCK_RE.finditer(raw):
            try:
                result: dict[str, Any] = json.loads(match.group(1))
                return result
            except json.JSONDecodeError:
                continue

        # Fallback: try the whole string
        try:
            result = json.loads(raw.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        return None

    def reset(self) -> None:
        """Clear per-level state. Currently just clears the hypothesis.

        This is the single method where future cross-level knowledge
        preservation logic would go (e.g., keeping a generic objective
        hypothesis while clearing entity-specific details).
        For now: simply clears the hypothesis.
        """
        self._hypothesis = None
        self._last_update_frame = -NOTEPAD_COOLDOWN_FRAMES

    def to_bundle_dict(self) -> dict[str, Any] | None:
        """Return compact hypothesis dict for the planner, or None if no hypothesis."""
        if self._hypothesis is None:
            return None
        return self._hypothesis.to_bundle_dict()


__all__ = [
    "MechanicsHypothesis",
    "MechanicsNotepad",
    "MAX_OBJECTIVE_CHARS",
    "MAX_MECHANICS_ITEMS",
    "MAX_PROGRESS_ITEMS",
    "MAX_CHANGES_CHARS",
    "MAX_NEXT_STEPS_CHARS",
    "COLD_START_FRAMES",
    "NOTEPAD_COOLDOWN_FRAMES",
]
