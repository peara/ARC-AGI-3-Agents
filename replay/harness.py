from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arc_agi import Arcade, EnvironmentWrapper, OperationMode
from arcengine import FrameData, FrameDataRaw, GameAction, GameState


class ReplayHarness:
    """Thin wrapper that reconstructs game state at any frame from a recording.

    Replays recorded actions through a fresh offline Arcade environment and
    stores converted FrameData frames. No perception, entity, or agent state.
    """

    def __init__(self, env: EnvironmentWrapper, action_inputs: list[dict[str, Any]]) -> None:
        self.env = env
        self.action_inputs = action_inputs
        self.frames: list[FrameData] = []

    @classmethod
    def from_recording(cls, path: str | Path, *, seed: int = 0) -> ReplayHarness:
        """Load a recording, create a fresh offline environment, and return a harness.

        The reset frame is NOT in the recording; callers must replay_to(0) to
        capture the initial state.
        """
        recording_path = Path(path)
        if not recording_path.is_file():
            raise FileNotFoundError(f"Recording not found: {recording_path}")

        game_id: str | None = None
        action_inputs: list[dict[str, Any]] = []

        with open(recording_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                data = event.get("data", {})
                if "action_input" not in data:
                    continue
                if game_id is None:
                    game_id = data.get("game_id")
                action_inputs.append(data["action_input"])

        if game_id is None:
            raise ValueError(f"No action lines found in {recording_path}")

        arc = Arcade(operation_mode=OperationMode.NORMAL)
        env = arc.make(game_id, seed=seed)
        if env is None:
            raise RuntimeError(f"Arcade.make returned None for game_id={game_id} seed={seed}")

        return cls(env, action_inputs)

    def replay_to(self, frame: int) -> EnvironmentWrapper:
        """Replay actions 0..frame-1 and return the environment.

        After the call, self.frames holds the reset frame plus one frame per
        replayed action. replay_to(0) captures the reset frame only.
        """
        if frame < 0:
            raise ValueError(f"frame must be non-negative, got {frame}")

        if not self.frames:
            initial = self.env.reset()
            if initial is None:
                raise RuntimeError("env.reset() returned None")
            self.frames.append(self._convert_raw_frame_data(initial))

        target_len = frame + 1
        while len(self.frames) < target_len:
            action_index = len(self.frames) - 1
            ai = self.action_inputs[action_index]

            action_id = ai["id"]
            if action_id == 0 or action_id == "RESET":
                raw = self.env.reset()
            else:
                action_data = ai.get("data", {}).copy()
                action_data.pop("game_id", None)
                reasoning = ai.get("reasoning")
                if reasoning is not None and not isinstance(reasoning, dict):
                    reasoning = {"text": str(reasoning)}

                action = GameAction.from_id(action_id)
                action.set_data(action_data)

                raw = self.env.step(action, data=action_data, reasoning=reasoning)

            if raw is None:
                raise RuntimeError(f"env returned None at action_index={action_index}")

            frame_data = self._convert_raw_frame_data(raw)
            self.frames.append(frame_data)

            if frame_data.state == GameState.GAME_OVER:
                break

        return self.env

    def replay_all(self) -> EnvironmentWrapper:
        """Replay every recorded action and return the environment."""
        return self.replay_to(len(self.action_inputs))

    @staticmethod
    def _convert_raw_frame_data(raw: FrameDataRaw) -> FrameData:
        """Convert arcengine FrameDataRaw to a serializable FrameData."""
        return FrameData(
            game_id=raw.game_id,
            frame=[arr.tolist() for arr in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )
