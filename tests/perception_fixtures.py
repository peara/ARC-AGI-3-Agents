"""Shared helpers for perception/planning tests across game recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from perception import EntityCatalog, ObjectRegistry, load_recording_frames
from perception.session import PerceptionSession

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / "reference_recordings.json"


@dataclass(frozen=True)
class RecordingRef:
    name: str
    path: Path


@dataclass(frozen=True)
class PlanCase:
    recording: RecordingRef
    entity_id: int
    start_frame: int
    goal_frame: int


@dataclass(frozen=True)
class PerceptionExpect:
    recording: RecordingRef
    controllable_entity_id: int | None
    min_counters: int = 0
    min_animation_events: int = 0
    expect_non_markovian: bool = False


@dataclass
class PerceptionStack:
    """Everything built from one recording file."""

    recording: RecordingRef
    frames: list
    action_ids: list[int]
    registry: ObjectRegistry
    catalog: EntityCatalog
    session: PerceptionSession


def plan_case_id(case: PlanCase) -> str:
    return (
        f"{case.recording.name}-e{case.entity_id}"
        f"-f{case.start_frame}-g{case.goal_frame}"
    )


def load_perception_expectations(
    manifest_path: Path | None = None,
) -> list[PerceptionExpect]:
    path = manifest_path or MANIFEST_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[PerceptionExpect] = []
    for entry in data.get("recordings", []):
        perc = entry.get("perception")
        if not perc:
            continue
        rec = RecordingRef(
            name=entry["name"],
            path=REPO_ROOT / entry["path"],
        )
        ctrl = perc.get("controllable_entity_id")
        out.append(
            PerceptionExpect(
                recording=rec,
                controllable_entity_id=int(ctrl) if ctrl is not None else None,
                min_counters=int(perc.get("min_counters", 0)),
                min_animation_events=int(perc.get("min_animation_events", 0)),
                expect_non_markovian=bool(perc.get("expect_non_markovian", False)),
            )
        )
    return out


def load_manifest(manifest_path: Path | None = None) -> list[PlanCase]:
    path = manifest_path or MANIFEST_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: list[PlanCase] = []
    for entry in data.get("recordings", []):
        rec = RecordingRef(
            name=entry["name"],
            path=REPO_ROOT / entry["path"],
        )
        for pc in entry.get("plan_cases", []):
            cases.append(
                PlanCase(
                    recording=rec,
                    entity_id=int(pc["entity_id"]),
                    start_frame=int(pc["start_frame"]),
                    goal_frame=int(pc["goal_frame"]),
                )
            )
    return cases


def build_perception_stack(recording_path: Path) -> PerceptionStack:
    frames, action_ids, _subframes = load_recording_frames(str(recording_path))
    session, _ = PerceptionSession.from_recording(recording_path)
    scene = session.snapshot()
    name = recording_path.stem.replace(".recording", "")
    return PerceptionStack(
        recording=RecordingRef(name=name, path=recording_path),
        frames=frames,
        action_ids=action_ids,
        registry=session.registry,
        catalog=scene.catalog,
        session=session,
    )
