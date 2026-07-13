"""Experiment: replay LLM planner calls with grid images added.

Pairs a recording (grid frames) with its LLM log (planner messages),
reconstructs each planner call as a multimodal message (image + original
text), sends it to a vision-capable model, and compares the response to
the original text-only response.

Usage:
    uv run python scripts/vision_planner_experiment.py \
        recordings/ls20-9607627b.llmcuriosity.1ec2be1d-5275-4dbf-bae7-cbee5b3ccb61 \
        --model gemini-3-flash-preview --max-calls 5

The recording base path (without .recording.jsonl / .llm.jsonl suffix) is
resolved automatically.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai

log = logging.getLogger("vision_experiment")

from vision.render import grid_to_image, image_to_base64, make_image_block
from vision.palette import ARCADE_PALETTE


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def resolve_paths(base: str) -> tuple[Path, Path]:
    bp = Path(base)
    if bp.suffix == ".jsonl":
        bp = bp.with_suffix("")
    rec = bp.with_suffix(".recording.jsonl")
    llm = bp.with_suffix(".llm.jsonl")
    if not rec.exists():
        # Try with full stem
        rec = Path(f"{base}.recording.jsonl")
        llm = Path(f"{base}.llm.jsonl")
    if not rec.exists():
        raise FileNotFoundError(f"Recording not found: {rec}")
    if not llm.exists():
        raise FileNotFoundError(f"LLM log not found: {llm}")
    return rec, llm


def load_recording_frames(rec_path: Path) -> dict[int, list[list[int]]]:
    """Return frame_index → last subframe grid (the 64x64 color grid)."""
    frames: dict[int, list[list[int]]] = {}
    with rec_path.open() as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            frame_data = obj["data"]["frame"]
            # frame[subframe][row][col]; take the last subframe (final state)
            grid = frame_data[-1]
            frames[idx] = grid
    return frames


def load_planner_calls(llm_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with llm_path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("kind") == "planner":
                calls.append(obj)
    return calls


# ---------------------------------------------------------------------------
# Multimodal message construction
# ---------------------------------------------------------------------------


def to_multimodal(
    messages: list[dict[str, str]],
    grid_b64: str | None,
) -> list[dict[str, Any]]:
    """Convert text-only messages to multimodal, prepending an image to the user message."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and grid_b64 is not None:
            out.append({
                "role": role,
                "content": [
                    make_image_block(grid_b64),
                    {"type": "text", "text": content},
                ],
            })
        else:
            out.append({"role": role, "content": content})
    return out


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment(
    rec_path: Path,
    llm_path: Path,
    model: str,
    max_calls: int,
    base_url: str,
    api_key: str,
    output_path: Path | None = None,
) -> None:
    frames = load_recording_frames(rec_path)
    planner_calls = load_planner_calls(llm_path)
    log.info(
        "Loaded %d frames, %d planner calls", len(frames), len(planner_calls)
    )

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    results: list[dict[str, Any]] = []
    n = min(max_calls, len(planner_calls))

    for i in range(n):
        call = planner_calls[i]
        frame_idx = call["frame_index"]
        original_messages = call["messages"]
        original_response = call.get("response_raw", "")

        grid = frames.get(frame_idx)
        if grid is None:
            log.warning("No grid for frame_index=%d, skipping", frame_idx)
            continue

        grid_img = grid_to_image(grid)
        grid_b64 = image_to_base64(grid_img)

        multimodal_messages = to_multimodal(original_messages, grid_b64)

        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=multimodal_messages,  # type: ignore[arg-type]
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            vision_response = resp.choices[0].message.content or ""
            ok = True
            error = None
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            vision_response = ""
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        # Parse both responses as JSON for structured comparison
        original_parsed = _try_parse_json(original_response)
        vision_parsed = _try_parse_json(vision_response)

        result = {
            "call_index": i,
            "frame_index": frame_idx,
            "ok": ok,
            "latency_ms": latency_ms,
            "error": error,
            "original_response": original_response[:2000],
            "vision_response": vision_response[:2000],
            "original_parsed": original_parsed,
            "vision_parsed": vision_parsed,
        }
        results.append(result)

        status = "OK" if ok else "FAIL"
        print(f"\n{'='*70}")
        print(f"[{status}] Call {i} (frame={frame_idx}, {latency_ms}ms)")
        print(f"  Original: {_summarize(original_parsed)}")
        print(f"  Vision:   {_summarize(vision_parsed)}")
        if not ok:
            print(f"  Error: {error}")

        # Save grid image for manual inspection
        if output_path:
            img_dir = output_path.parent / "vision_experiment_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            grid_img.save(img_dir / f"frame_{frame_idx:04d}.png")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("Results saved to %s", output_path)


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM response."""
    import re
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _summarize(parsed: dict[str, Any] | None) -> str:
    if parsed is None:
        return "<unparseable>"
    target = parsed.get("target", {})
    reason = str(parsed.get("reason", ""))[:80]
    action = parsed.get("action")
    near = target.get("near", "?")
    if isinstance(near, dict):
        near = f"entity {near.get('of', '?')} r={near.get('radius', '?')}"
    return f"target={near} action={action} reason={reason!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_base", help="Recording base path (without .recording.jsonl)")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--max-calls", type=int, default=5)
    parser.add_argument("--output", "-o", default=None, help="Save JSON results to this path")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load credentials from .env if not provided
    env_path = REPO_ROOT / ".env"
    base_url = args.base_url
    api_key = args.api_key
    if env_path.exists():
        env_text = env_path.read_text()
        if not base_url:
            m = re.search(r"^LLM_BASE_URL=(.+)$", env_text, re.MULTILINE)
            if m:
                base_url = m.group(1).strip()
        if not api_key:
            m = re.search(r"^LLM_API_KEY=(.+)$", env_text, re.MULTILINE)
            if m:
                api_key = m.group(1).strip()
    if not base_url:
        base_url = os.environ.get("LLM_BASE_URL", "")
    if not api_key:
        api_key = os.environ.get("LLM_API_KEY", "")
    if not base_url or not api_key:
        print("Error: LLM_BASE_URL and LLM_API_KEY required (from .env, --base-url, or env)", file=sys.stderr)
        sys.exit(1)

    rec_path, llm_path = resolve_paths(args.recording_base)
    output_path = Path(args.output) if args.output else None

    run_experiment(
        rec_path=rec_path,
        llm_path=llm_path,
        model=args.model,
        max_calls=args.max_calls,
        base_url=base_url,
        api_key=api_key,
        output_path=output_path,
    )


if __name__ == "__main__":
    import re  # noqa: E402
    main()