"""Remove controllable references from two recorded prompts and replay them."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.llm_client import LLMClient  # noqa: E402

RECORDING = (
    Path(__file__).resolve().parent.parent
    / "recordings"
    / "wa30-ee6fef47.llmcuriosity.4a42dc54-ae1c-48b7-ad38-06ed7e3108e1.llm.jsonl"
)
OUT = Path(".omo/experiments/remove_controllable_results.json")


def get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and "text" in p
        )
    return ""


def strip_controllable(content):
    # JSON fields
    content = re.sub(r'"controllable_id":\s*(?:\d+|null),?\s*\n', "", content)
    content = re.sub(r'"controllable_pos":\s*\[[\d.,\s]+\],?\s*\n', "", content)
    content = re.sub(r'"controllable":\s*(?:true|false|null),?\s*\n', "", content)
    content = re.sub(r'"role":\s*"controllable"', '"role": null', content)
    content = re.sub(
        r'"motion_by_action":\s*\{[^}]*\},?\s*\n', "", content, flags=re.DOTALL
    )
    content = re.sub(r'"motion_agreement":\s*[\d.]+,?\s*\n', "", content)
    content = re.sub(r'"detector":\s*"action_displacement_v1",?\s*\n', "", content)
    content = re.sub(
        r'"expected_motion":\s*(?:\[[^\]]*\]|null),?\s*\n', "", content
    )
    content = re.sub(r'"blocked":\s*(?:true|false),?\s*\n', "", content)
    # System prompt text
    content = content.replace(
        "the controllable's expected motion (`expected_motion`), "
        "and whether the\ncontrollable was blocked",
        "which entities changed",
    )
    content = content.replace("the controllable's\npos", "an entity's pos")
    content = content.replace(
        "the controllable tried to move into", "the moving entity tried to move into"
    )
    content = content.replace(
        "the controllable and the\nblocker", "the moving entity and the\nblocker"
    )
    content = content.replace("<controllable>", "<moving_entity>")
    content = content.replace(
        "already labels `controllable` (the player) and `counter`",
        "already labels `counter`",
    )
    content = re.sub(
        r"Controllable entity: \S+ at \[[\d.,\s]+\]\s*\n", "", content
    )
    # Clean trailing commas
    content = re.sub(r",\s*}", "}", content)
    content = re.sub(r",\s*]", "]", content)
    return content


def strip_messages(messages):
    out = []
    for m in messages:
        text = get_text(m["content"])
        text = strip_controllable(text)
        out.append({"role": m["role"], "content": text})
    return out


def find_prompt(kind, frame):
    with open(RECORDING) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == kind and rec.get("frame_index") == frame:
                return rec["messages"], rec.get("response", "") or rec.get(
                    "response_raw", ""
                )
    raise ValueError(f"prompt not found: {kind} frame={frame}")


def run(kind, frame):
    messages, original_response = find_prompt(kind, frame)
    modified = strip_messages(messages)

    client = LLMClient()
    resp = client._client.chat.completions.create(
        model=client.model,
        messages=modified,
        temperature=0.3,
        max_tokens=4096,
    )
    modified_response = resp.choices[0].message.content or ""

    print(f"=== {kind} frame={frame} ===")
    print("ORIGINAL (from recording):")
    print(original_response[:600])
    print()
    print("MODIFIED (sent to LLM):")
    print(modified_response[:600])
    print()

    return {
        "kind": kind,
        "frame": frame,
        "original_response": original_response,
        "modified_response": modified_response,
    }


def main():
    results = [
        run("rule_proposer", 8),
        run("mechanics", 5),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
