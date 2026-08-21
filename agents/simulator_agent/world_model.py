from __future__ import annotations

import re

_LABEL_LINE_RE = re.compile(
    r"^\s*[-*]*\s*"
    r"(.+?)"
    r"\s*:\s*"
    r"(.*)"
    r"$",
    re.MULTILINE,
)

_LABELS = ["notes", "plan"]


def extract_notes(content: str) -> dict[str, str]:
    result: dict[str, str] = {"notes": "", "plan": ""}
    matches = list(_LABEL_LINE_RE.finditer(content))
    label_set = {l.lower() for l in _LABELS}

    for idx, match in enumerate(matches):
        raw_label = match.group(1).strip().strip("*")
        if raw_label.lower() not in label_set:
            continue
        first_line = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        rest = content[start:end].strip()
        full = (first_line + "\n" + rest).strip() if first_line else rest
        result[raw_label.lower()] = full

    return result


def format_notes(notes: dict[str, str]) -> str:
    lines: list[str] = ["Notes carried from earlier turns:"]
    for label in _LABELS:
        value = notes.get(label, "")
        display = label.capitalize()
        lines.append(f"{display}: {value}" if value else f"{display}:")
    lines.append("- Revise anything above if it contradicts what you see now.")
    return "\n".join(lines)