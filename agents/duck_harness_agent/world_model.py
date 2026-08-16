"""Free-text world model parser for Duck Harness agent.

Parses labeled text blocks from LLM responses using regex (not JSON).
The Duck's world model uses 7 canonical labels carried across turns:
  World model, Goal model, Action model, Recent findings,
  Open questions, Plan, Cross-level notes

Plus 3 fallback labels that map to canonical keys:
  Hypothesis → world_model
  History check → recent_findings
  Next test → current_plan
"""

from __future__ import annotations

import re

# ── Canonical label definitions ──────────────────────────────────────────

# (display_label, dict_key) — order matters for format_world_model
CANONICAL_LABELS: list[tuple[str, str]] = [
    ("World model", "world_model"),
    ("Goal model", "goal_model"),
    ("Action model", "action_model"),
    ("Recent findings", "recent_findings"),
    ("Open questions", "open_questions"),
    ("Plan", "current_plan"),
    ("Cross-level notes", "cross_level_notes"),
]

# Fallback labels that map to canonical keys
FALLBACK_LABELS: dict[str, str] = {
    "hypothesis": "world_model",
    "history check": "recent_findings",
    "next test": "current_plan",
}

DISPLAY_TO_KEY: dict[str, str] = {label: key for label, key in CANONICAL_LABELS}
ALL_LABELS: list[str] = [label for label, _ in CANONICAL_LABELS]
ALL_KEYS: list[str] = [key for _, key in CANONICAL_LABELS]

# ── Regex for labeled blocks ──────────────────────────────────────────────

# Matches lines like:
#   "World model: some text"
#   "- Goal model: some text"
#   "* Action model: some text"
# The leading dash/star and whitespace before the label are stripped.
_LABEL_LINE_RE = re.compile(
    r"^\s*[-*]?\s*"            # optional leading - or * plus whitespace
    r"(.+?)"                    # capture group 1: the label name
    r"\s*:\s*"                  # colon separator (with optional whitespace)
    r"(.*)"                     # capture group 2: rest of the first line
    r"$",
    re.MULTILINE,
)


def extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    """Extract multi-line labeled blocks from free-form text.

    Each block starts with a line matching ``Label:`` (case-insensitive,
    optional leading ``-`` or ``*``).  Content continues until the next
    label or end of text.

    Args:
        content: The LLM response text to parse.
        labels:  Label names to look for (case-insensitive matching).

    Returns:
        Dict mapping *lowercased* label names to their stripped content.
        Labels not found in ``content`` map to empty strings.
    """
    # Build a case-insensitive lookup from normalised label → original label key
    label_set = {label.lower(): label for label in labels}
    result: dict[str, str] = {label.lower(): "" for label in labels}

    # Scan all label-line matches in order
    matches = list(_LABEL_LINE_RE.finditer(content))

    for idx, match in enumerate(matches):
        raw_label = match.group(1).strip()
        first_line_content = match.group(2).strip()
        normalised = raw_label.lower()

        if normalised not in label_set:
            continue

        # Collect continuation lines until the next label or end
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        block_rest = content[start:end]

        # Combine first-line content with continuation
        full_content = (first_line_content + "\n" + block_rest).strip() if first_line_content else block_rest.strip()

        # Only overwrite if we found actual content (or keep empty)
        result[normalised] = full_content

    return result


def extract_world_model(content: str) -> dict[str, str]:
    """Extract the Duck's 7-block world model from LLM response text.

    Uses the 7 canonical labels plus 3 fallback mappings.
    Returns a dict with all 7 keys; missing labels map to empty strings.
    """
    # First pass: extract with canonical + fallback labels
    all_search_labels = list(ALL_LABELS) + list(FALLBACK_LABELS.keys())
    raw_blocks = extract_labeled_blocks(content, all_search_labels)

    # Map to canonical keys
    result: dict[str, str] = {key: "" for key in ALL_KEYS}

    # Fill canonical labels first
    for display_label, key in CANONICAL_LABELS:
        normalised = display_label.lower()
        if normalised in raw_blocks and raw_blocks[normalised]:
            result[key] = raw_blocks[normalised]

    # Fill fallbacks only if canonical is still empty
    for fallback_label, target_key in FALLBACK_LABELS.items():
        if not result[target_key] and fallback_label in raw_blocks and raw_blocks[fallback_label]:
            result[target_key] = raw_blocks[fallback_label]

    return result


def format_world_model(model: dict[str, str]) -> str:
    """Format a world model dict into the Duck's text representation.

    Includes header and footer.  Labels appear in canonical order.
    Empty values produce just ``Label:`` (no trailing content).
    """
    lines: list[str] = ["Working world model carried from earlier turns:"]

    for display_label, key in CANONICAL_LABELS:
        value = model.get(key, "")
        lines.append(f"{display_label}: {value}" if value else f"{display_label}:")

    lines.append("- Revise any item above immediately if current_frame contradicts it.")
    return "\n".join(lines)


def clear_world_model(model: dict[str, str]) -> dict[str, str]:
    """Return a copy of *model* with all values set to empty strings."""
    return {key: "" for key in model}