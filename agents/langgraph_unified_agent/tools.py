"""OpenAI-compatible tool schemas for the unified LangGraph agent.

This module defines the static JSON-schema tool descriptions that are passed
to an OpenAI-compatible chat completion endpoint.  They are the single source
of truth for the ``inspect`` and ``decide`` tools used by the unified agent
loop.
"""

INSPECT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Run Python code in a sandbox to inspect game state. "
            "Variables available: objects (list of dicts with color/size/centroid/bbox/hash), "
            "adjacency (frozenset of index pairs), history (list of previous frames' "
            "objects/adjacency). Use print() to output results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the sandbox",
                }
            },
            "required": ["code"],
        },
    },
}

DECIDE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "decide",
        "description": "Make your final action decision. Call this after inspecting the state with inspect().",
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": "The action ID to execute",
                },
                "expectation": {
                    "type": "string",
                    "description": "What you expect to happen next frame",
                },
                "reflect": {
                    "type": "boolean",
                    "description": "Whether to update the world model (mechanics/tactical)",
                },
                "mechanics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated mechanics rules with [HIGH/MEDIUM/LOW] tags",
                },
                "mechanics_summary": {
                    "type": "string",
                    "description": "One paragraph summary of game mechanics",
                },
                "tactical": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tactical observations and next goals",
                },
                "tactical_summary": {
                    "type": "string",
                    "description": "One sentence summary of current strategy",
                },
            },
            "required": ["action_id", "expectation", "reflect"],
        },
    },
}

UNIFIED_TOOLS: list = [INSPECT_TOOL, DECIDE_TOOL]

__all__ = ["INSPECT_TOOL", "DECIDE_TOOL", "UNIFIED_TOOLS"]
