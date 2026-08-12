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

# --- Experimental: decide with nested world_model object ---

DECIDE_V2_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "decide",
        "description": (
            "Make your final action decision. Call this after inspecting the "
            "state with inspect(). You must also provide your updated world "
            "model — the current understanding of the game scene, mechanics, "
            "and tactical situation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": "The action ID to execute",
                },
                "expectation": {
                    "type": "string",
                    "description": (
                        "A specific, testable prediction about what will "
                        "change next frame. Next frame, you will see whether "
                        "this prediction was met. If it was not met, your "
                        "understanding of that action is wrong or incomplete."
                    ),
                },
                "reflect": {
                    "type": "boolean",
                    "description": (
                        "Set to true if you learned something new this frame "
                        "(confirmed or disproved a conjecture, an action "
                        "failed, something unexpected happened). Set to false "
                        "for routine moves that worked as expected."
                    ),
                },
                "world_model": {
                    "type": "object",
                    "description": (
                        "Your current understanding of the game. Update each "
                        "frame based on what you observed. Keep entries from "
                        "previous frames that are still valid; add new ones; "
                        "drop ones that are disproven."
                    ),
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "What each available action does. One entry per action ID, "
                                "e.g. '1=UP (confirmed)', '5=unknown, not yet tested'. "
                                "Mark actions you have tested as (confirmed) or (guessed). "
                                "Mark untested actions as (unknown). You must include ALL "
                                "available actions every turn."
                            ),
                        },
                        "mechanics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Confirmed rules about how the game works. "
                                "Tag each entry with [HIGH], [MEDIUM], or "
                                "[LOW] confidence."
                            ),
                        },
                        "mechanics_summary": {
                            "type": "string",
                            "description": (
                                "One paragraph summarizing the current game "
                                "mechanics."
                            ),
                        },
                        "tactical": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Tactical observations, conjectures, and next "
                                "goals. If you don't know the goal yet, "
                                "include at least one testable conjecture."
                            ),
                        },
                        "tactical_summary": {
                            "type": "string",
                            "description": (
                                "One sentence summarizing the current strategy."
                            ),
                        },
                    },
                    "required": ["actions", "mechanics", "tactical"],
                },
            },
            "required": ["action_id", "expectation", "reflect", "world_model"],
        },
    },
}

UNIFIED_TOOLS_V2: list = [INSPECT_TOOL, DECIDE_V2_TOOL]

__all__ = [
    "INSPECT_TOOL",
    "DECIDE_TOOL",
    "UNIFIED_TOOLS",
    "DECIDE_V2_TOOL",
    "UNIFIED_TOOLS_V2",
]
