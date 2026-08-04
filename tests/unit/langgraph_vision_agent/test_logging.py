from __future__ import annotations

import logging

import pytest
from arcengine import GameAction

from agents.langgraph_vision_agent.logging import (
    extract_state_for_recording,
    log_frame,
    log_node,
)


@pytest.mark.unit
class TestLogging:
    """Test log_frame and log_node produce correct output."""

    def test_log_frame_emits_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="langgraph.frame"):
            log_frame(
                frame_index=5,
                node_path=["observe", "reflect", "plan"],
                action=GameAction.ACTION1,
                uncertain=False,
                reason="clear path",
                latency_ms=120,
            )
        assert len(caplog.records) >= 1
        msg = caplog.records[-1].getMessage()
        assert "frame=5" in msg
        assert "path=observe/reflect/plan" in msg
        assert "action=1" in msg
        assert "uncertain=False" in msg
        assert "latency_ms=120" in msg

    def test_log_frame_none_action(self, caplog):
        with caplog.at_level(logging.INFO, logger="langgraph.frame"):
            log_frame(
                frame_index=0,
                node_path=[],
                action=None,
                uncertain=True,
                reason="testing",
                latency_ms=50,
            )
        msg = caplog.records[-1].getMessage()
        assert "action=None" in msg
        assert "path=none" in msg

    def test_log_node_emits_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="langgraph.node"):
            log_node(3, "observe", grid_changed=True, cells_changed=42)
        assert len(caplog.records) >= 1
        msg = caplog.records[-1].getMessage()
        assert "frame=3" in msg
        assert "node=observe" in msg
        assert "grid_changed=True" in msg
        assert "cells_changed=42" in msg

    def test_extract_state_for_recording(self):
        state = {
            "mechanics": ["player moves"],
            "tactical": ["avoid walls"],
            "plan": "go north",
            "uncertain_about": None,
            "node_path": ["observe", "reflect", "plan"],
            "extra_field": "should_not_appear",
        }
        extracted = extract_state_for_recording(state)
        assert extracted["mechanics"] == ["player moves"]
        assert extracted["tactical"] == ["avoid walls"]
        assert extracted["plan"] == "go north"
        assert extracted["uncertain_about"] is None
        assert extracted["node_path"] == ["observe", "reflect", "plan"]
        assert "extra_field" not in extracted

    def test_extract_state_for_recording_defaults(self):
        extracted = extract_state_for_recording({})
        assert extracted["mechanics"] == []
        assert extracted["tactical"] == []
        assert extracted["plan"] == ""
        assert extracted["uncertain_about"] is None
        assert extracted["node_path"] == []
