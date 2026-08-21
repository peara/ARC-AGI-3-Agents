"""Thin wrapper — real implementation in agents/simulator_agent/experiment.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.simulator_agent.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
