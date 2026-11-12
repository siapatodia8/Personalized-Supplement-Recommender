"""
LangGraph Studio Entry Point

Provides the graph factory function for LangGraph Studio
visualization and debugging.

File lives in langgraph-studio/ folder at project root.
sys.path adds project root so imports from src work correctly.

Runs the same full workflow as the API.
One shared builder means Studio and the API cannot drift apart - what is developed and debugged in Studio is what ships.
"""

import sys
from pathlib import Path

# Add project root to Python path so `from src.X import Y` works
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.graph_builder import build_workflow  # noqa: E402


def create_graph():
    """
    Create and return the workflow graph for LangGraph Studio.

    Connections are created lazily on first use.
    Returns the compiled workflow.

    Returns:
        Compiled LangGraph workflow
    """
    return build_workflow()


# LangGraph Studio looks for this module-level `graph` variable
graph = create_graph()