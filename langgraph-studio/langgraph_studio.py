"""
LangGraph Studio Entry Point

Provides the graph factory function for LangGraph Studio
visualization and debugging.

File lives in langgraph-studio/ folder at project root.
sys.path adds project root so imports from src work correctly.

Current phase: entity_extractor → entity_normalizer → END
"""

import sys
from pathlib import Path

# Add project root to Python path so `from src.X import Y` works
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

# Import connections — these are lazy proxies; the Neo4j connection and schema are created on first use, not at import
from src.graph.connections import graph_interface, schema_provider  # noqa: E402, F401

from src.workflow.graph_builder import build_workflow  # noqa: E402

load_dotenv()


def create_graph():
    """
    Create and return the workflow graph for LangGraph Studio.

    Connections are already initialized by importing src.graph.connections.
    Returns the compiled workflow.

    Returns:
        Compiled LangGraph workflow
    """
    return build_workflow()


# LangGraph Studio looks for this module-level `graph` variable
graph = create_graph()