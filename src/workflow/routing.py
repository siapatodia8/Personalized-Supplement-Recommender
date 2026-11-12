"""
Routing - Workflow Traffic Controller

Maps supervisor decisions to node names.
No intelligence here — pure string-to-node-name mapping.
This file turns the supervisor's decision, written in words, into the name of the step that runs next. 
It's a lookup table with no thinking in it.
"""

from typing import Any, Dict


# ==================== NODE NAMES ====================

class NodeNames:
    """
    Constants for all node names in the workflow.
    Used in graph_builder.py and routing functions.
    """
    # Phase 1: Pre-processing pipeline
    ENTITY_EXTRACTOR = "entity_extractor"
    ENTITY_NORMALIZER = "entity_normalizer"

    # Phase 2: Dynamic routing loop
    SUPERVISOR = "supervisor"
    SAFETY_CHECK = "safety_check"
    DEFICIENCY_CHECK = "deficiency_check"
    RECOMMENDATION = "recommendation"

    # Phase 3: Final output
    SYNTHESIS = "synthesis"

    # END is not defined here. The workflow uses LangGraph's own END,
    # imported in graph_builder.py (from langgraph.graph import StateGraph, END).
    # END = "END"


# ==================== SUPERVISOR ROUTING ====================

def route_supervisor_decision(state: Dict[str, Any]) -> str:
    """
    Route based on supervisor's decision string.

    Reads state['supervisor_decision'] and maps it to a node name.
    Called by LangGraph's conditional edge from the supervisor node.

    Decision strings must match exactly what supervisor.py writes
    to state['supervisor_decision'].

    Args:
        state: Current ConversationState

    Returns:
        Node name string to route to next
    """
    decision = state.get('supervisor_decision', '')

    route_map = {
        'check_safety':       NodeNames.SAFETY_CHECK,
        'check_deficiency':   NodeNames.DEFICIENCY_CHECK,
        'get_recommendations': NodeNames.RECOMMENDATION,
        'synthesize':         NodeNames.SYNTHESIS,
        'loop_back':          NodeNames.SUPERVISOR,
    }

    next_node = route_map.get(decision, NodeNames.SYNTHESIS)
    print(f"🚦 ROUTING: '{decision}' → {next_node}")
    return next_node