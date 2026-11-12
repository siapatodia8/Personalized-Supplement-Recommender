"""
Graph Builder - LangGraph Workflow Construction

Builds the agentic workflow using LangGraph.
Defines nodes, edges, and entry point.

Pipeline:
  entity_extractor → entity_normalizer → supervisor →
  [safety_check | deficiency_check | recommendation] → supervisor (loop) →
  synthesis → END
"""

from langgraph.graph import StateGraph, END  # END is LangGraph's own end marker (NodeNames has no END)

from src.workflow.state import ConversationState, InputState
from src.workflow.routing import NodeNames, route_supervisor_decision
from src.agents.entity_extractor import entity_extractor
from src.agents.entity_normalizer import entity_normalizer
from src.agents.supervisor import supervisor_agent
from src.agents.synthesis import synthesis_agent
from src.tools.safety_check import safety_check
from src.tools.recommendation import recommendation
from src.tools.deficiency_check import deficiency_check


def build_workflow():
    """
    Build and compile the LangGraph workflow.

    Active nodes:
        entity_extractor → entity_normalizer → supervisor →
        [safety_check | deficiency_check | recommendation] → supervisor (loop) →
        synthesis → END

    Returns:Compiled LangGraph workflow
    """
    print("🏗️  Building workflow graph...")

    workflow = StateGraph(ConversationState, input=InputState)

    # ==================== NODES: PRE-PROCESSING, SUPERVISOR, SPECIALISTS ====================

    print(f"   Adding node: {NodeNames.ENTITY_EXTRACTOR}")
    workflow.add_node(NodeNames.ENTITY_EXTRACTOR, entity_extractor)

    print(f"   Adding node: {NodeNames.ENTITY_NORMALIZER}")
    workflow.add_node(NodeNames.ENTITY_NORMALIZER, entity_normalizer)
    
    print(f"   Adding node: {NodeNames.SUPERVISOR}")
    workflow.add_node(NodeNames.SUPERVISOR, supervisor_agent)

    print(f"   Adding node: {NodeNames.SAFETY_CHECK}")
    workflow.add_node(NodeNames.SAFETY_CHECK, safety_check)

    print(f"   Adding node: {NodeNames.RECOMMENDATION}")
    workflow.add_node(NodeNames.RECOMMENDATION, recommendation)

    print(f"   Adding node: {NodeNames.DEFICIENCY_CHECK}")
    workflow.add_node(NodeNames.DEFICIENCY_CHECK, deficiency_check)

    # ==================== PHASE 3: SYNTHESIS NODE ====================
    print(f"   Adding node: {NodeNames.SYNTHESIS}")
    workflow.add_node(NodeNames.SYNTHESIS, synthesis_agent)

    # ==================== EDGES ====================

    print(f"   Edge: {NodeNames.ENTITY_EXTRACTOR} → {NodeNames.ENTITY_NORMALIZER}")
    workflow.add_edge(NodeNames.ENTITY_EXTRACTOR, NodeNames.ENTITY_NORMALIZER)

    print(f"   Edge: {NodeNames.ENTITY_NORMALIZER} → {NodeNames.SUPERVISOR}")
    workflow.add_edge(NodeNames.ENTITY_NORMALIZER, NodeNames.SUPERVISOR)

    print(f"   Conditional edge: {NodeNames.SUPERVISOR} → specialists / END")
    workflow.add_conditional_edges(
        NodeNames.SUPERVISOR,
        route_supervisor_decision,
        {
            NodeNames.SAFETY_CHECK: NodeNames.SAFETY_CHECK,
            NodeNames.RECOMMENDATION: NodeNames.RECOMMENDATION,
            NodeNames.DEFICIENCY_CHECK: NodeNames.DEFICIENCY_CHECK,
            NodeNames.SYNTHESIS: NodeNames.SYNTHESIS,
            # END: END,
            
        }
    )

    workflow.add_edge(NodeNames.SAFETY_CHECK, NodeNames.SUPERVISOR)
    workflow.add_edge(NodeNames.RECOMMENDATION, NodeNames.SUPERVISOR)
    workflow.add_edge(NodeNames.DEFICIENCY_CHECK, NodeNames.SUPERVISOR)

    # ==================== EDGES: PHASE 3 ====================
    workflow.add_edge(NodeNames.SYNTHESIS, END)

    # ==================== ENTRY POINT ====================

    print(f"   Entry point: {NodeNames.ENTITY_EXTRACTOR}")
    workflow.set_entry_point(NodeNames.ENTITY_EXTRACTOR)

    # ==================== COMPILE ====================

    print("   Compiling workflow...")
    compiled = workflow.compile()

    print("✅ Workflow built successfully!\n")
    return compiled