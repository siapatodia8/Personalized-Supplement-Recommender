"""
State Definition - Conversation State

Defines the shared state that flows between all agents in the workflow.
All agents read from and write to this state.
"""

from typing import TypedDict, List, Dict, Any, Optional
# from langgraph.graph import add_messages



class InputState(TypedDict):
    """
    The only fields the user provides at the start.
    LangGraph Studio shows only these two fields as inputs.
    """
    user_question: str
    patient_profile: Dict[str, Any]


# Default patient profile template shown in LangGraph Studio input
DEFAULT_PATIENT_PROFILE = {
    "medications": "",
    "supplements": "",
    "conditions": [],
    "dietary_restrictions": []
}


class ConversationState(TypedDict):
    """
    Shared state passed between all nodes in the workflow.
    LangGraph automatically manages state updates and passing.
    """

    # ==================== USER INPUTS ====================

    user_question: str

    patient_profile: Dict[str, Any]

    # ==================== ENTITY EXTRACTION ====================

    entities_extracted: bool

    extracted_entities: Optional[Dict[str, List[str]]]

    # ==================== ENTITY NORMALIZATION ====================

    entities_normalized: bool

    normalized_medications: Optional[List[Dict[str, Any]]]

    normalized_supplements: Optional[List[Dict[str, Any]]]

    normalized_dietary_restrictions: Optional[List[Dict[str, Any]]]

    # Clean deduplicated lists for downstream agents to consume
    medications_list: List[str]

    supplements_list: List[str]

    candidate_supplements_list: List[str]

    dietary_restrictions_list: List[str]

    conditions_list: List[str]

    # ==================== SUPERVISOR CONTROL ====================

    supervisor_decision: str

    iterations: int

    confidence_level: float

    # ==================== SPECIALIST RESULTS ====================
    # Populated by specialist tools

    safety_checked: bool
    safety_results: Optional[Dict[str, Any]]

    generated_safety_queries: List[Dict[str, Any]]
    """
    Stores LLM-generated Cypher queries for inspection — one entry per supplement.
    Visible in LangGraph Studio state panel for debugging.
    Each entry: { supplement, cypher, explanation, executed, result_count, error }
    """

    deficiency_checked: bool
    deficiency_results: Optional[Dict[str, Any]]

    recommendations_checked: bool
    recommendation_results: Optional[Dict[str, Any]]

    # ==================== EVIDENCE & LOGGING ====================

    evidence_chain: List[str]

    query_history: List[Dict]

    # ==================== FINAL OUTPUT ====================

    final_answer: Optional[str]

    error_message: Optional[str]

    # ==================== MESSAGES ====================

    # messages: Annotated[List[Dict], add_messages]
    messages: List[Dict]
    """
    Chat message history, kept as a plain list (no automatic appending).
    Format: [{'role': 'user', 'content': '...'}, ...]
    """


# ==================== INITIAL STATE ====================

def create_initial_state(
    user_question: str,
    patient_profile: Dict[str, Any],
) -> ConversationState:
    """
    Create a fresh state for a new conversation.

    Args:
        user_question: The user's question
        patient_profile: Raw profile from sidebar

    Returns:
        ConversationState with all defaults set
    """
    return ConversationState(
        # User inputs
        user_question=user_question,
        patient_profile=patient_profile,

        # Entity extraction
        entities_extracted=False,
        extracted_entities=None,

        # Entity normalization
        entities_normalized=False,
        normalized_medications=None,
        normalized_supplements=None,
        normalized_dietary_restrictions=None,

        # Clean lists for agents
        medications_list=[],
        supplements_list=[],
        candidate_supplements_list=[],
        dietary_restrictions_list=[],
        conditions_list=[],

        # Supervisor control
        supervisor_decision="",
        iterations=0,
        confidence_level=0.0,

        # Specialist results
        safety_checked=False,
        safety_results=None,
        generated_safety_queries=[],
        deficiency_checked=False,
        deficiency_results=None,
        recommendations_checked=False,
        recommendation_results=None,

        # Evidence & logging
        evidence_chain=[],
        query_history=[],

        # Final output
        final_answer=None,
        error_message=None,

        # Messages
        messages=[{"role": "user", "content": user_question}],
    )


# ==================== HELPER FUNCTIONS ====================

def get_state_summary(state: ConversationState) -> str:
    """Human-readable summary of current state — useful for debugging."""
    return f"""
State Summary
-------------
Question   : {state['user_question']}
Iterations : {state.get('iterations', 0)}
Confidence : {state.get('confidence_level', 0):.2f}

Pipeline Progress:
  Entities Extracted  : {'✓' if state.get('entities_extracted') else '✗'}
  Entities Normalized : {'✓' if state.get('entities_normalized') else '✗'}
  Safety Checked      : {'✓' if state.get('safety_checked') else '✗'}
  Deficiency Checked  : {'✓' if state.get('deficiency_checked') else '✗'}
  Recommendations     : {'✓' if state.get('recommendations_checked') else '✗'}

Clean Lists:
  Medications          : {state.get('medications_list', [])}
  Supplements          : {state.get('supplements_list', [])}
  Dietary Restrictions : {state.get('dietary_restrictions_list', [])}
  Conditions           : {state.get('conditions_list', [])}

Supervisor Decision : {state.get('supervisor_decision', 'None')}
Evidence Steps      : {len(state.get('evidence_chain', []))}
Queries Made        : {len(state.get('query_history', []))}
Final Answer        : {'Generated' if state.get('final_answer') else 'Not yet'}
"""