"""
Entity Extractor - Pipeline Node 1

Extracts medications, supplements, conditions, and dietary restrictions
from two sources:
1. User's natural language question  (LLM)
2. Patient profile sidebar form      (simple string parsing)

Merges both sources, deduplicates, and writes clean raw lists to state.
Normalization to database IDs happens in the next node: entity_normalizer.py

Type: Hybrid
- LLM call for unstructured natural language input
- Deterministic string parsing for structured profile form input
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from anthropic import Anthropic

from src.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


# ==================== LLM EXTRACTION ====================

def _extract_from_question(question: str, client: Anthropic) -> Dict[str, List[str]]:
    """
    Use LLM to extract entities from a natural language question.

    Handles:
    - Brand names and generics ('Advil' → medication)
    - Implicit conditions ('heart health' → condition)
    - Dietary mentions ('I am vegan' → dietary restriction)
    - Typos are kept as-is here — normalizer handles correction

    Args:
        question: User's natural language question
        client: Anthropic client

    Returns:
        {
            'medications': ['Warfarin'],
            'supplements': ['Fish Oil'],
            'conditions': ['heart health'],
            'dietary_restrictions': ['vegan']
        }
    """
    prompt = load_prompt("entity_extractor")["extraction"].format(question=question)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present (newer models wrap JSON in ```json ... ```)
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"medications": [], "supplements": [], "conditions": [], "dietary_restrictions": []}


# ==================== PROFILE PARSING ====================

def _parse_profile(profile: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Parse structured patient profile from sidebar form.

    Profile fields are already categorized by the user — no LLM needed.
    Just split comma-separated strings and normalize list formats.

    Args:
        profile: {
            'medications': 'Warfarin, Metformin',   # string or list
            'supplements': 'Fish Oil, Vitamin D',   # string or list
            'conditions': ['Diabetes'],             # string or list
            'dietary_restrictions': ['Vegan']       # string or list
        }

    Returns:
        {
            'medications': ['Warfarin', 'Metformin'],
            'supplements': ['Fish Oil', 'Vitamin D'],
            'conditions': ['Diabetes'],
            'dietary_restrictions': ['Vegan']
        }
    """
    def to_list(value) -> List[str]:
        """Convert string or list to clean list of strings."""
        if not value:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    return {
        'medications': to_list(profile.get('medications', '')),
        'supplements': to_list(profile.get('supplements', '')),
        'conditions': to_list(profile.get('conditions', [])),
        'dietary_restrictions': to_list(profile.get('dietary_restrictions', []))
    }


# ==================== MERGE + DEDUP ====================

def _merge(
    from_question: Dict[str, List[str]],
    from_profile: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Merge entities from question and profile, deduplicating case-insensitively.

    Profile entries take priority — if 'Fish Oil' is in profile and
    'fish oil' appears in question, keep 'Fish Oil'.

    Args:
        from_question: Entities extracted from natural language question
        from_profile: Entities parsed from structured profile

    Returns:
        Merged and deduplicated entity dict
    """
    merged = {}
    for key in ['medications', 'supplements', 'conditions', 'dietary_restrictions']:
        seen = {}  # lowercase -> preferred name
        # Profile first (takes priority in casing)
        for item in from_profile.get(key, []):
            seen[item.lower()] = item
        # Question second (only add if not already seen)
        for item in from_question.get(key, []):
            if item.lower() not in seen:
                seen[item.lower()] = item
        merged[key] = list(seen.values())
    return merged


# ==================== LANGGRAPH NODE ====================

def entity_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Extract entities from question + profile.

    Reads from state:
        - user_question
        - patient_profile

    Writes to state:
        - extracted_entities: raw merged entities (pre-normalization)
        - entities_extracted: True
        - evidence_chain: appended with extraction summary

    Args:
        state: Current ConversationState

    Returns:
        Partial state update dict
    """
    print("\n" + "=" * 60)
    print("🔍 ENTITY EXTRACTOR: Extracting entities...")
    print("=" * 60)

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Step 1: Extract from natural language question
    question = state.get('user_question', '')
    print(f"   Question: {question}")
    from_question = _extract_from_question(question, client)
    print(f"   From question: {from_question}")

    # Step 2: Parse structured profile
    profile = state.get('patient_profile', {})
    from_profile = _parse_profile(profile)
    print(f"   From profile: {from_profile}")

    # Step 3: Merge and deduplicate
    merged = _merge(from_question, from_profile)
    print(f"   Merged: {merged}")

    # Step 4: Build evidence entry
    extraction_evidence = (
        f"Extracted — medications: {merged['medications']}, "
        f"supplements: {merged['supplements']}, "
        f"conditions: {merged['conditions']}, "
        f"dietary restrictions: {merged['dietary_restrictions']}"
    )

    print(f"   ✅ Extraction complete")
    print("=" * 60 + "\n")

    return {
        'extracted_entities': merged,
        'evidence_chain': [extraction_evidence],
        'iterations': 0,
        'supervisor_decision': '',
        'candidate_supplements_list': [],
        'safety_checked': False,
        'deficiency_checked': False,
        'recommendations_checked': False,
        'safety_results': None,
        'deficiency_results': None,
        'recommendation_results': None,
        'final_answer': None,
        'generated_safety_queries': [],
    }