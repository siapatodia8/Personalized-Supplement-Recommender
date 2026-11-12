"""
Recommendation - Recommendation Specialist

Finds supplements that treat the patient's conditions/symptoms.
Safety checking of candidates is handled separately by safety_check.py
after the supervisor routes back through the loop.

Query flow:
1. SYMPTOM_QUERY  — find supplements via TREATS relationship (substring match on
                    the symptom name, so "pain" also matches "back pain")
2. BROAD_QUERY    — fallback keyword search if the first query finds nothing
                    (any single word of the condition, capped at 10 results)

Current supplements are excluded from candidates — no point recommending
what the patient already takes.

Reads from state : conditions_list, supplements_list
Writes to state  : recommendations_checked, recommendation_results,
                   candidate_supplements_list, evidence_chain
                   (candidate_supplements_list is what safety_check.py reads)

No LLM is used here: both queries are fixed Cypher with parameters, so the
results are the same every run.
"""

import logging
from typing import Any, Dict, List

from src.graph.connections import graph_interface

logger = logging.getLogger(__name__)


# ==================== CYPHER QUERIES ====================

# Find supplements that treat a condition — substring match on the symptom name (CONTAINS)

SYMPTOM_QUERY = """
MATCH (s:Supplement)-[:TREATS]->(sym:Symptom)
WHERE toLower(sym.symptom_name) CONTAINS $condition_lower
RETURN DISTINCT
    s.supplement_id AS supplement_id,
    s.supplement_name AS supplement_name,
    s.safety_rating AS safety_rating,
    sym.symptom_name AS symptom_treated
ORDER BY s.supplement_name
"""

# Fallback — search by individual keywords when the first query finds nothing

BROAD_QUERY = """
MATCH (s:Supplement)-[:TREATS]->(sym:Symptom)
WHERE ANY(word IN $words WHERE toLower(sym.symptom_name) CONTAINS word)
RETURN DISTINCT
    s.supplement_id AS supplement_id,
    s.supplement_name AS supplement_name,
    s.safety_rating AS safety_rating,
    sym.symptom_name AS symptom_treated
ORDER BY s.supplement_name
LIMIT 10
"""


# ==================== CANDIDATE FINDER ====================

def _find_candidates(condition: str) -> List[Dict[str, Any]]:
    """
    Find supplements that treat the given condition.

    Tries a substring match on the symptom name first, falls back to a
    keyword search if no results are found (or the first query errors).

    Args:
        condition: Condition or symptom string from conditions_list

    Returns:
        List of candidate dicts with supplement_id, supplement_name,
        safety_rating, symptom_treated
    """
    condition_lower = condition.lower().strip()

    # Primary: substring match on the symptom name
    try:
        results = graph_interface.execute_query(
            SYMPTOM_QUERY,
            {'condition_lower': condition_lower}
        )
        if results:
            print(f"      ✅ Found {len(results)} candidates for '{condition}'")
            return results
    except Exception as e:
        logger.error(f"Symptom query failed for '{condition}': {e}")

    # Fallback: keyword search
    words = [w for w in condition_lower.split() if len(w) > 3]
    if not words:
        return []

    print(f"      🔄 No exact match for '{condition}', trying keyword search: {words}")
    try:
        results = graph_interface.execute_query(
            BROAD_QUERY,
            {'words': words}
        )
        if results:
            print(f"      ✅ Keyword search found {len(results)} candidates")
            return results
    except Exception as e:
        logger.error(f"Broad query failed for '{condition}': {e}")

    return []


# ==================== RESULT BUILDER ====================

def _build_results(
    candidates: List[Dict[str, Any]],
    conditions: List[str],
    medications: List[str]
) -> Dict[str, Any]:
    """
    Package candidates into the compact dict the supervisor and synthesis read.

    status is 'found' or 'not_found'; the summary ends with "safety check
    pending" because safety_check.py checks these candidates afterwards.

    STALE CODE: the `medications` parameter is always passed as [] by
    recommendation(), so 'entities_checked' holds conditions only. It was meant
    for a medication cross-check, which now lives in safety_check.py.
    """
    if not candidates:
        status = 'not_found'
        summary = f"No supplement candidates found for: {', '.join(conditions)}"
    else:
        status = 'found'
        summary = (
            f"Found {len(candidates)} supplement candidate(s) for "
            f"{', '.join(conditions)} — safety check pending"
        )

    return {
        'specialist': 'recommendation',
        'status': status,
        'entities_checked': conditions + medications,
        'summary': summary,
        'recommendations': candidates,
        'candidate_supplements': [c.get('supplement_name') for c in candidates],
        'conditions_checked': conditions,
    }


# ==================== LANGGRAPH NODE ====================

def recommendation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Find candidate supplements for the patient's conditions.

    Only finds and ranks candidates; the safety check on them runs afterwards
    in safety_check.py, once the supervisor routes there.

    Reads from state:
        - conditions_list: health conditions / symptoms from question + profile
        - medications_list: NOT read here. STALE: it was meant for a safety
          cross-check, which now lives in safety_check.py
        - supplements_list: patient's current supplements (excluded from candidates)

    Writes to state:
        - recommendations_checked: True
        - recommendation_results: compact structured summary
        - candidate_supplements_list: candidate names, read by safety_check.py
          (not written on the no-conditions early exit; state default is [])
        - evidence_chain: appended with recommendation finding

    Args:
        state: Current ConversationState

    Returns:
        Partial state update dict
    """
    print("\n" + "=" * 60)
    print("💊 RECOMMENDATION: Finding supplement options...")
    print("=" * 60)

    conditions = state.get('conditions_list', [])
    current_supplements = state.get('supplements_list', [])
    current_supplements_lower = {s.lower() for s in current_supplements}

    # ── Early exit: no conditions to address ──
    if not conditions:
        print("   ⚠️  No conditions to find recommendations for")
        results = {
            'specialist': 'recommendation',
            'status': 'no_conditions',
            'entities_checked': [],
            'summary': 'No conditions or symptoms to find recommendations for',
            'recommendations': [],
            'candidate_supplements': []
        }
        return {
            'recommendations_checked': True,
            'recommendation_results': results,
            'evidence_chain': state.get('evidence_chain', []) + [
                'Recommendation check: skipped — no conditions to address'
            ]
        }

    print(f"   Conditions  : {conditions}")
    print(f"   Current supplements (excluded): "
          f"{current_supplements if current_supplements else 'none'}")

    # ── Find candidates for each condition (skip duplicates and current supplements) ──
    seen_supplements: set = set()
    all_evaluated: List[Dict[str, Any]] = []

    for condition in conditions:
        print(f"\n   Searching for: '{condition}'")
        candidates = _find_candidates(condition)

        if not candidates:
            print(f"      ❌ No candidates found")
            continue

        for candidate in candidates:
            name = candidate.get('supplement_name', '')

            # Skip if already seen or is a current supplement
            if not name or name in seen_supplements:
                continue
            if name.lower() in current_supplements_lower:
                print(f"      ⏭️  Skipping '{name}' — already in current supplements")
                continue

            seen_supplements.add(name)
            all_evaluated.append(candidate)
            print(f"      ✅ {name} (safety_rating: {candidate.get('safety_rating', 'unknown')})")

    # ── Sort by safety_rating (best first; unknown ratings rank like 'Not recommended') ──
    safety_rating_rank = {'Generally safe': 2, 'Use with caution': 1, 'Not recommended': 0}
    all_evaluated.sort(
        key=lambda r: safety_rating_rank.get(r.get('safety_rating', ''), 0),
        reverse=True
    )

    # ── Build compact results ──
    # [] = medications: unused (STALE CODE, see _build_results)
    results = _build_results(all_evaluated, conditions, [])

    print(f"\n   {results['summary']}")
    print("=" * 60 + "\n")

    return {
        'recommendations_checked': True,
        'recommendation_results': results,
        'candidate_supplements_list': results['candidate_supplements'],
        'evidence_chain': state.get('evidence_chain', []) + [
            f"Recommendation check: {results['summary']}"
        ]
    }