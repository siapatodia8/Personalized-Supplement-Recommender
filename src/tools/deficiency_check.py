"""
Deficiency Check - Deficiency Specialist

Identifies nutrient deficiency risks from three sources:

1. DIET_DEFICIENCY_QUERY      — DietaryRestriction -[DEFICIENT_IN]-> Nutrient
2. MEDICATION_DEPLETION_QUERY — Drug -[INTERACTS_WITH_NUTRIENT]-> Nutrient
3. SUPPLEMENT_DEPLETION_QUERY — Supplement -[NEGATIVE_INTERACTION]-> Nutrient

Results from all three pathways are merged and cross-referenced.
Nutrients at risk from multiple sources are flagged as critical overlaps.

Scoring is deliberately simple: criticality is count-based (2+ sources on the same
nutrient = critical overlap) and severity labels are carried along, not summed into
a score. Earlier versions used hand-set confidence numbers and a risk_multiplier;
both were dropped. How best to weigh severity is still to be decided.

Reads from state : dietary_restrictions_list, medications_list, supplements_list
Writes to state  : deficiency_checked, deficiency_results, evidence_chain
"""

import logging
from typing import Any, Dict, List, Tuple

from src.graph.connections import graph_interface

logger = logging.getLogger(__name__)


# ==================== CYPHER QUERIES ====================

# STALE CODE (the ORDER BY at the end of this query): meant to sort HIGH, then MEDIUM, then
# the rest. The graph stores mixed-case labels ('High', 'Moderate', 'Low to Moderate', ...),
# so no row matches and rows sort by nutrient name only. Which nutrients are returned is unaffected.
DIET_DEFICIENCY_QUERY = """
MATCH (dr:DietaryRestriction)-[r:DEFICIENT_IN]->(n:Nutrient)
WHERE toLower(dr.dietary_restriction_name) IN $restrictions_lower
RETURN dr.dietary_restriction_name AS source,
       n.nutrient_name AS nutrient,
       n.category AS nutrient_category,
       r.risk_level AS risk_level,
       'diet' AS source_type
ORDER BY
    CASE r.risk_level
        WHEN 'HIGH' THEN 0
        WHEN 'MEDIUM' THEN 1
        ELSE 2
    END,
    n.nutrient_name
"""

MEDICATION_DEPLETION_QUERY = """
MATCH (d:Drug)-[r:INTERACTS_WITH_NUTRIENT]->(n:Nutrient)
WHERE toLower(d.drug_name) IN $medication_names_lower
RETURN d.drug_name AS source,
       n.nutrient_name AS nutrient,
       n.category AS nutrient_category,
       r.interaction_type AS risk_level,
       'medication' AS source_type
"""

SUPPLEMENT_DEPLETION_QUERY = """
MATCH (s:Supplement)-[r:NEGATIVE_INTERACTION]->(n:Nutrient)
WHERE toLower(s.supplement_name) IN $supplement_names_lower
RETURN s.supplement_name AS source,
       n.nutrient_name AS nutrient,
       n.category AS nutrient_category,
       r.severity AS risk_level,
       r.mechanism AS mechanism,
       'supplement' AS source_type
"""

# Turns the stored interaction verb into a risk label. Covers the six verbs in the data;
# any other spelling falls back to MEDIUM (e.g. the one 'deletes' row: Ketorolac -> Vitamin C).
INTERACTION_TYPE_RISK = {
    'depletes': 'HIGH',
    'antagonizes': 'HIGH',
    'interferes_with_absorption': 'MEDIUM',
    'increases_level': 'MEDIUM',
    'may_cause_loss': 'MEDIUM',
    'redistributes': 'LOW',
}

# NOTE (known drift): keys are uppercase. Medication and supplement labels match them, but
# diet labels from the graph are mixed-case ('High', 'Moderate', ...) and fall back to 3
# (lowest) when _aggregate looks up the highest individual risk.
RISK_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'MODERATE': 2, 'LOW': 3}


def _check_diet(restrictions: List[str]) -> List[Dict[str, Any]]:
    """
    Query for nutrient deficiencies from dietary restrictions.

    Args:
        restrictions: dietary_restrictions_list from state. Plain text as extracted, not
            matched to graph names first: each must equal a DietaryRestriction name
            (Vegan, Vegetarian, Pescatarian, Keto, Gluten-Free, Dairy-Free), ignoring case.

    Returns:
        List of deficiency records with source_type='diet'
    """
    if not restrictions:
        return []

    try:
        rows = graph_interface.execute_query(
            DIET_DEFICIENCY_QUERY,
            {'restrictions_lower': [r.lower() for r in restrictions]}
        )
    except Exception as e:
        logger.error(f"Diet deficiency query failed: {e}")
        return []

    deficiencies = []
    for row in rows or []:
        deficiencies.append({
            'nutrient': row.get('nutrient'),
            'nutrient_category': row.get('nutrient_category'),
            'source_type': 'diet',
            'source_name': row.get('source'),
            'risk_level': row.get('risk_level', 'MEDIUM'),
            'mechanism': 'dietary exclusion',
            'evidence': f"{row.get('source')} diet is deficient in {row.get('nutrient')}",
        })
        print(f"      ✅ Diet: {row.get('source')} → {row.get('nutrient')} "
              f"({row.get('risk_level', 'MEDIUM')})")

    return deficiencies


def _check_medications(medications: List[str]) -> List[Dict[str, Any]]:
    """
    Query for nutrient depletion from medications.

    Args:
        medications: medications_list from state

    Returns:
        List of deficiency records with source_type='medication'
    """
    if not medications:
        return []

    try:
        rows = graph_interface.execute_query(
            MEDICATION_DEPLETION_QUERY,
            {'medication_names_lower': [m.lower() for m in medications]}
        )
    except Exception as e:
        logger.error(f"Medication depletion query failed: {e}")
        return []

    deficiencies = []
    for row in rows or []:
        # NOTE: the query aliases r.interaction_type AS risk_level, so 'risk_level' here is the
        # raw verb (e.g. 'depletes'); INTERACTION_TYPE_RISK turns it into the real risk label.
        interaction_type = row.get('risk_level', '')
        risk_level = INTERACTION_TYPE_RISK.get(interaction_type, 'MEDIUM')

        deficiencies.append({
            'nutrient': row.get('nutrient'),
            'nutrient_category': row.get('nutrient_category'),
            'source_type': 'medication',
            'source_name': row.get('source'),
            'risk_level': risk_level,
            'mechanism': interaction_type,
            'evidence': f"{row.get('source')} {interaction_type} {row.get('nutrient')}",
        })
        print(f"      ✅ Medication: {row.get('source')} → {row.get('nutrient')} "
              f"({interaction_type})")

    return deficiencies


def _check_supplements(supplements: List[str]) -> List[Dict[str, Any]]:
    """
    Query for nutrient depletion from current supplements.

    Args:
        supplements: supplements_list from state

    Returns:
        List of deficiency records with source_type='supplement'
    """
    if not supplements:
        return []

    try:
        rows = graph_interface.execute_query(
            SUPPLEMENT_DEPLETION_QUERY,
            {'supplement_names_lower': [s.lower() for s in supplements]}
        )
    except Exception as e:
        logger.error(f"Supplement depletion query failed: {e}")
        return []

    deficiencies = []
    for row in rows or []:
        deficiencies.append({
            'nutrient': row.get('nutrient'),
            'nutrient_category': row.get('nutrient_category'),
            'source_type': 'supplement',
            'source_name': row.get('source'),
            'risk_level': row.get('risk_level', 'MEDIUM'),
            'mechanism': row.get('mechanism', ''),
            'evidence': f"{row.get('source')} negatively interacts with {row.get('nutrient')}",
        })
        print(f"      ✅ Supplement: {row.get('source')} → {row.get('nutrient')} "
              f"({row.get('risk_level', 'MEDIUM')})")

    return deficiencies


# ==================== AGGREGATION + OVERLAP DETECTION ====================

def _aggregate(
    diet_def: List[Dict],
    med_def: List[Dict],
    supp_def: List[Dict]
) -> Tuple[Dict[str, List], List[Dict]]:
    """
    Merge all deficiencies by nutrient and detect critical overlaps.

    A critical overlap is when 2+ sources affect the same nutrient —
    the combined risk is higher than any single source alone.

    Count-based on purpose: any overlap is labeled CRITICAL and severity is not summed.
    overlap_type only records how many different source types are involved
    (TRIPLE_OVERLAP = diet + medication + supplement, DOUBLE_OVERLAP = two types,
    SINGLE_SOURCE_MULTIPLE = several records from one type).

    Args:
        diet_def: Diet pathway results
        med_def: Medication pathway results
        supp_def: Supplement pathway results

    Returns:
        (all_at_risk, critical_overlaps)
        - all_at_risk: nutrient -> list of source records
        - critical_overlaps: list of overlap dicts for nutrients with 2+ sources
    """
    all_at_risk: Dict[str, List] = {}

    for deficiency in diet_def + med_def + supp_def:
        nutrient = deficiency.get('nutrient')
        if not nutrient:
            continue
        if nutrient not in all_at_risk:
            all_at_risk[nutrient] = []
        all_at_risk[nutrient].append({
            'source_type': deficiency['source_type'],
            'source_name': deficiency['source_name'],
            'risk_level': deficiency['risk_level'],
            'mechanism': deficiency['mechanism'],
        })

    # Detect overlaps
    critical_overlaps = []
    for nutrient, sources in all_at_risk.items():
        if len(sources) < 2:
            continue

        source_types = {s['source_type'] for s in sources}
        if len(source_types) == 3:
            overlap_type = 'TRIPLE_OVERLAP'
        elif len(source_types) == 2:
            overlap_type = 'DOUBLE_OVERLAP'
        else:
            overlap_type = 'SINGLE_SOURCE_MULTIPLE'

        highest_risk = min(
            sources,
            key=lambda s: RISK_ORDER.get(s['risk_level'], 3)
        )['risk_level']

        # STALE CODE (the 'overlap_type', 'combined_risk' and 'highest_individual_risk' fields):
        # meant to describe how severe an overlap is. Synthesis only reads 'nutrient' and
        # 'warning'; the other fields only reach the supervisor as text in its prompt. Kept as is.
        critical_overlaps.append({
            'nutrient': nutrient,
            'sources': sources,
            'source_names': [s['source_name'] for s in sources],
            'overlap_type': overlap_type,
            'combined_risk': 'CRITICAL',
            'highest_individual_risk': highest_risk,
            'warning': (
                f"{nutrient} is affected by {len(sources)} sources: "
                f"{', '.join(s['source_name'] for s in sources)}"
            )
        })
        print(f"      🚨 CRITICAL OVERLAP: {nutrient} affected by "
              f"{[s['source_name'] for s in sources]}")

    return all_at_risk, critical_overlaps


# ==================== RESULT BUILDER ====================

def _build_results(
    diet_def: List[Dict],
    med_def: List[Dict],
    supp_def: List[Dict],
    all_at_risk: Dict[str, List],
    critical_overlaps: List[Dict],
    restrictions_checked: List[str],
    medications_checked: List[str],
    supplements_checked: List[str],
) -> Dict[str, Any]:
    """
    Package the pathway results, merged nutrients and overlaps into one results dict.

    status is 'found' or 'not_found'; summary is a one-line description of the outcome.
    """

    total = len(all_at_risk)
    critical_count = len(critical_overlaps)

    # STALE CODE: meant to build the 'nothing_to_check' result; unused because
    # deficiency_check() exits early and builds that dict itself when all three lists are empty.
    if not restrictions_checked and not medications_checked and not supplements_checked:
        status = 'nothing_to_check'
        summary = 'No dietary restrictions, medications, or supplements to analyze'
    elif total == 0:
        status = 'not_found'
        summary = 'No nutrient deficiency risks detected'
    else:
        status = 'found'
        parts = []
        if diet_def:
            parts.append(f"{len(diet_def)} from diet")
        if med_def:
            parts.append(f"{len(med_def)} from medications")
        if supp_def:
            parts.append(f"{len(supp_def)} from supplements")
        summary = (
            f"Found {total} nutrient(s) at risk ({', '.join(parts)})"
            + (f" — {critical_count} critical overlap(s)" if critical_count else "")
        )

    return {
        'specialist': 'deficiency',
        'status': status,
        'entities_checked': restrictions_checked + medications_checked + supplements_checked,
        'summary': summary,
        'diet_based': diet_def,
        'medication_based': med_def,
        'supplement_based': supp_def,
        'all_at_risk': list(all_at_risk.keys()),
        'all_at_risk_details': all_at_risk,
        'critical_overlaps': critical_overlaps,
        'total_count': total,
        'critical_count': critical_count,
        'restrictions_checked': restrictions_checked,
        'medications_checked': medications_checked,
        'supplements_checked': supplements_checked,
    }


# ==================== LANGGRAPH NODE ====================

def deficiency_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Check nutrient deficiency risks across three pathways.

    Reads from state:
        - dietary_restrictions_list: patient's dietary restrictions
        - medications_list: patient's medications
        - supplements_list: patient's current supplements

    Writes to state:
        - deficiency_checked: True
        - deficiency_results: structured results (per-pathway findings, merged
          nutrients at risk, critical overlaps, one-line summary)
        - evidence_chain: appended with deficiency finding

    Args:
        state: Current ConversationState

    Returns:
        Partial state update dict
    """
    print("\n" + "=" * 60)
    print("🥗 DEFICIENCY CHECK: Analyzing nutrient gaps...")
    print("=" * 60)

    restrictions = state.get('dietary_restrictions_list', [])
    medications = state.get('medications_list', [])
    supplements = state.get('supplements_list', [])

    # ── Early exit: nothing to check ──
    if not restrictions and not medications and not supplements:
        print("   ⚠️  No dietary restrictions, medications, or supplements to analyze")
        results = {
            'specialist': 'deficiency',
            'status': 'nothing_to_check',
            'entities_checked': [],
            'summary': 'No dietary restrictions, medications, or supplements to analyze',
            'diet_based': [], 'medication_based': [], 'supplement_based': [],
            'all_at_risk': [], 'all_at_risk_details': {},
            'critical_overlaps': [], 'total_count': 0, 'critical_count': 0,
            'restrictions_checked': [], 'medications_checked': [], 'supplements_checked': [],
        }
        return {
            'deficiency_checked': True,
            'deficiency_results': results,
            'evidence_chain': state.get('evidence_chain', []) + [
                'Deficiency check: skipped — nothing to analyze'
            ]
        }

    print(f"   Dietary restrictions : {restrictions if restrictions else 'none'}")
    print(f"   Medications          : {medications if medications else 'none'}")
    print(f"   Supplements          : {supplements if supplements else 'none'}")

    # ── Run all three pathways ──
    print(f"\n   Pathway 1: Diet-based deficiencies")
    diet_def = _check_diet(restrictions)
    if not diet_def:
        print(f"      ⊘  None found")

    print(f"\n   Pathway 2: Medication-induced depletions")
    med_def = _check_medications(medications)
    if not med_def:
        print(f"      ⊘  None found")

    print(f"\n   Pathway 3: Supplement-induced depletions")
    supp_def = _check_supplements(supplements)
    if not supp_def:
        print(f"      ⊘  None found")

    # ── Aggregate and detect overlaps ──
    print(f"\n   Aggregating and checking for critical overlaps...")
    all_at_risk, critical_overlaps = _aggregate(diet_def, med_def, supp_def)

    # ── Build results ──
    results = _build_results(
        diet_def, med_def, supp_def,
        all_at_risk, critical_overlaps,
        restrictions, medications, supplements
    )

    print(f"\n   {'⚠️  Deficiencies found' if results['total_count'] > 0 else '✅ No deficiencies found'}: "
          f"{results['summary']}")
    print("=" * 60 + "\n")

    return {
        'deficiency_checked': True,
        'deficiency_results': results,
        'evidence_chain': state.get('evidence_chain', []) + [
            f"Deficiency check: {results['summary']}"
        ]
    }