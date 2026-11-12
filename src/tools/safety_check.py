"""
Safety Check - Interaction Specialist (Agentic)

LLM reads the knowledge graph schema and generates a Cypher query to find
ALL dangerous interaction paths between a supplement and medications.
No hardcoded pathways — the LLM discovers paths from the schema itself.

Workflow per supplement:
  1. Generate Cypher via LLM (schema-aware UNION query)
  2. Validate syntax via EXPLAIN (pre-execution gate)
  3. Execute against Neo4j
  4. Group results by `pathway` field
  5. Store execution metadata in generated_safety_queries
  6. Accumulate interactions across all supplements

Reads from state : supplements_list, candidate_supplements_list, medications_list
Writes to state  : safety_checked, safety_results, generated_safety_queries, evidence_chain
"""

import logging
import os
import re
from typing import Any, Dict, List

from anthropic import Anthropic

from src.graph.connections import graph_interface, schema_provider
from src.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


# ==================== QUERY GENERATOR ====================

def _generate_query(
    supplement: str,
    medications: List[str],
    schema_str: str,
    client: Anthropic,
) -> Dict[str, str]:
    """
    Ask LLM to generate a Cypher query covering all dangerous interaction
    paths it can find in the schema for the given supplement + medications.

    Returns dict with:
        cypher      — the raw Cypher query string
        explanation — one-sentence plain English description of what it checks
    """
    prompt = load_prompt("safety_check")["generate"].format(
        supplement=supplement,
        medications=medications,
        schema_str=schema_str,
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
    except Exception as e:
        logger.error(f"LLM query generation failed for '{supplement}': {e}")
        return {"cypher": "", "explanation": f"Generation failed: {e}"}

    # ── Parse cypher and explanation ──
    cypher = ""
    explanation = ""

    if "EXPLANATION:" in raw:
        parts = raw.split("EXPLANATION:", 1)
        cypher = parts[0].strip()
        explanation = parts[1].strip()
    else:
        cypher = raw
        explanation = "No explanation provided"

    # Strip markdown fences if present
    cypher = re.sub(r'^```(?:cypher)?\s*', '', cypher)
    cypher = re.sub(r'\s*```$', '', cypher)
    cypher = cypher.strip()

    # Fix common LLM parameter syntax error — backticks instead of dollar sign
    cypher = re.sub(r'`(supplement_lower|medications_lower)', r'$\1', cypher)

    return {"cypher": cypher, "explanation": explanation}


# ==================== QUERY EXECUTOR ====================

def _execute_query(
    supplement: str,
    medications: List[str],
    cypher: str,
) -> Dict[str, Any]:
    """
    Validate and execute a generated Cypher query for one supplement.

    Steps:
        1. validate_query() via EXPLAIN — catches syntax and planning errors
        2. execute_query() with correct parameter names
        3. Group results by pathway field

    Args:
        supplement: Supplement name (used to build parameters)
        medications: Medication names (used to build parameters)
        cypher: LLM-generated Cypher query string

    Returns dict with:
        results             — flat list of interaction rows (empty if none)
        by_pathway          — rows grouped by pathway field
        pathways_with_results — list of pathway names that returned rows
        result_count        — total rows returned
        error               — error string if failed, None if success
    """
    parameters = {
        'supplement_lower': supplement.lower(),
        'medications_lower': [m.lower() for m in medications],
    }

    # ── Step 1: Validate syntax before execution ──
    is_valid = graph_interface.validate_query(cypher)
    if not is_valid:
        error_msg = "Query failed syntax/planning validation (EXPLAIN)"
        logger.error(f"Safety query invalid for '{supplement}': {error_msg}")
        return {
            'results': [],
            'by_pathway': {},
            'pathways_with_results': [],
            'result_count': 0,
            'error': error_msg,
        }

    # ── Step 2: Execute ──
    try:
        rows = graph_interface.execute_query(cypher, parameters)
        rows = rows or []
    except Exception as e:
        error_msg = f"Execution error: {e}"
        logger.error(f"Safety query execution failed for '{supplement}': {e}")
        return {
            'results': [],
            'by_pathway': {},
            'pathways_with_results': [],
            'result_count': 0,
            'error': error_msg,
        }

    # ── Step 3: Group by pathway — only non-empty pathways ──
    by_pathway: Dict[str, List] = {}
    for row in rows:
        pathway = row.get('pathway', 'UNKNOWN')
        by_pathway.setdefault(pathway, []).append(row)

    return {
        'results': rows,
        'by_pathway': by_pathway,
        'pathways_with_results': list(by_pathway.keys()),
        'result_count': len(rows),
        'error': None,
    }


# ==================== RESULT BUILDER ====================

def _build_results(
    all_interactions: List[Dict[str, Any]],
    all_by_pathway: Dict[str, List],
    supplements_checked: List[str],
    medications_checked: List[str],
    has_errors: bool = False,
) -> Dict[str, Any]:
    """
    Build the safety_results dict from accumulated interactions.

    Args:
        all_interactions: Flat list of all interaction rows across supplements
        all_by_pathway: Rows grouped by pathway name across supplements
        supplements_checked: All supplement names that were checked
        medications_checked: All medication names checked against

    Returns:
        Structured safety_results dict for synthesis consumption
    """
    if all_interactions:
        status = 'found'
        pathway_summary = ', '.join(
            f"{pathway}: {len(records)}"
            for pathway, records in all_by_pathway.items()
        )
        summary = (
            f"Found {len(all_interactions)} interaction(s) across "
            f"{len(all_by_pathway)} pathway(s) "
            f"between {', '.join(supplements_checked)} and "
            f"{', '.join(medications_checked)} "
            f"({pathway_summary})"
        )
    elif has_errors:
        # Queries failed — absence of interactions does not mean safe
        status = 'error'
        summary = (
            f"Safety check failed — query errors prevented interaction checking "
            f"for {', '.join(supplements_checked)}. "
            f"Safety status is unknown, do not assume safe."
        )
    else:
        status = 'not_found'
        summary = (
            f"No interactions found between "
            f"{', '.join(supplements_checked)} and "
            f"{', '.join(medications_checked)}"
        )

    return {
        'specialist': 'safety',
        'status': status,
        'supplements_checked': supplements_checked,
        'medications_checked': medications_checked,
        'entities_checked': supplements_checked + medications_checked,
        'summary': summary,
        'interactions': all_interactions,
        'by_pathway': all_by_pathway,
    }


# ==================== LANGGRAPH NODE ====================

def safety_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Generate and execute safety queries.

    For each supplement:
        1. Generate Cypher via LLM (schema-aware UNION query)
        2. Validate syntax via EXPLAIN
        3. Execute against Neo4j with correct parameters
        4. Group results by pathway field
        5. Store execution metadata in generated_safety_queries

    Reads from state:
        - supplements_list: patient's current supplements
        - candidate_supplements_list: candidates from recommendation
        - medications_list: patient's medications

    Writes to state:
        - safety_checked: True
        - safety_results: structured interaction findings
        - generated_safety_queries: one entry per supplement with execution metadata
        - evidence_chain: appended with safety finding
    """
    print("\n" + "=" * 60)
    print("🔬 SAFETY CHECK: Generating and executing interaction queries...")
    print("=" * 60)

    current_supplements = state.get('supplements_list', [])
    candidate_supplements = state.get('candidate_supplements_list', [])
    supplements = list(dict.fromkeys(current_supplements + candidate_supplements))
    medications = state.get('medications_list', [])

    # ── Early exit: no supplements ──
    if not supplements:
        print("   ⚠️  No supplements to check")
        return {
            'safety_checked': True,
            'generated_safety_queries': [],
            'safety_results': {
                'specialist': 'safety',
                'status': 'no_supplements',
                'entities_checked': [],
                'summary': 'No supplements identified to check',
                'interactions': [],
                'by_pathway': {},
                'supplements_checked': [],
                'medications_checked': [],
            },
            'evidence_chain': state.get('evidence_chain', []) + [
                'Safety check: skipped — no supplements to check'
            ]
        }

    # ── Early exit: no medications ──
    if not medications:
        print("   ⚠️  No medications to check against")
        return {
            'safety_checked': True,
            'generated_safety_queries': [],
            'safety_results': {
                'specialist': 'safety',
                'status': 'no_medications',
                'entities_checked': supplements,
                'summary': 'No medications to check against',
                'interactions': [],
                'by_pathway': {},
                'supplements_checked': supplements,
                'medications_checked': [],
            },
            'evidence_chain': state.get('evidence_chain', []) + [
                'Safety check: skipped — no medications to check against'
            ]
        }

    if candidate_supplements:
        print(f"   Current supplements  : {current_supplements or 'none'}")
        print(f"   Candidate supplements: {candidate_supplements}")
        print(f"   Checking combined    : {supplements}")
    else:
        print(f"   Supplements : {supplements}")
    print(f"   Medications : {medications}")

    # ── Set up LLM and schema ──
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    schema_str = schema_provider.to_prompt_string()

    # ── Accumulators across all supplements ──
    all_interactions: List[Dict[str, Any]] = []
    all_by_pathway: Dict[str, List] = {}
    generated_queries: List[Dict[str, Any]] = []
    has_errors: bool = False

    # ── Process each supplement ──
    for supplement in supplements:
        print(f"\n   Processing: {supplement}")

        # Step 1: Generate
        print(f"   Generating query...")
        gen = _generate_query(supplement, medications, schema_str, client)
        cypher = gen["cypher"]
        explanation = gen["explanation"]

        entry: Dict[str, Any] = {
            'supplement': supplement,
            'medications': medications,
            'cypher': cypher,
            'explanation': explanation,
            'executed': False,
            'result_count': None,
            'pathways_with_results': [],
            'error': None,
        }

        if not cypher:
            print(f"   ❌ Query generation failed")
            entry['error'] = 'Empty query returned by LLM'
            generated_queries.append(entry)
            has_errors = True
            continue

        print(f"   ✅ Query generated")
        print(f"   📋 Explanation: {explanation}")

        # Step 2 + 3: Validate and execute
        print(f"   Executing query...")
        execution = _execute_query(supplement, medications, cypher)

        entry['executed'] = True
        entry['result_count'] = execution['result_count']
        entry['pathways_with_results'] = execution['pathways_with_results']
        entry['error'] = execution['error']

        if execution['error']:
            print(f"   ❌ Execution failed: {execution['error']}")
            generated_queries.append(entry)
            has_errors = True
            continue

        # Step 4: Accumulate results
        rows = execution['results']
        by_pathway = execution['by_pathway']

        if rows:
            for pathway, records in by_pathway.items():
                print(f"      ⚠️  [{pathway}] — {len(records)} interaction(s)")
                for ix in records:
                    print(f"         {ix.get('supplement')} ↔ {ix.get('target')}: "
                          f"{str(ix.get('description', ''))[:80]}")
                all_by_pathway.setdefault(pathway, []).extend(records)
            all_interactions.extend(rows)
        else:
            print(f"   ✅ No interactions found")

        generated_queries.append(entry)

    # ── Build final results ──
    results = _build_results(
        all_interactions, all_by_pathway, supplements, medications, has_errors
    )

    print(f"\n   {'⚠️  Interactions found' if all_interactions else '✅ No interactions found'}: "
          f"{results['summary']}")
    print("=" * 60 + "\n")

    return {
        'safety_checked': True,
        'generated_safety_queries': generated_queries,
        'safety_results': results,
        'evidence_chain': state.get('evidence_chain', []) + [
            f"Safety check: {results['summary']}"
        ]
    }