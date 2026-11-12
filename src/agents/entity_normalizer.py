"""
Entity Normalizer - Pipeline Node 2

Maps extracted entity names to database IDs using LLM-generated Cypher queries.
Uses schema.py to give the LLM accurate database context so it generates
valid Cypher rather than hallucinating property or node names.

After normalization, performs a deep deduplication by database ID —
catching cases the extractor missed, e.g. 'folic acid' and 'folate'
both resolving to the same supplement_id.

Type: Agent
- LLM reads schema to understand database structure
- LLM generates Cypher dynamically for each entity
- Hardcoded fallback only for NOT_FOUND cases
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from src.graph.connections import graph_interface, schema_provider
from src.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


# ==================== CYPHER GENERATION ====================

def _generate_normalization_query(
    entity_name: str,
    entity_type: str,
    schema_str: str,
    client: Anthropic
) -> str:
    """
    Use LLM to generate a Cypher query that finds a database match
    for the given entity name.

    Args:
        entity_name: User's input e.g. 'folic acid', 'Advil', 'B12'
        entity_type: 'medication' or 'supplement'
        schema_str: Output of SchemaProvider.to_prompt_string()
        client: Anthropic client

    Returns:
        Cypher query string
    """
    prompt = load_prompt("entity_normalizer")["primary"].format(
        entity_type=entity_type, entity_name=entity_name, schema_str=schema_str
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        cypher = response.content[0].text.strip()
        # Strip markdown fences if present
        cypher = re.sub(r'^```(?:cypher)?\s*', '', cypher)
        cypher = re.sub(r'\s*```$', '', cypher)
        return cypher.strip()
    except Exception as e:
        logger.error(f"Cypher generation failed for '{entity_name}': {e}")
        return ""


# ==================== SINGLE ENTITY NORMALIZATION ====================
def _normalize_entity(
    entity_name: str,
    entity_type: str,
    schema_str: str,
    graph_interface,
    client: Anthropic
) -> Dict[str, Any]:
    """
    Version A: Fallback prompt uses schema_str for context.
    LLM has full schema and reasons about which paths to try.
    """
    print(f"   🔍 Normalizing {entity_type}: '{entity_name}'")

    # Step 1: Generate and execute primary query
    cypher = _generate_normalization_query(entity_name, entity_type, schema_str, client)

    if cypher:
        try:
            results = graph_interface.execute_query(
                cypher,
                {"entity_name": entity_name}
            )

            if results:
                if len(results) == 1:
                    print(f"      ✅ Matched: '{entity_name}' → '{results[0].get('name')}'")
                    return {
                        'user_input': entity_name,
                        'matched_name': results[0].get('name'),
                        'db_id': results[0].get('id'),
                        'confidence': 'HIGH',
                        'match_type': 'llm_generated_query'
                    }

                best = min(results, key=lambda r: len(str(r.get('name', ''))))
                print(f"      ⚠️  Multiple matches for '{entity_name}', using closest: '{best.get('name')}'")
                return {
                    'user_input': entity_name,
                    'matched_name': best.get('name'),
                    'db_id': best.get('id'),
                    'confidence': 'MEDIUM',
                    'match_type': 'llm_generated_query_multiple',
                    'all_matches': [r.get('name') for r in results]
                }

        except Exception as e:
            logger.warning(f"Query execution failed for '{entity_name}': {e}")

    # Step 2: Fallback — full schema_str, LLM reasons about broader paths
    print(f"      🔄 No results for '{entity_name}', trying schema-aware fallback...")

    fallback_prompt = load_prompt("entity_normalizer")["fallback"].format(
        entity_name=entity_name, entity_type=entity_type, schema_str=schema_str
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": fallback_prompt}]
        )
        fallback_cypher = response.content[0].text.strip()
        fallback_cypher = re.sub(r'^```(?:cypher)?\s*', '', fallback_cypher)
        fallback_cypher = re.sub(r'\s*```$', '', fallback_cypher)

        results = graph_interface.execute_query(
            fallback_cypher.strip(),
            {"entity_name": entity_name}
        )

        if results:
            best = min(results, key=lambda r: len(str(r.get('name', ''))))
            print(f"      ✅ Fallback matched: '{entity_name}' → '{best.get('name')}'")
            return {
                'user_input': entity_name,
                'matched_name': best.get('name'),
                'db_id': best.get('id'),
                'confidence': 'LOW',
                'match_type': 'llm_fallback_query'
            }

    except Exception as e:
        logger.warning(f"Fallback query failed for '{entity_name}': {e}")

    # Step 3: Not found
    print(f"      ❌ Not found in database: '{entity_name}'")
    return {
        'user_input': entity_name,
        'matched_name': None,
        'db_id': None,
        'confidence': 'NOT_FOUND',
        'match_type': 'none'
    }


# ==================== DEEP DEDUPLICATION ====================

def _dedup_by_db_id(normalized_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate normalized entities by database ID.

    This catches cases the extractor's shallow string dedup missed —
    e.g. 'folic acid' and 'folate' both resolving to supplement_id 'S12'.

    Strategy:
    - NOT_FOUND entries are kept as-is (no ID to dedup on)
    - For entries with the same db_id, keep the one with higher confidence
      (HIGH > MEDIUM > LOW)
    - If confidence is equal, keep the first occurrence

    Args:
        normalized_list: List of normalization result dicts

    Returns:
        Deduplicated list
    """
    confidence_rank = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NOT_FOUND': 0}

    seen_ids: Dict[str, Dict] = {}   # db_id -> best entry so far
    no_id_entries: List[Dict] = []   # NOT_FOUND entries — keep all

    for entry in normalized_list:
        db_id = entry.get('db_id')

        if not db_id:
            # NOT_FOUND — nothing to dedup on, keep it
            no_id_entries.append(entry)
            continue

        if db_id not in seen_ids:
            seen_ids[db_id] = entry
        else:
            # Duplicate ID found — keep higher confidence entry
            existing = seen_ids[db_id]
            existing_rank = confidence_rank.get(existing.get('confidence', 'LOW'), 0)
            new_rank = confidence_rank.get(entry.get('confidence', 'LOW'), 0)

            if new_rank > existing_rank:
                print(
                    f"   🔁 Dedup: '{entry['user_input']}' and "
                    f"'{existing['user_input']}' → same DB ID '{db_id}', "
                    f"keeping '{entry['user_input']}' (higher confidence)"
                )
                seen_ids[db_id] = entry
            else:
                print(
                    f"   🔁 Dedup: '{entry['user_input']}' and "
                    f"'{existing['user_input']}' → same DB ID '{db_id}', "
                    f"keeping '{existing['user_input']}'"
                )

    return list(seen_ids.values()) + no_id_entries


# ==================== CLEAN LIST BUILDER ====================

def _build_clean_list(normalized_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extract final names from normalized list for downstream agents.

    Only includes entries with confidence HIGH or MEDIUM.
    NOT_FOUND and LOW confidence entries are excluded from clean lists
    since downstream agents rely on these names to query the database.

    Args:
        normalized_list: Deduplicated normalized entity list

    Returns:
        List of matched name strings e.g. ['Warfarin', 'Fish oil']
    """
    include = {'HIGH', 'MEDIUM'}
    return [
        entry['matched_name']
        for entry in normalized_list
        if entry.get('confidence') in include and entry.get('matched_name')
    ]


# ==================== LANGGRAPH NODE ====================

def entity_normalizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Normalize extracted entities to database IDs.

    Reads from state:
        - extracted_entities: raw entities from entity_extractor
        - graph_interface: Neo4j connection
        - schema_provider: SchemaProvider instance

    Writes to state:
        - normalized_medications
        - normalized_supplements
        - normalized_dietary_restrictions
        - medications_list: clean deduped medication names
        - supplements_list: clean deduped supplement names
        - dietary_restrictions_list: clean dietary restriction names
        - conditions_list: clean condition names (pass-through, no DB mapping)
        - entities_normalized: True
        - evidence_chain: appended with normalization summary

    Args:
        state: Current ConversationState

    Returns:
        Partial state update dict
    """
    print("\n" + "=" * 60)
    print("🧬 ENTITY NORMALIZER: Mapping entities to database IDs...")
    print("=" * 60)

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    extracted = state.get('extracted_entities', {})
    schema_str = schema_provider.to_prompt_string()

    # ── Normalize medications ──
    raw_medications = extracted.get('medications', [])
    print(f"\n   Normalizing {len(raw_medications)} medication(s): {raw_medications}")
    normalized_meds = [
        _normalize_entity(name, 'medication', schema_str, graph_interface, client)
        for name in raw_medications
    ]
    normalized_meds = _dedup_by_db_id(normalized_meds)

    # ── Normalize supplements ──
    raw_supplements = extracted.get('supplements', [])
    print(f"\n   Normalizing {len(raw_supplements)} supplement(s): {raw_supplements}")
    normalized_supps = [
        _normalize_entity(name, 'supplement', schema_str, graph_interface, client)
        for name in raw_supplements
    ]
    normalized_supps = _dedup_by_db_id(normalized_supps)

    # ── Dietary restrictions — pass through, no DB mapping needed ──
    dietary_restrictions = extracted.get('dietary_restrictions', [])

    # ── Conditions — pass through, no DB mapping needed ──
    conditions = extracted.get('conditions', [])

    # ── Build clean lists for downstream agents ──
    medications_list = _build_clean_list(normalized_meds)
    supplements_list = _build_clean_list(normalized_supps)

    print(f"\n   ✅ Normalization complete:")
    print(f"      Medications : {medications_list}")
    print(f"      Supplements : {supplements_list}")
    print(f"      Restrictions: {dietary_restrictions}")
    print(f"      Conditions  : {conditions}")

    # ── Evidence entry ──
    normalization_evidence = (
        f"Normalized — medications: {medications_list}, "
        f"supplements: {supplements_list}, "
        f"conditions: {conditions}, "
        f"dietary restrictions: {dietary_restrictions}"
    )

    print("=" * 60 + "\n")

    return {
    'normalized_medications': normalized_meds,
    'normalized_supplements': normalized_supps,
    'normalized_dietary_restrictions': dietary_restrictions,
    'medications_list': medications_list,
    'supplements_list': supplements_list,
    'dietary_restrictions_list': dietary_restrictions,
    'conditions_list': conditions,
    'entities_normalized': True,
    'evidence_chain': state.get('evidence_chain', []) + [normalization_evidence]
    }